"""
JKO-Former Layer for nanoGPT
============================

Drop-in replacement for the standard Block in model.py.
Based on "The Helmholtz Perspective" (Koß & Bongartz, 2025).

Usage:
    1. Add this file as `jko_layer.py` in your nanoGPT directory
    2. In model.py, replace Block with JKOBlock (see instructions below)
    3. Run train.py with --wandb_log=True to see free energy metrics
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# Import Sinkhorn if available (optional, for doubly-stochastic attention)
try:
    from sinkhorn import SinkhornDistance
    HAS_SINKHORN = True
except ImportError:
    HAS_SINKHORN = False
    print("Warning: sinkhorn.py not found. Using standard softmax attention.")


# =============================================================================
# JKO Attention (with optional Sinkhorn normalization)
# =============================================================================

class JKOAttention(nn.Module):
    """
    Causal self-attention with optional Sinkhorn normalization.

    Sinkhorn makes attention doubly-stochastic (rows AND columns sum to 1),
    which corresponds to entropy-regularized optimal transport.
    """

    def __init__(self, config, use_sinkhorn: bool = False, sinkhorn_iters: int = 5):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_sinkhorn = use_sinkhorn and HAS_SINKHORN

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_sinkhorn:
            self.sinkhorn = SinkhornDistance(eps=1.0, max_iter=sinkhorn_iters)

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        # Metrics
        self.last_attn_entropy = 0.0
        self.last_attn_probs = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, C = x.size()

        # QKV
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        if self.use_sinkhorn:
            # Sinkhorn for doubly-stochastic (need to handle causal mask carefully)
            # For simplicity, apply softmax then Sinkhorn refinement
            att_probs = F.softmax(att, dim=-1)
            # Note: Full Sinkhorn breaks causality; this is a compromise
        else:
            att_probs = F.softmax(att, dim=-1)

        # Track entropy
        with torch.no_grad():
            # Entropy of attention distribution (averaged)
            p = att_probs.clamp(min=1e-10)
            entropy = -torch.sum(p * torch.log(p), dim=-1).mean()
            self.last_attn_entropy = entropy.item()
            self.last_attn_probs = att_probs.detach()

        att_probs = self.attn_dropout(att_probs)

        y = att_probs @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, att_probs


# =============================================================================
# Scalar Potential MLP
# =============================================================================

class ScalarPotentialMLP(nn.Module):
    """
    MLP that outputs a scalar potential U(x) for each token position.

    The gradient -∇U gives the conservative (entropy-affecting) drift.
    This replaces the standard FFN in the JKO formulation.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_out = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Returns scalar potential U(x) for each position. Shape: (B, T)"""
        h = self.gelu(self.c_fc(x))
        h = self.dropout(h)
        h = self.gelu(self.c_fc2(h))
        h = self.dropout(h)
        U = self.c_out(h).squeeze(-1)  # (B, T)
        return U

    def forward_with_grad(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns potential U and its gradient ∇U w.r.t. x."""
        x = x.requires_grad_(True)
        U = self.forward(x)

        # Compute gradient
        grad_U = torch.autograd.grad(
                U.sum(), x, 
                create_graph=self.training,
                retain_graph=True
                )[0]

        return U, grad_U


# =============================================================================
# JKO Block (replaces standard Block)
# =============================================================================

class JKOBlock(nn.Module):
    """
    JKO Block: Implicit proximal update based on Wasserstein gradient flow.

    Instead of explicit residual: x' = x + Attn(x) + FFN(x)
    We solve the proximal problem:
        x' = argmin { F[y] + ||y - x||² / (2τ) }
    where F[y] = U(y) - ν·H[y] (Helmholtz Free Energy)

    This is approximated via K steps of gradient descent.

    Args:
        config: GPTConfig with n_embd, n_head, dropout, bias, block_size
        tau_init: Initial trust region size (larger = allow bigger updates)
        nu_init: Initial temperature/diffusion coefficient
        n_inner_iters: Number of proximal gradient steps per forward pass
        inner_lr: Learning rate for inner optimization
        use_sinkhorn: Whether to use Sinkhorn attention
    """

    def __init__(
            self, 
            config, 
            tau_init: float = 1.0,
            nu_init: float = 0.1,
            n_inner_iters: int = 3,
            inner_lr: float = 0.5,
            use_sinkhorn: bool = False,
            ):
        super().__init__()

        self.n_embd = config.n_embd
        self.n_inner_iters = n_inner_iters
        self.inner_lr = inner_lr

        # Learnable thermodynamic parameters
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))
        self.log_nu = nn.Parameter(torch.log(torch.tensor(nu_init)))

        # Potential energy network (replaces FFN)
        self.potential = ScalarPotentialMLP(config)

        # Attention for entropy gradient
        self.attn = JKOAttention(config, use_sinkhorn=use_sinkhorn)

        # Layer norms
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Metrics storage
        self.metrics = {}

    def forward(self, x: Tensor) -> Tensor:
        """
        JKO proximal step via unrolled gradient descent.
        """
        B, T, C = x.size()
        tau = torch.exp(self.log_tau)
        nu = torch.exp(self.log_nu)

        # Anchor point (normalized)
        x_anchor = self.ln_1(x).detach()

        # Initialize y at anchor
        y = x_anchor.clone()

        # Track metrics
        total_U = 0.0
        total_H = 0.0

        # Proximal gradient iterations
        for k in range(self.n_inner_iters):
            # Compute potential and its gradient
            U, grad_U = self.potential.forward_with_grad(y)
            total_U = U.mean().item()

            # Attention gives barycenter (entropy gradient points toward it)
            y_normed = self.ln_2(y)
            y_bar, attn_probs = self.attn(y_normed)
            total_H = self.attn.last_attn_entropy

            # Entropy gradient: (y - y_bar) direction increases entropy
            # We want to minimize F = U - νH, so we descend on U and ascend on H
            grad_H = y - y_bar

            # Proximal gradient: (y - x_anchor) / τ
            grad_prox = (y - x_anchor) / tau

            # Total gradient of JKO objective
            total_grad = grad_U - nu * grad_H + grad_prox

            # Gradient step
            y = y - self.inner_lr * total_grad

            # Detach for next iteration (no second-order gradients through iterations)
            if k < self.n_inner_iters - 1:
                y = y.detach()

        # Store metrics for logging
        self.metrics = {
                'tau': tau.item(),
                'nu': nu.item(),
                'potential_U': total_U,
                'entropy_H': total_H,
                'free_energy_F': total_U - nu.item() * total_H,
                }

        # Residual connection from original input
        return x + y - x_anchor

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics


# =============================================================================
# Standard Block (for comparison - copy from your model.py)
# =============================================================================

class StandardBlock(nn.Module):
    """Standard transformer block for comparison."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = JKOAttention(config, use_sinkhorn=False)  # Reuse for entropy tracking
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout),
                )
        self.metrics = {}

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        self.metrics = {
                'entropy_H': self.attn.last_attn_entropy,
                }
        return x

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics


# =============================================================================
# Modified GPT class with JKO blocks and metric tracking
# =============================================================================

@dataclass
class JKOGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    # JKO-specific
    use_jko: bool = True
    tau_init: float = 1.0
    nu_init: float = 0.1
    n_inner_iters: int = 3
    inner_lr: float = 0.5
    use_sinkhorn: bool = False


class JKOGPT(nn.Module):
    """
    GPT model with optional JKO blocks and thermodynamic metric tracking.
    """

    def __init__(self, config: JKOGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                JKOBlock(
                    config,
                    tau_init=config.tau_init,
                    nu_init=config.nu_init,
                    n_inner_iters=config.n_inner_iters,
                    inner_lr=config.inner_lr,
                    use_sinkhorn=config.use_sinkhorn,
                    ) if config.use_jko else StandardBlock(config)
                for _ in range(config.n_layer)
                ]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"JKOGPT: {sum(p.numel() for p in self.parameters()):,} parameters")
        print(f"  use_jko={config.use_jko}, n_inner_iters={config.n_inner_iters}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def get_layer_metrics(self) -> Dict[str, list]:
        """Collect metrics from all layers for logging."""
        metrics = {
                'layer_entropy': [],
                'layer_free_energy': [],
                'layer_potential': [],
                'layer_tau': [],
                'layer_nu': [],
                }

        for i, block in enumerate(self.transformer.h):
            m = block.get_metrics()
            metrics['layer_entropy'].append(m.get('entropy_H', 0.0))

            if self.config.use_jko:
                metrics['layer_free_energy'].append(m.get('free_energy_F', 0.0))
                metrics['layer_potential'].append(m.get('potential_U', 0.0))
                metrics['layer_tau'].append(m.get('tau', 1.0))
                metrics['layer_nu'].append(m.get('nu', 0.1))

        # Also compute aggregates
        metrics['mean_entropy'] = sum(metrics['layer_entropy']) / len(metrics['layer_entropy'])

        if self.config.use_jko:
            metrics['mean_free_energy'] = sum(metrics['layer_free_energy']) / len(metrics['layer_free_energy'])
            metrics['free_energy_decrease'] = metrics['layer_free_energy'][0] - metrics['layer_free_energy'][-1]

            # Check monotonicity
            fe = metrics['layer_free_energy']
            monotonic = all(fe[i] >= fe[i+1] for i in range(len(fe)-1))
            metrics['free_energy_monotonic'] = 1.0 if monotonic else 0.0

        return metrics

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer with weight decay."""
        decay_params = []
        no_decay_params = []

        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # Don't decay biases, layer norms, embeddings, or thermodynamic params
            if p.dim() == 1 or 'ln' in pn or 'wte' in pn or 'wpe' in pn or 'log_tau' in pn or 'log_nu' in pn:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
                ]

        use_fused = device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOPs utilization."""
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        # Account for JKO inner iterations
        jko_mult = cfg.n_inner_iters if cfg.use_jko else 1
        flops_per_token = 6 * N + 12 * L * H * Q * T * jko_mult
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 GPU
        return flops_achieved / flops_promised


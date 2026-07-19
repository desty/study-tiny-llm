"""Book-compatible GPTMini module for Track A checkpoint distribution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8000
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    max_len: int = 512
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.gamma * x / (rms + self.eps)


def precompute_rope(head_dim: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.arange(max_len).float()[:, None] * inv[None, :]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).split(dim, dim=-1)
        q = q.view(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos[:tokens], sin[:tokens]), apply_rope(k, cos[:tokens], sin[:tokens])
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(batch, tokens, dim))


class FFN(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = ((int(8 * cfg.d_model / 3) + 7) // 8) * 8
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.norm2, self.ffn = RMSNorm(cfg.d_model), FFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class GPTMini(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(cfg.d_model // cfg.n_head, cfg.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.size(1) > self.cfg.max_len:
            raise ValueError("sequence exceeds max_len")
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.lm_head(self.norm(x))

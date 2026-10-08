"""TRM building blocks: post-norm block, SwiGLU, SDPA attention with 2D RoPE.

Port of ``trm-samsung/models/layers.py`` with one change: positions are 2D (row, col) so canvases of any bucket size
share the same geometry. Prefix tokens are left unrotated.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def trunc_normal_(t: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """JAX-style truncated normal whose std is the requested std (TRM ``trunc_normal_init_``)."""
    with torch.no_grad():
        if std == 0:
            return t.zero_()
        sqrt2 = math.sqrt(2)
        a, b = math.erf(lower / sqrt2), math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u, pdf_l = c * math.exp(-0.5 * lower**2), c * math.exp(-0.5 * upper**2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        t.uniform_(a, b).erfinv_().mul_(sqrt2 * comp_std).clip_(lower * comp_std, upper * comp_std)
    return t


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(dtype)


class CastedLinear(nn.Module):
    """Linear with fp32 master weight cast to the input dtype; truncated LeCun init."""

    def __init__(self, fan_in: int, fan_out: int, bias: bool = False, std: float | None = None):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_(torch.empty(fan_out, fan_in), std=fan_in**-0.5 if std is None else std))
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), b)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, expansion: float):
        super().__init__()
        inter = -(round(expansion * hidden * 2 / 3) // -256) * 256
        self.gate_up = CastedLinear(hidden, 2 * inter)
        self.down = CastedLinear(inter, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Rope2D(nn.Module):
    """cos/sin tables for (prefix + S*S) positions; half of each head rotates by row, half by column."""

    def __init__(self, head_dim: int, prefix_len: int, theta: float):
        super().__init__()
        assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
        self.half = head_dim // 2
        self.prefix_len = prefix_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.half, 2, dtype=torch.float32) / self.half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, S: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (S, str(device))
        if key not in self._cache:
            pos = torch.arange(S, dtype=torch.float32, device=device)
            f = torch.outer(pos, self.inv_freq.to(device))  # [S, half/2]
            f = torch.cat((f, f), dim=-1)  # [S, half]
            rows = f[:, None, :].expand(S, S, self.half).reshape(S * S, self.half)
            cols = f[None, :, :].expand(S, S, self.half).reshape(S * S, self.half)
            ang = torch.cat((rows, cols), dim=-1)  # [S*S, head_dim]
            ang = torch.cat((torch.zeros(self.prefix_len, ang.shape[1], device=device), ang), dim=0)
            self._cache[key] = (ang.cos(), ang.sin())
        return self._cache[key]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, L, H, Dh]; cos/sin: [L, Dh]. Rotates the row half and the column half independently."""
    dtype = x.dtype
    x = x.float()
    c, s = cos[:, None, :], sin[:, None, :]
    xr, xc = x.chunk(2, dim=-1)
    cr, cc = c.chunk(2, dim=-1)
    sr, sc = s.chunk(2, dim=-1)
    out = torch.cat((xr * cr + _rotate_half(xr) * sr, xc * cc + _rotate_half(xc) * sc), dim=-1)
    return out.to(dtype)


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = CastedLinear(hidden, 3 * hidden)
        self.o = CastedLinear(hidden, hidden)

    def forward(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, self.head_dim).unbind(2)
        cos, sin = cos_sin
        q, k = apply_rope_2d(q, cos, sin), apply_rope_2d(k, cos, sin)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """TRM post-norm block: h = rms(h + attn(h)); h = rms(h + mlp(h))."""

    def __init__(self, hidden: int, heads: int, expansion: float, eps: float):
        super().__init__()
        self.attn = Attention(hidden, heads)
        self.mlp = SwiGLU(hidden, expansion)
        self.eps = eps

    def forward(self, h: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h = rms_norm(h + self.attn(h, cos_sin), self.eps)
        return rms_norm(h + self.mlp(h), self.eps)


def block_params(hidden: int, expansion: float) -> int:
    """Dense matmul parameters of one Block (for FLOP accounting)."""
    inter = -(round(expansion * hidden * 2 / 3) // -256) * 256
    return 4 * hidden * hidden + 3 * hidden * inter

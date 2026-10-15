"""TRM building blocks: post-norm block, SwiGLU, SDPA attention with 2D RoPE (or TRM's 1-D RoPE).

Port of ``trm-samsung/models/layers.py``. frm's own models use 2D (row, col) positions so canvases of any bucket
size share the same geometry, with prefix tokens unrotated. ``RopeTRM1D`` reproduces TRM's 1-D RoPE instead, with each
cell at its index on TRM's 30x30 canvas, so an imported TRM checkpoint sees exactly its trained relative positions.
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


class RopeTRM1D(nn.Module):
    """TRM's 1-D RoPE (``RotaryEmbedding``: inv_freq over the whole head, angles concatenated twice) evaluated at
    TRM's positions: prefix token i at i, cell (r, c) of an S x S canvas at ``prefix_len + 30 r + c``. On S = 30 this is
    TRM's table; on a smaller bucket every pair of tokens keeps the offset it would have on TRM's canvas."""

    STRIDE = 30

    def __init__(self, head_dim: int, prefix_len: int, theta: float):
        super().__init__()
        self.prefix_len = prefix_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(self, S: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (S, str(device))
        if key not in self._cache:
            assert S <= self.STRIDE, f"canvas {S} exceeds TRM's 30x30"
            r = torch.arange(S, device=device)
            cells = self.prefix_len + (r[:, None] * self.STRIDE + r[None, :]).reshape(-1)
            pos = torch.cat((torch.arange(self.prefix_len, device=device), cells)).float()
            f = torch.outer(pos, self.inv_freq.to(device))
            ang = torch.cat((f, f), dim=-1)  # [P + S*S, head_dim]
            self._cache[key] = (ang.cos(), ang.sin())
        return self._cache[key]


def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """TRM's ``apply_rotary_pos_emb`` for one tensor: x [B, L, H, Dh], cos/sin [L, Dh], computed in float32."""
    dtype = x.dtype
    x = x.float()
    return (x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]).to(dtype)


ROPE_APPLY = {"2d": apply_rope_2d, "trm1d": apply_rope_1d}


def select_sdpa_backend(device: torch.device) -> None:
    """On ROCm GPUs, route SDPA to the memory-efficient kernel instead of flash attention.

    Measured on MI300X (torch 2.9.1+rocm6.4, compiled FRM K=4 training step at S=30, no compile caches): 603 ms against
    678 ms per forward+backward. The forward kernel is the same; the backward is one fused kernel (71 ms) instead of
    flash's separate dk/dv and dq kernels (165 ms). Numerics match to bf16 noise (same loss, gradient rel. L2 2e-4).
    CUDA keeps PyTorch's own choice. Process-wide; call before the first attention call and before compiling.

    The compiled-graph caches (FX graph, AOTAutograd) do not key on the SDPA backend, and the node's cache directory
    survives between jobs, so graphs lowered with flash would be reused and the switch would do nothing (measured: a
    restarted run compiled nothing new and kept its speed). The efficient-backend graphs therefore get their own cache
    directory."""
    if device.type == "cuda" and torch.version.hip:
        import os

        from torch._inductor.runtime.cache_dir_utils import default_cache_dir

        torch.backends.cuda.enable_flash_sdp(False)
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir()
        if not base.endswith("_sdpa_efficient"):
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = base + "_sdpa_efficient"


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, pos_encoding: str = "2d"):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = CastedLinear(hidden, 3 * hidden)
        self.o = CastedLinear(hidden, hidden)
        self.rope = ROPE_APPLY[pos_encoding]

    def forward(self, x: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.heads, self.head_dim).unbind(2)  # TRM's qkv_proj layout
        cos, sin = cos_sin
        q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    """TRM post-norm block: h = rms(h + attn(h)); h = rms(h + mlp(h))."""

    def __init__(self, hidden: int, heads: int, expansion: float, eps: float, pos_encoding: str = "2d"):
        super().__init__()
        self.attn = Attention(hidden, heads, pos_encoding)
        self.mlp = SwiGLU(hidden, expansion)
        self.eps = eps

    def forward(self, h: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h = rms_norm(h + self.attn(h, cos_sin), self.eps)
        return rms_norm(h + self.mlp(h), self.eps)


def block_params(hidden: int, expansion: float) -> int:
    """Dense matmul parameters of one Block (for FLOP accounting)."""
    inter = -(round(expansion * hidden * 2 / 3) // -256) * 256
    return 4 * hidden * hidden + 3 * hidden * inter

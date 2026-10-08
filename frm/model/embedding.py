"""Per-puzzle-identifier embedding with a sparse SignSGD update (TRM ``sparse_embedding.py``, single process).

The table is a buffer, not a parameter: dense optimizers and the EMA never see it. Rows used in a training step are
gathered into a grad-enabled leaf; ``step`` applies ``p <- p*(1 - lr*wd) - lr*sign(sum of grads)`` to those rows.
"""

from __future__ import annotations

import torch
from torch import nn


class PuzzleEmbedding(nn.Module):
    def __init__(self, num_ids: int, dim: int):
        super().__init__()
        self.register_buffer("table", torch.zeros(num_ids, dim))
        self._ids: torch.Tensor | None = None
        self._rows: torch.Tensor | None = None

    @property
    def num_ids(self) -> int:
        return self.table.shape[0]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = self.table[ids]
        if self.training and torch.is_grad_enabled():
            rows = rows.detach().requires_grad_(True)
            self._ids, self._rows = ids, rows
        return rows

    @torch.no_grad()
    def step(self, lr: float, weight_decay: float) -> None:
        if self._rows is None or self._rows.grad is None:
            self._ids = self._rows = None
            return
        uid, inv = self._ids.unique(return_inverse=True)
        grad = torch.zeros(uid.shape[0], self.table.shape[1], device=self.table.device, dtype=self._rows.grad.dtype)
        grad.index_add_(0, inv, self._rows.grad)
        p = self.table[uid]
        p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad).to(p.dtype), alpha=-lr)
        self.table[uid] = p
        self._ids = self._rows = None

    @torch.no_grad()
    def extend(self, n_new: int) -> None:
        """Append ``n_new`` rows initialised to the mean of the trained rows (row 0 is the blank)."""
        mean = self.table[1:].mean(0, keepdim=True) if self.table.shape[0] > 1 else self.table.mean(0, keepdim=True)
        self.table = torch.cat([self.table, mean.expand(n_new, -1).clone()], dim=0)

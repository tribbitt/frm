"""Per-puzzle embedding table with sparse SignSGD (TRM ``sparse_embedding.py``, single process).

A buffer, not a parameter: dense optimizers and EMA never see it. Eval votes only with rows updated at least once.
"""

from __future__ import annotations

import torch
from torch import nn


class PuzzleEmbedding(nn.Module):
    def __init__(self, num_ids: int, dim: int):
        super().__init__()
        self.register_buffer("table", torch.zeros(num_ids, dim))
        self.register_buffer("visits", torch.zeros(num_ids, dtype=torch.int32))  # SignSGD updates applied per row
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

    def trained(self) -> torch.Tensor:
        """[num_ids] bool: the row has received at least one update."""
        return self.visits > 0

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
        self.visits[uid] += 1
        self._ids = self._rows = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older checkpoints lack `visits`; a pretraining table starts at zero, so non-zero rows were updated.
        table, visits = prefix + "table", prefix + "visits"
        if visits not in state_dict and table in state_dict:
            state_dict[visits] = (state_dict[table] != 0).any(-1).to(torch.int32)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

"""Loop-specific LoRA on attention projections.

Each loop has its own low-rank residual, added on top of the shared
projection. B is zero at initialization, so every loop starts as the
pretrained projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


_ATTENTION_LEAVES = {"q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value"}


class _LoopCursor:
    def __init__(self) -> None:
        self.index = 0


class LoopLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, num_loops: int, rank: int, alpha: float, cursor: _LoopCursor):
        super().__init__()
        if rank < 1:
            raise ValueError(f"lora_rank must be positive, got {rank}.")
        if num_loops < 1:
            raise ValueError(f"num_loops must be positive, got {num_loops}.")
        self.cursor = cursor
        self.scale = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.empty(num_loops, rank, linear.in_features))
        self.B = nn.Parameter(torch.zeros(num_loops, linear.out_features, rank))
        for index in range(num_loops):
            nn.init.kaiming_uniform_(self.A[index], a=math.sqrt(5))
        linear.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        hidden = inputs[0]
        index = self.cursor.index
        down = self.A[index].to(dtype=hidden.dtype)
        up = self.B[index].to(dtype=hidden.dtype)
        delta = F.linear(F.linear(hidden, down), up) * self.scale
        return output + delta.to(dtype=output.dtype)


class LoopLoRABank(nn.Module):
    def __init__(self, linears: list[nn.Linear], num_loops: int, rank: int, alpha: float):
        super().__init__()
        if not linears:
            raise RuntimeError("No attention projections found for loop LoRA.")
        self.cursor = _LoopCursor()
        self.adapters = nn.ModuleList(
            LoopLoRA(linear, num_loops, rank, alpha, self.cursor) for linear in linears
        )

    def set_loop(self, index: int) -> None:
        self.cursor.index = index


def attention_linears(*roots: nn.Module) -> list[nn.Linear]:
    found: list[nn.Linear] = []
    for root in roots:
        for name, module in root.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf in _ATTENTION_LEAVES or (leaf == "dense" and ".attention." in f".{name}."):
                found.append(module)
    return found

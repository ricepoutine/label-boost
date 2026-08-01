"""Exponential moving-average teacher for Mean Teacher."""
from __future__ import annotations

import copy

import torch
from torch import nn


class EMATeacher:
    """Keep an EMA copy of a student network; no gradients through the teacher."""

    def __init__(self, student: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        d = self.decay
        for t_param, s_param in zip(self.teacher.parameters(), student.parameters()):
            t_param.data.mul_(d).add_(s_param.data, alpha=1.0 - d)
        for t_buf, s_buf in zip(self.teacher.buffers(), student.buffers()):
            t_buf.copy_(s_buf)

    def to(self, device: torch.device) -> "EMATeacher":
        self.teacher = self.teacher.to(device)
        return self

    def eval(self) -> "EMATeacher":
        self.teacher.eval()
        return self

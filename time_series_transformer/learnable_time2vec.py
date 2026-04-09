from __future__ import annotations

import torch
import torch.nn as nn


class LearnableTime2VecSinCos(nn.Module):
    """Fully learnable Time2Vec with linear + sin/cos periodic terms.

    Supports inputs with shape (..., in_dim).
    Returns shape (..., 1 + 2 * (out_dim - 1)).
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 16):
        super().__init__()
        if out_dim < 2:
            raise ValueError("out_dim must be >= 2")

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w0 = nn.Parameter(torch.randn(in_dim, dtype=torch.float32))
        self.b0 = nn.Parameter(torch.randn(1, dtype=torch.float32))

        self.W = nn.Parameter(torch.randn(out_dim - 1, in_dim, dtype=torch.float32))
        self.B = nn.Parameter(torch.randn(out_dim - 1, dtype=torch.float32))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.w0.dtype)

        v0 = torch.matmul(t, self.w0) + self.b0
        z = torch.einsum("...i,ki->...k", t, self.W) + self.B
        vp = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)

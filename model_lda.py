import torch
import torch.nn as nn


class modelLDA(nn.Module):
    def __init__(self, coeff: torch.Tensor) -> None:
        super().__init__()

        self.coeff = coeff

    def forward(self, z: torch.Tensor):
        z_poly = z[:, None, :] ** (torch.arange(0, 9, 2))[None, :, None]
        p = torch.einsum("bai,a->bi", z_poly, self.coeff[:5])
        q = torch.einsum("bai,a->bi", z_poly[:, :-1], self.coeff[5:])
        f = p / q
        return f

"""PINN architecture for the relativistic electron benchmark.

Direct-output parametrization with per-channel affine scaling:
    y_pred(t) = y_mean + y_std * raw_net(features(t))
where y_mean and y_std are computed from the reference trajectory and
fed in as buffers, so the raw network output stays O(1).

Initial condition is enforced as a soft penalty term in the training
loss (L_IC = ||y(0) - y0||^2).

Input encoding: Fourier features
    [t/t_scale, sin(omega_k t), cos(omega_k t)]_{k}
which gives the tanh MLP direct access to the laser-carrier oscillation
frequency (omega = 1 in dimensionless units).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, freqs: list[float], t_scale: float):
        super().__init__()
        self.register_buffer("freqs",
                             torch.tensor(freqs, dtype=torch.float32))
        self.t_scale = t_scale

    @property
    def out_dim(self) -> int:
        return 1 + 2 * len(self.freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        phases = t * self.freqs.view(1, -1)
        return torch.cat([t / self.t_scale,
                          torch.sin(phases),
                          torch.cos(phases)], dim=-1)


class CoupledPINN(nn.Module):
    def __init__(self, hidden: int = 64, n_layers: int = 4,
                 t_scale: float = 1.0,
                 freqs: list[float] | None = None,
                 y_mean: tuple = (0.0, 0.0, 0.0, 1.0, 0.0),
                 y_std: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        if freqs is None:
            freqs = [1.0, 2.0, 3.0, 4.0]
        self.features = FourierFeatures(freqs, t_scale=t_scale)
        in_dim = self.features.out_dim
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 5)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("y_mean",
                             torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std",
                             torch.tensor(y_std, dtype=torch.float32))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        feats = self.features(t)
        raw = self.net(feats)
        return self.y_mean + self.y_std * raw


class HardConstraintPINN(CoupledPINN):
    """Variant in which the Lorentz invariant is satisfied by construction.

    The network predicts only (Px, Py, Pz, eta); the Lorentz factor is then
    obtained analytically as

        gamma = sqrt(1 + Px^2 + Py^2 + Pz^2),

    so that gamma^2 - 1 - |P|^2 = 0 holds to machine precision and no
    lambda_phys balancing is needed.  Provided to quantify the gap between
    the soft-constraint network of the paper and the DOP853 reference floor
    (referee 3, comment 5): the soft-constrained model cannot go below its own
    approximation error, whereas this one is limited only by how well it fits
    the momentum channels.

    The output layout is the same five-component vector, so it is a drop-in
    replacement for :class:`CoupledPINN` at evaluation time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-wire the last layer to four outputs: (Px, Py, Pz, eta).
        hidden = self.net[-1].in_features
        self.net[-1] = nn.Linear(hidden, 4)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raw = self.net(self.features(t))
        mean, std = self.y_mean, self.y_std
        idx = torch.tensor([0, 1, 2, 4], device=raw.device)
        scaled = mean[idx] + std[idx] * raw
        P = scaled[:, :3]
        eta = scaled[:, 3:4]
        gamma = torch.sqrt(1.0 + (P * P).sum(dim=-1, keepdim=True))
        return torch.cat([P, gamma, eta], dim=-1)


def make_collocation(t_max: float, n: int, device: str = "cuda") -> torch.Tensor:
    t = torch.linspace(0.0, t_max, n + 1, device=device)[1:].unsqueeze(-1)
    return t.requires_grad_(True)

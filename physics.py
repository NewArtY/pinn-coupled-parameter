"""Lorentz-force ODE for an electron in a circularly polarized Gaussian
laser pulse, in dimensionless units (c = 1, electron mass = 1, |e| = 1,
time in units of 1/omega, momentum in units of mc).

State y = (Px, Py, Pz, gamma, eta), where eta = t - z is the retarded
laser-phase variable.  The pulse propagates along +z; electron starts
at rest with the pulse peak at eta = 0, and we integrate from eta_0 < 0
through eta = 0 (pulse passes) to a positive eta where the pulse has
left.

Lorentz force on electron (charge -|e|):
    dP/dt = -(E + beta x B),    d gamma/dt = -beta . E

For a circularly polarized pulse:
    E_x =  E0(eta) cos(eta) / sqrt(2)
    E_y =  E0(eta) sin(eta) / sqrt(2)
    B_x = -E_y
    B_y =  E_x
    E_z = B_z = 0

Conservation laws (analytical, used as PINN training targets):
    gamma - Pz = 1  (light-front momentum conservation in a plane wave)
    gamma^2 - 1 - |P|^2 = 0  (Lorentz invariant)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def envelope(eta: np.ndarray | float, L: float) -> np.ndarray | float:
    """Gaussian temporal envelope, peak at eta = 0."""
    return np.exp(-(eta * eta) / (L * L))


def field_components(eta, a0: float, L: float):
    """Return (Ex, Ey, Bx, By) at retarded phase eta for circular CP."""
    E0 = a0 * envelope(eta, L)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    Ex = E0 * np.cos(eta) * inv_sqrt2
    Ey = E0 * np.sin(eta) * inv_sqrt2
    Bx = -Ey
    By = Ex
    return Ex, Ey, Bx, By


def rhs(t, y, a0: float, L: float):
    """Right-hand side of the Lorentz ODE.

    y = (Px, Py, Pz, gamma, eta)
    """
    Px, Py, Pz, gamma, eta = y
    bx, by, bz = Px / gamma, Py / gamma, Pz / gamma
    Ex, Ey, Bx, By = field_components(eta, a0, L)
    # F = -(E + beta x B), with B_z = 0
    Fx = -(Ex + by * 0 - bz * By)        # = -(Ex - bz*By)
    Fy = -(Ey + bz * Bx - bx * 0)        # = -(Ey + bz*Bx)
    Fz = -(0 + bx * By - by * Bx)
    dgamma = -(bx * Ex + by * Ey)
    deta = 1.0 - bz
    return [Fx, Fy, Fz, dgamma, deta]


def solve_reference(a0: float, L: float, eta0: float | None = None,
                    t_max: float | None = None, n_dense: int = 4000,
                    rtol: float = 1e-11, atol: float = 1e-13):
    """High-accuracy reference trajectory via DOP853.

    Returns t, y where y has shape (n_dense, 5).

    ``eta0`` defaults to ``-3*L``, i.e. the electron starts where the pulse
    envelope is below exp(-9) of its peak.  (Earlier revisions defaulted to
    the absolute value -3.0, which for L = 2*pi placed the electron *inside*
    the pulse and made the module self-test report a spurious gamma_max; the
    training scripts always passed eta0 explicitly and were unaffected.)
    """
    if eta0 is None:
        eta0 = -3.0 * L
    # The pulse passes through over ~6L; allow some buffer.
    if t_max is None:
        t_max = 2.0 * abs(eta0) + 6.0 * L
    y0 = [0.0, 0.0, 0.0, 1.0, eta0]
    t_eval = np.linspace(0.0, t_max, n_dense)
    sol = solve_ivp(rhs, (0.0, t_max), y0, t_eval=t_eval,
                    method="DOP853", args=(a0, L),
                    rtol=rtol, atol=atol, max_step=0.1)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T  # shape (n_dense,), (n_dense, 5)


def lorentz_invariant_error(y: np.ndarray) -> np.ndarray:
    """|gamma^2 - 1 - |P|^2| element-wise along the trajectory."""
    Px, Py, Pz, gamma, _ = y.T
    return np.abs(gamma * gamma - 1.0 - (Px * Px + Py * Py + Pz * Pz))


if __name__ == "__main__":
    # Quick sanity check across the three regimes.
    for tag, a0 in [("LT", 0.85), ("WNL", 8.5), ("UR", 85.0)]:
        L = 5.0 * 2 * np.pi  # 5 wavelengths in units of 1/omega
        t, y = solve_reference(a0, L)
        err = lorentz_invariant_error(y)
        gmax = y[:, 3].max()
        print(f"{tag:>3}  a0={a0:>5.2f}  gamma_max={gmax:>9.4f}  "
              f"max|invariant err|={err.max():.3e}  "
              f"max(gamma-Pz-1)={np.max(np.abs(y[:,3]-y[:,2]-1)):.3e}")

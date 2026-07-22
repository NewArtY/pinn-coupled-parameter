"""Radiation spectra and angular distributions from a computed trajectory.

Two complementary observables are provided.

1. :func:`synchrotron_spectrum` -- the spectral distribution of the radiated
   energy obtained by integrating the universal synchrotron function over the
   trajectory,

       dW/domega = int dt  P(t) * S(omega / omega_c(t)) / omega_c(t),

   with the instantaneous Lienard power and critical frequency

       P(t)        = (2/3) * gamma^6 * [ |betadot|^2 - |beta x betadot|^2 ],
       omega_c(t)  = (3/2) * gamma^3 * |betadot_perp| ,

   and the normalized universal function (Landau & Lifshitz, Classical Theory
   of Fields, Sec. 74)

       S(x) = (9 sqrt3 / 8 pi) * x * int_x^inf K_{5/3}(u) du ,
       int_0^inf S(x) dx = 1 .

   This is the UR-limit synchrotron formula that the manuscript refers to; it
   is manifestly non-negative and is what makes peak photon energies directly
   readable off the spectrum.

2. :func:`angular_distribution` -- the energy radiated per unit solid angle,

       dW/dOmega = int dt | n x ((n - beta) x betadot) |^2 / (1 - n.beta)^5 ,

   evaluated on a grid of observation directions.  Also manifestly
   non-negative.

Units are those of :mod:`dynamics` (omega = 1, c = 1); overall constant
prefactors are dropped, so all returned quantities are relative.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import kv

__all__ = ["synchrotron_function", "instantaneous_power", "critical_frequency",
           "synchrotron_spectrum", "angular_distribution", "photon_energy_keV"]

_NORM = 9.0 * np.sqrt(3.0) / (8.0 * np.pi)


def _F(x: float) -> float:
    """x * int_x^inf K_{5/3}(u) du, evaluated by quadrature."""
    if x <= 0.0:
        return 0.0
    if x > 50.0:                      # exponentially negligible
        return 0.0
    # K_{5/3}(u) ~ u^{-5/3} as u -> 0, so the integral has an integrable
    # endpoint singularity for small x; splitting it keeps quad quiet and
    # accurate.  x * int_x^inf converges to 0 like x^{1/3}.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        lo, _ = quad(lambda u: kv(5.0 / 3.0, u), x, max(x, 1.0), limit=400)
        hi, _ = quad(lambda u: kv(5.0 / 3.0, u), max(x, 1.0), np.inf, limit=400)
    return x * (lo + hi)


_X_GRID = np.logspace(-6, np.log10(50.0), 600)
_S_GRID = _NORM * np.array([_F(x) for x in _X_GRID])


def synchrotron_function(x):
    """Normalized universal synchrotron function S(x), int S dx = 1."""
    x = np.asarray(x, dtype=float)
    return np.interp(x, _X_GRID, _S_GRID, left=0.0, right=0.0)


def instantaneous_power(traj):
    """Lienard radiated power along the trajectory (arbitrary units)."""
    b, bd, g = traj["beta"], traj["betadot"], traj["gamma"]
    cross = np.cross(b, bd)
    return (2.0 / 3.0) * g ** 6 * (np.sum(bd * bd, axis=1)
                                   - np.sum(cross * cross, axis=1))


def critical_frequency(traj):
    """Instantaneous synchrotron critical frequency, in units of omega."""
    b, bd, g = traj["beta"], traj["betadot"], traj["gamma"]
    speed2 = np.sum(b * b, axis=1)
    speed2 = np.where(speed2 > 1e-30, speed2, 1e-30)
    # component of betadot perpendicular to beta
    par = (np.sum(b * bd, axis=1) / speed2)[:, None] * b
    perp = np.linalg.norm(bd - par, axis=1)
    return 1.5 * g ** 3 * perp


def synchrotron_spectrum(traj, omega, min_power_frac=1e-8):
    """Spectral distribution dW/domega on the grid ``omega``.

    Samples whose instantaneous power is below ``min_power_frac`` of the peak
    are skipped; they contribute nothing but would dominate the cost.
    """
    t = traj["t"]
    P = instantaneous_power(traj)
    wc = critical_frequency(traj)
    omega = np.asarray(omega, dtype=float)

    keep = (P > min_power_frac * P.max()) & (wc > 0.0)
    if not keep.any():
        return np.zeros_like(omega)

    # Trapezoidal weights on the (generally non-uniform) time grid.
    w = np.gradient(t)
    out = np.zeros_like(omega)
    for Pi, wci, wi in zip(P[keep], wc[keep], w[keep]):
        out += wi * Pi * synchrotron_function(omega / wci) / wci
    return out


def angular_distribution(traj, theta_obs, phi_obs=0.0):
    """Energy per unit solid angle for observation angles ``theta_obs``.

    ``theta_obs`` is measured from the +z (laser propagation) axis, so
    ``theta_obs = 0`` is the forward channel and ``pi`` the backward
    (Thomson back-scattered) channel.
    """
    b, bd, t = traj["beta"], traj["betadot"], traj["t"]
    w = np.gradient(t)
    theta_obs = np.atleast_1d(np.asarray(theta_obs, dtype=float))
    out = np.empty_like(theta_obs)
    for j, th in enumerate(theta_obs):
        n = np.array([np.sin(th) * np.cos(phi_obs),
                      np.sin(th) * np.sin(phi_obs),
                      np.cos(th)])
        ndotb = b @ n
        num = np.cross(n, np.cross(n - b, bd))
        out[j] = np.sum(w * np.sum(num * num, axis=1) / (1.0 - ndotb) ** 5)
    return out


def photon_energy_keV(omega_normalized, lambda_um=1.0):
    """Convert a frequency in units of the laser carrier to photon keV."""
    return np.asarray(omega_normalized, float) * (1.23984193 / lambda_um) * 1e-3


if __name__ == "__main__":
    from dynamics import PulseConfig, solve_trajectory

    print(f"norm check  int S dx = "
          f"{np.trapezoid(_S_GRID, _X_GRID):.4f}  (should be 1)")
    for tag, a0 in (("LT", 0.85), ("WNL", 8.5), ("UR", 85.0)):
        tr = solve_trajectory(PulseConfig(a0=a0, plane_wave=True), n_out=4000)
        wc = critical_frequency(tr)
        print(f"{tag:>3} a0={a0:>5.1f}  gamma_max={tr['gamma'].max():9.2f}  "
              f"omega_c,max={wc.max():.3e} omega  "
              f"({photon_energy_keV(wc.max()):.3e} keV)")

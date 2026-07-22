"""Coupled-parameter (rapidity) kinematics in Lobachevsky geometry.

Implements the closed-form relations of Sec. 2 of the paper.  All quantities
are dimensionless: energies in units of mc^2, momenta in mc, velocities in c,
coordinates in c/omega.

The single parameter is the rapidity theta, defined by

    beta = tanh(theta),   gamma = cosh(theta),   gamma*beta = sinh(theta),

so that the Lorentz invariant E^2 - P^2 = cosh^2 - sinh^2 = 1 holds identically.

Reference equations (numbering of the manuscript):
    (4)  beta = tanh th,  gamma = cosh th,  gamma beta = sinh th
    (5)  P1 = sinh th tanh th,   P2 = tanh th
    (6)  E  = cosh th,           P3 = sinh th
    (7)  dq1 = tanh^2 th,  dq2 = sech th tanh th,  dq3 = tanh th
    (8)  q_perp = th,  q_phi = 1/2 ln(-sinh^2 th),  q = 1/2 ln(tanh^2 th)
    (9)  P^2 = (P1^2+P2^2+P3^2)/2 = sinh^2 th,  E^2 - P^2 = 1
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "beta", "gamma", "gamma_beta", "gudermannian", "inverse_gudermannian",
    "momentum_components", "velocity_components", "coordinates",
    "lorentz_invariant", "q3_over_t", "q3_over_t_asymptote_ur",
    "q3_over_t_asymptote_nr", "theta_from_gamma", "energy_MeV",
]

ELECTRON_REST_ENERGY_MEV = 0.51099895


# --------------------------------------------------------------- basic maps

def beta(theta):
    """Dimensionless velocity beta = v/c = tanh(theta)."""
    return np.tanh(theta)


def gamma(theta):
    """Lorentz factor gamma = cosh(theta)."""
    return np.cosh(theta)


def gamma_beta(theta):
    """Momentum magnitude gamma*beta = sinh(theta)."""
    return np.sinh(theta)


def theta_from_gamma(g):
    """Inverse map theta = arccosh(gamma)."""
    return np.arccosh(np.asarray(g, dtype=float))


def energy_MeV(theta):
    """Total electron energy in MeV for a given rapidity."""
    return gamma(theta) * ELECTRON_REST_ENERGY_MEV


def gudermannian(theta):
    """gd(theta) = arctan(sinh theta) = 2 arctan(tanh(theta/2)).

    The unique smooth bijection R -> (-pi/2, pi/2) with d(gd)/dtheta = sech.
    """
    return np.arctan(np.sinh(theta))


def inverse_gudermannian(phi):
    """Inverse Gudermannian, theta = arcsinh(tan phi)."""
    return np.arcsinh(np.tan(phi))


# ------------------------------------------------- coupled-parameter tuples

def momentum_components(theta):
    """(P1, P2, P3) of Eqs. (5)-(6), projected on the gyrovector triad."""
    th = np.asarray(theta, dtype=float)
    return np.sinh(th) * np.tanh(th), np.tanh(th), np.sinh(th)


def velocity_components(theta):
    """(dq1, dq2, dq3) of Eq. (7)."""
    th = np.asarray(theta, dtype=float)
    return np.tanh(th) ** 2, np.tanh(th) / np.cosh(th), np.tanh(th)


def coordinates(theta):
    """(q_perp, Re q_phi, q) of Eq. (8).

    q_phi = 1/2 ln(-sinh^2 th) is complex on the principal branch; its
    imaginary part (pi/2) is a constant phase that drops out of every
    real-valued observable, so only the real part is returned here.
    """
    th = np.asarray(theta, dtype=float)
    with np.errstate(divide="ignore"):
        q_phi = 0.5 * np.log(np.sinh(th) ** 2)
        q = 0.5 * np.log(np.tanh(th) ** 2)
    return th, q_phi, q


def lorentz_invariant(theta):
    """E^2 - P^2, identically 1 by the hyperbolic Pythagorean identity."""
    th = np.asarray(theta, dtype=float)
    return np.cosh(th) ** 2 - np.sinh(th) ** 2


# ----------------------------------------------------------------- Figure 1

def q3_over_t(theta):
    """Normalized longitudinal coordinate q_3/t = q/q_perp = ln(tanh th)/th.

    This is the ratio of the radial coordinate q = 1/2 ln(tanh^2 th) of
    Eq. (8) to the transverse coordinate q_perp = th.  Its two asymptotic
    limits are exactly the curves annotated in Fig. 1:

        theta -> infinity :  ln(1 - 2 e^{-2th})/th  ->  -2 e^{-2th}/th
        theta -> 0        :  ln(tanh th)/th         ->  ln(th)/th
    """
    th = np.asarray(theta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.tanh(th)) / th


def q3_over_t_asymptote_ur(theta):
    """Ultrarelativistic asymptote of q3_over_t: -2 e^{-2 theta}/theta."""
    th = np.asarray(theta, dtype=float)
    return -2.0 * np.exp(-2.0 * th) / th


def q3_over_t_asymptote_nr(theta):
    """Nonrelativistic asymptote of q3_over_t: ln(theta)/theta."""
    th = np.asarray(theta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(th) / th


if __name__ == "__main__":
    th = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    print("theta      gamma      E[MeV]     gd(th)     q3/t")
    for t in th:
        print(f"{t:6.2f} {gamma(t):10.4f} {energy_MeV(t):10.4f} "
              f"{gudermannian(t):10.4f} {q3_over_t(t):10.4f}")
    err = np.abs(lorentz_invariant(np.linspace(0, 10, 1001)) - 1.0).max()
    print(f"max |E^2 - P^2 - 1| over theta in [0,10] : {err:.3e}")

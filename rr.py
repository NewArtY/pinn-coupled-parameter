"""Radiation reaction and QED-nonlinearity diagnostics.

Provides (i) the dimensionless parameters that delimit the validity of the
classical, radiation-reaction-free treatment used throughout the paper, and
(ii) the gamma^2-dominant term of the reduced Landau-Lifshitz force so that
the size of the neglected back-reaction can be measured directly.

Units
-----
Dimensionless throughout: c = 1, m = 1, |e| = 1, time in 1/omega, momenta in
mc, fields in m c omega / e.  The only place where a dimensional quantity
enters is the laser wavelength, which fixes both

    eps_rad = 4 pi r_e / (3 lambda)          (classical RR coupling)
    eps_ph  = hbar omega / (m c^2)           (photon recoil parameter)

Two geometries must be distinguished, and they behave very differently.

**Geometry A -- electron initially at rest (the configuration actually
integrated in this repository).**  For a plane wave the light-front momentum
is conserved, gamma - P_z = 1, and therefore

    chi = a0 * eps_ph * (gamma - P_z) = a0 * eps_ph

with *no* enhancement by gamma: the electron runs away from the wave and the
field it experiences in its own frame saturates.  The classical RR importance
parameter (fractional energy loss over the pulse) is R ~ eps_rad * a0^3.

**Geometry B -- relativistic electron beam counter-propagating into the
pulse (the laser-synchrotron-source application).**  Here the Doppler factor
is gamma(1 + beta) ~ 2 gamma0 and

    chi ~ 2 gamma0 a0 eps_ph,      R ~ 2 eps_rad gamma0 a0^2 ,

so both grow linearly with the beam energy.  At a0 = 85 and 500 MeV one finds
chi ~ 0.4 and R ~ 0.17: radiation reaction is dominant and the process is
quantum, exactly the regime measured by Cole et al., Phys. Rev. X 8, 011020
(2018).  The classical model of this paper does *not* apply there.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "CLASSICAL_ELECTRON_RADIUS_UM", "eps_rad", "eps_photon",
    "chi_from_rest", "chi_counterprop", "R_from_rest", "R_counterprop",
    "landau_lifshitz_force", "regime_table",
]

CLASSICAL_ELECTRON_RADIUS_UM = 2.8179403262e-9   # r_e in micrometres
ELECTRON_REST_ENERGY_EV = 510998.95
HC_EV_UM = 1.23984193                            # h*c in eV*um


def eps_rad(lambda_um: float = 1.0) -> float:
    """Classical radiation-reaction coupling 4 pi r_e / (3 lambda).

    1.180e-8 for lambda = 1 um.
    """
    return 4.0 * math.pi * CLASSICAL_ELECTRON_RADIUS_UM / (3.0 * lambda_um)


def eps_photon(lambda_um: float = 1.0) -> float:
    """Photon recoil parameter hbar*omega / (m c^2).  2.426e-6 at 1 um."""
    return (HC_EV_UM / lambda_um) / ELECTRON_REST_ENERGY_EV


# ------------------------------------------------------ validity parameters

def chi_from_rest(a0, lambda_um: float = 1.0):
    """Quantum nonlinearity parameter for an electron starting at rest.

    Exact for a plane wave, where gamma - P_z = 1 identically.
    """
    return np.asarray(a0, dtype=float) * eps_photon(lambda_um)


def chi_counterprop(a0, gamma0, lambda_um: float = 1.0):
    """chi for a beam of Lorentz factor gamma0 counter-propagating into the pulse."""
    a0 = np.asarray(a0, dtype=float)
    gamma0 = np.asarray(gamma0, dtype=float)
    return 2.0 * gamma0 * a0 * eps_photon(lambda_um)


def R_from_rest(a0, lambda_um: float = 1.0):
    """Classical RR importance (fractional energy loss), electron from rest."""
    return eps_rad(lambda_um) * np.asarray(a0, dtype=float) ** 3


def R_counterprop(a0, gamma0, lambda_um: float = 1.0):
    """Classical RR importance for the counter-propagating geometry."""
    a0 = np.asarray(a0, dtype=float)
    gamma0 = np.asarray(gamma0, dtype=float)
    return 2.0 * eps_rad(lambda_um) * gamma0 * a0 ** 2


# -------------------------------------------------------- Landau-Lifshitz

def landau_lifshitz_force(P, E, B, lambda_um: float = 1.0):
    """gamma^2-dominant term of the reduced Landau-Lifshitz radiation force.

        f_rad = - eps_rad * gamma^2 * [ (E + beta x B)^2 - (beta.E)^2 ] * beta

    This term is antiparallel to the velocity, is independent of the sign of
    the charge, and dominates the full LL expression by O(gamma^2); the
    remaining terms are retained in neither the estimates above nor here.

    Parameters
    ----------
    P : (3,) array_like -- momentum in units of mc
    E, B : (3,) array_like -- fields in units of m c omega / e

    Returns
    -------
    (3,) ndarray : the radiation-reaction force in units of m c omega.
    """
    P = np.asarray(P, dtype=float)
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    g = math.sqrt(1.0 + float(P @ P))
    beta = P / g
    lorentz = E + np.cross(beta, B)
    scalar = float(lorentz @ lorentz) - float(beta @ E) ** 2
    return -eps_rad(lambda_um) * g * g * scalar * beta


# --------------------------------------------------------------- reporting

def regime_table(lambda_um: float = 1.0):
    """Return the two validity tables of Sec. 3.5 as lists of dicts."""
    from_rest = [
        {"regime": tag, "a0": a0,
         "chi": float(chi_from_rest(a0, lambda_um)),
         "R": float(R_from_rest(a0, lambda_um))}
        for tag, a0 in (("LT", 0.85), ("WNL", 8.5), ("UR", 85.0))
    ]
    counter = [
        {"a0": a0, "E0_MeV": E0, "gamma0": E0 / 0.51099895,
         "chi": float(chi_counterprop(a0, E0 / 0.51099895, lambda_um)),
         "R": float(R_counterprop(a0, E0 / 0.51099895, lambda_um))}
        for a0 in (0.85, 8.5, 85.0) for E0 in (100.0, 500.0, 1000.0)
    ]
    return from_rest, counter


if __name__ == "__main__":
    lam = 1.0
    print(f"lambda = {lam} um :  eps_rad = {eps_rad(lam):.3e}   "
          f"eps_ph = {eps_photon(lam):.3e}\n")
    a, b = regime_table(lam)
    print("A) electron initially at rest (this paper's numerics)")
    print(f"{'reg':>4} {'a0':>7} {'chi':>10} {'R_RR':>10}")
    for r in a:
        print(f"{r['regime']:>4} {r['a0']:>7.2f} {r['chi']:>10.2e} {r['R']:>10.2e}")
    print("\nB) counter-propagating beam (LSS geometry)")
    print(f"{'a0':>7} {'E0[MeV]':>9} {'gamma0':>8} {'chi':>10} {'R_RR':>10}")
    for r in b:
        print(f"{r['a0']:>7.2f} {r['E0_MeV']:>9.0f} {r['gamma0']:>8.0f} "
              f"{r['chi']:>10.2e} {r['R']:>10.2e}")

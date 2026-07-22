"""General relativistic electron dynamics in a Gaussian laser pulse.

This is the solver used for the *analytical-section* figures (Figs. 1-8, 13).
It is deliberately kept separate from :mod:`physics`, which implements the
reduced plane-wave benchmark that the PINN checkpoints were trained against
and whose interface must stay frozen.

Model
-----
Dimensionless units: c = 1, m = 1, |e| = 1, time in 1/omega, lengths in
c/omega (so one laser wavelength is 2*pi), fields in m c omega / e.

The pulse propagates along +z.  With the retarded phase eta = t - z the
transverse vector-potential amplitude is (Eq. (11) of the paper)

    A(rho, z, eta) = a0 * exp(-eta^2/L^2) * exp(-rho^2/b(z)^2) * b0/b(z),
    b(z) = b0 * sqrt(1 + z^2/z_f^2),      z_f = b0^2/2,

and the carrier phase including the Gouy and wavefront-curvature corrections
(Eq. (13)) is

    Phi = eta - z/z_f - rho^2/(2 R(z)) + phi_0,   R(z) = z (1 + z_f^2/z^2).

Following the field convention of Eqs. (24)-(26) of the paper, the laser
magnetic field is taken as B = z_hat x E, which is exact for a plane wave and
is the paraxial leading order for the focused beam.  A static, homogeneous
field H0 z_hat may be superposed.

The equations of motion for an electron (charge -1) are

    dr/dt = P/gamma,
    dP/dt = -(E + beta x (B + H0 z_hat)) + f_rad,
    gamma = sqrt(1 + |P|^2),

with f_rad the optional Landau-Lifshitz term of :mod:`rr`.  Integrating P and
obtaining gamma algebraically makes E^2 - P^2 = 1 exact by construction.

Polarizations
-------------
``"linear"``      a = A (cos Phi, 0, 0)
``"circular"``    a = A/sqrt2 (cos Phi, sin Phi, 0)
``"elliptical"``  a = A/sqrt(1+eps^2) (cos Phi, eps sin Phi, 0)
``"oam"``         circular, multiplied by (sqrt2 rho/b)^|l| and with the
                  azimuthal phase l*arctan2(y, x) added -- a Laguerre-Gauss
                  LG_{0l} beam carrying l hbar of orbital angular momentum
                  per photon.
``"mixed"``       equal-weight superposition of the linear and circular cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dcfield

import numpy as np
from scipy.integrate import solve_ivp

from rr import landau_lifshitz_force

__all__ = ["PulseConfig", "fields", "solve_trajectory", "TWO_PI"]

TWO_PI = 2.0 * np.pi


@dataclass
class PulseConfig:
    """Parameters of the laser pulse and of the static magnetic field."""

    a0: float = 1.0
    L: float = 5.0 * TWO_PI          # pulse half-width (5 wavelengths)
    b0: float = 40.0 * TWO_PI        # beam waist (40 wavelengths)
    polarization: str = "circular"
    ellipticity: float = 0.6         # used by "elliptical"
    oam_l: int = 1                   # used by "oam"
    phi0: float = 0.0                # carrier envelope phase
    H0: float = 0.0                  # static field along +z, in units of m c omega / e
    plane_wave: bool = False         # ignore transverse structure / Gouy / curvature
    radiation_reaction: bool = False
    lambda_um: float = 1.0

    @property
    def z_f(self) -> float:
        """Rayleigh length z_f = b0^2 / 2 (paper convention)."""
        return 0.5 * self.b0 ** 2


def _beam_radius(z, cfg: PulseConfig):
    return cfg.b0 * np.sqrt(1.0 + (z / cfg.z_f) ** 2)


def fields(t, r, cfg: PulseConfig):
    """Return (E, B) of the laser at time ``t`` and position ``r`` (3-vector)."""
    x, y, z = r
    eta = t - z
    rho2 = x * x + y * y

    if cfg.plane_wave:
        amp = cfg.a0 * np.exp(-(eta * eta) / (cfg.L * cfg.L))
        phase = eta + cfg.phi0
        b = cfg.b0
    else:
        b = _beam_radius(z, cfg)
        amp = (cfg.a0 * np.exp(-(eta * eta) / (cfg.L * cfg.L))
               * np.exp(-rho2 / (b * b)) * (cfg.b0 / b))
        gouy = z / cfg.z_f
        # R(z) = z (1 + z_f^2/z^2) = z + z_f^2/z ; regularized at z = 0.
        curv = 0.0 if z == 0.0 else rho2 / (2.0 * (z + cfg.z_f ** 2 / z))
        phase = eta - gouy - curv + cfg.phi0

    pol = cfg.polarization
    if pol == "linear":
        Ex, Ey = amp * np.cos(phase), 0.0
    elif pol == "circular":
        s = amp / np.sqrt(2.0)
        Ex, Ey = s * np.cos(phase), s * np.sin(phase)
    elif pol == "elliptical":
        eps = cfg.ellipticity
        s = amp / np.sqrt(1.0 + eps * eps)
        Ex, Ey = s * np.cos(phase), eps * s * np.sin(phase)
    elif pol == "oam":
        l = int(cfg.oam_l)
        radial = (np.sqrt(2.0 * rho2) / b) ** abs(l)
        azim = l * np.arctan2(y, x)
        s = amp * radial / np.sqrt(2.0)
        Ex, Ey = s * np.cos(phase + azim), s * np.sin(phase + azim)
    elif pol == "mixed":
        s = amp / 2.0
        Ex = s * (np.cos(phase) + np.cos(phase) / np.sqrt(2.0))
        Ey = s * np.sin(phase) / np.sqrt(2.0)
    else:
        raise ValueError(f"unknown polarization {pol!r}")

    E = np.array([Ex, Ey, 0.0])
    B = np.array([-Ey, Ex, 0.0])          # B = z_hat x E
    return E, B


def _force(t, r, P, cfg: PulseConfig):
    """Total force on the electron (charge -1), including optional RR."""
    g = np.sqrt(1.0 + P @ P)
    beta = P / g
    E, B = fields(t, r, cfg)
    Btot = B + np.array([0.0, 0.0, cfg.H0])
    f = -(E + np.cross(beta, Btot))
    if cfg.radiation_reaction:
        f = f + landau_lifshitz_force(P, E, Btot, cfg.lambda_um)
    return f, beta, g


def _rhs_t(t, s, cfg: PulseConfig):
    """RHS with the laboratory time as the independent variable."""
    r, P = s[:3], s[3:6]
    f, beta, _ = _force(t, r, P, cfg)
    return np.concatenate((beta, f))


def _rhs_eta(eta, s, cfg: PulseConfig):
    """RHS with the retarded phase eta = t - z as the independent variable.

    Since d(eta)/dt = 1 - beta_z = 1/gamma > 0 for a plane wave, eta is a
    strictly monotone reparametrization of the worldline.  Using it as the
    integration variable guarantees that the electron traverses the whole
    pulse, which a fixed time window does not: at large a0 the particle is
    dragged forward and its retarded phase advances gamma times more slowly.
    """
    r, P, t = s[:3], s[3:6], s[6]
    f, beta, _ = _force(t, r, P, cfg)
    denom = 1.0 - beta[2]
    return np.concatenate((beta / denom, f / denom, [1.0 / denom]))


def solve_trajectory(cfg: PulseConfig, n_out=4000, r0=None, P0=None,
                     eta_span=None, independent="eta", t_max=None,
                     rtol=1e-10, atol=1e-12):
    """Integrate the electron trajectory through the pulse.

    Parameters
    ----------
    cfg : PulseConfig
    independent : {"eta", "t"}
        Integration variable.  ``"eta"`` (default) spans the retarded phase
        and always covers the full pulse; ``"t"`` integrates in laboratory
        time up to ``t_max``.
    eta_span : (float, float), optional
        Retarded-phase interval; defaults to ``(-3L, +3L)``, i.e. the pulse
        envelope is below ``e^{-9}`` of its peak at both ends.
    r0, P0 : (3,) array_like, optional
        Initial position and momentum.  Default: electron at rest on the beam
        axis at the focus.

    Returns
    -------
    dict with keys ``t``, ``r`` (n,3), ``P`` (n,3), ``gamma`` (n,),
    ``beta`` (n,3), ``betadot`` (n,3), ``eta`` (n,).
    """
    if r0 is None:
        r0 = np.zeros(3)
    if P0 is None:
        P0 = np.zeros(3)
    r0 = np.asarray(r0, dtype=float)
    P0 = np.asarray(P0, dtype=float)

    if independent == "eta":
        if eta_span is None:
            eta_span = (-3.0 * cfg.L, 3.0 * cfg.L)
        # t = eta + z, so the starting time follows from the starting phase.
        s0 = np.concatenate((r0, P0, [eta_span[0] + r0[2]]))
        grid = np.linspace(eta_span[0], eta_span[1], n_out)
        sol = solve_ivp(_rhs_eta, eta_span, s0, t_eval=grid, method="DOP853",
                        args=(cfg,), rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        y = sol.y.T
        r, P, t = y[:, :3], y[:, 3:6], y[:, 6]
    elif independent == "t":
        if t_max is None:
            t_max = 6.0 * cfg.L
        s0 = np.concatenate((r0, P0))
        grid = np.linspace(0.0, t_max, n_out)
        sol = solve_ivp(_rhs_t, (0.0, t_max), s0, t_eval=grid, method="DOP853",
                        args=(cfg,), rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        y = sol.y.T
        r, P, t = y[:, :3], y[:, 3:6], sol.t
    else:
        raise ValueError("independent must be 'eta' or 't'")

    g = np.sqrt(1.0 + np.sum(P * P, axis=1))
    beta = P / g[:, None]
    # Acceleration from the equations of motion (not by differencing the
    # output grid, which would lose accuracy on the fast carrier oscillation).
    betadot = np.empty_like(beta)
    for i in range(len(t)):
        f, _, gi = _force(t[i], r[i], P[i], cfg)
        betadot[i] = f / gi - P[i] * (P[i] @ f) / gi ** 3

    return {"t": t, "r": r, "P": P, "gamma": g, "beta": beta,
            "betadot": betadot, "eta": t - r[:, 2]}


if __name__ == "__main__":
    for tag, a0 in (("LT", 0.85), ("WNL", 8.5), ("UR", 85.0)):
        cfg = PulseConfig(a0=a0, plane_wave=True)
        out = solve_trajectory(cfg, n_out=2000)
        inv = np.abs(out["gamma"] ** 2 - 1.0 - np.sum(out["P"] ** 2, axis=1)).max()
        lf = np.abs(out["gamma"] - out["P"][:, 2] - 1.0).max()
        print(f"{tag:>3} a0={a0:>5.2f}  gamma_max={out['gamma'].max():9.3f}  "
              f"|E^2-P^2-1|max={inv:.2e}  |gamma-Pz-1|max={lf:.2e}")

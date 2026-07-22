"""Figure 3 -- phase-synchronization quality map Q(a0, H0).

Two constructions are provided.

``--analytic`` (default) reproduces the published map from the closed-form
model

    Q(a0, H0) = A(a0) * exp( -[(H0 - 1) / Delta(a0)]^2 ),
    A(a0)     = a0 / (a_c + a0),
    Delta(a0) = Delta_max * A(a0),

with a_c = 0.35 and Delta_max = 0.40.  The reading is: the cyclotron lock of
Eq. (36) is centred on H0 = omega_0/omega = 1; the fraction A(a0) of the
cyclotron period over which the laser force exceeds the detuning threshold
saturates at unity for a0 >> a_c, and the tolerable detuning bandwidth scales
with the same factor because the field-induced transverse momentum is what
sets it.  This is an interpolation formula, not a derivation -- it was the
implicit content of the published figure and is written out here so that the
figure is reproducible.

``--ode`` computes a first-principles synchronization quality by integrating
the equations of motion for each (a0, H0) and measuring the energy the
electron *retains* after the pulse has passed,

    Q_ODE(a0, H0) = (gamma_final - 1) / (gamma_max - 1),

which is identically zero when the Lawson-Woodward theorem applies (H0 = 0)
and approaches unity when the cyclotron lock holds the particle in the
accelerating phase.  This is the recommended definition for the revision; it
is slower (a few minutes for the default grid).
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from _style import apply_style, save, save_data, panel_label

A_C = 0.35
DELTA_MAX = 0.40


def q_analytic(a0, H0):
    """Closed-form synchronization quality (see module docstring)."""
    a0 = np.asarray(a0, dtype=float)
    H0 = np.asarray(H0, dtype=float)
    amp = a0 / (A_C + a0)
    width = DELTA_MAX * amp
    return amp * np.exp(-((H0 - 1.0) / width) ** 2)


def q_ode(a0_vals, H0_vals, n_out=1500):
    """First-principles synchronization quality from the equations of motion."""
    from dynamics import PulseConfig, solve_trajectory, TWO_PI

    Q = np.zeros((len(H0_vals), len(a0_vals)))
    for i, H0 in enumerate(H0_vals):
        for j, a0 in enumerate(a0_vals):
            cfg = PulseConfig(a0=float(a0), H0=float(H0), L=2.0 * TWO_PI,
                              plane_wave=True, polarization="circular")
            tr = solve_trajectory(cfg, n_out=n_out)
            g = tr["gamma"]
            denom = g.max() - 1.0
            Q[i, j] = 0.0 if denom <= 0 else (g[-1] - 1.0) / denom
        print(f"    H0 = {H0:5.2f} done")
    return np.clip(Q, 0.0, 1.0)


def main(mode="analytic"):
    apply_style()
    if mode == "analytic":
        a0 = np.logspace(-1, 2, 400)
        H0 = np.linspace(0.0, 3.0, 400)
        A, H = np.meshgrid(a0, H0)
        Q = q_analytic(A, H)
        stem = "fig_new3_sync_map"
    else:
        a0 = np.logspace(-1, 2, 60)
        H0 = np.linspace(0.0, 3.0, 40)
        print("  integrating trajectories for the ODE-based map ...")
        Q = q_ode(a0, H0)
        A, H = np.meshgrid(a0, H0)
        stem = "fig_new3_sync_map_ode"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    levels = np.linspace(0.0, 1.0, 26)
    cf = ax.contourf(A, H, Q, levels=levels, cmap="RdYlGn", extend="neither")
    cs = ax.contour(A, H, Q, levels=[0.5, 0.7, 0.9], colors="white",
                    linewidths=1.1)
    ax.clabel(cs, fmt="%.2f", fontsize=8)

    ax.axhline(1.0, ls="--", lw=1.6, color="#2b3a8f",
               label=r"$H_0=1$  ($\omega_0=\omega$)")
    for a_sep in (1.0, 10.0):
        ax.axvline(a_sep, ls=":", lw=0.9, color="0.3")

    box = dict(boxstyle="round,pad=0.3", fc="#ffe9ec", ec="none", alpha=0.9)
    ax.text(0.3, 2.72, "LT\n$a_L\\ll1$", ha="center", fontsize=8, bbox=box)
    ax.text(3.2, 2.72, "WNL\n$a_L\\sim1$", ha="center", fontsize=8, bbox=box)
    ax.text(32.0, 2.72, "UR\n$a_L\\gg1$", ha="center", fontsize=8, bbox=box)

    ax.set_xscale("log")
    ax.set_xlabel(r"$a_0$")
    ax.set_ylabel(r"Normalized magnetic field $H_0=\omega_0/\omega$")
    ax.set_ylim(0.0, 3.0)
    ax.legend(loc="lower right", framealpha=0.95)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(r"Synchronization quality $Q(a_0,H_0)$")

    save_data(stem, a0=A.ravel(), H0=H.ravel(), Q=Q.ravel())
    return save(fig, stem)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ode", action="store_true",
                   help="compute Q by integrating the equations of motion")
    args = p.parse_args()
    main("ode" if args.ode else "analytic")

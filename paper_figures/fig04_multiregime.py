"""Figure 4 -- multi-regime trajectory comparison.

Rows are the four physical scenarios (circular / linear polarization, with and
without the resonant static field H0), columns the three intensity regimes.
The plotted quantity is the rapidity theta(eta) = arccosh(gamma), normalized
per panel, against the retarded phase eta/L -- i.e. the single coupled
parameter of the paper, on a common abscissa for every panel.

NOTE ON THE PUBLISHED VERSION.  The submitted Fig. 4 plotted a normalized
longitudinal coordinate on twelve *different* and very narrow time windows,
and its panel (a) carried the annotation ``max = 4.44e-18``: that panel was
displaying floating-point round-off amplified to full scale, and panel (i)
(``max = 5.00e+02``) showed the spikes of a diverging closed-form expression.
Those panels are numerical artefacts, not trajectories.  This script rebuilds
the figure by direct DOP853 integration of the equations of motion, on a
single common abscissa, with the Lorentz invariant conserved to ~1e-13.

The plane-wave limit of the pulse is used (transverse beam structure switched
off), matching both the closed-form solutions of Sec. 4.3 and the PINN
benchmark of Sec. 7.3, so that the three figures describe the same system.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (apply_style, save, save_data, panel_label,
                    REGIME_COLORS, REGIME_A0, REGIME_INTENSITY)
from dynamics import PulseConfig, solve_trajectory, TWO_PI

L_PULSE = 5.0 * TWO_PI          # Table 3: L = 5 lambda

SCENARIOS = [
    ("circular", 0.0, "Circ. pol., no $H_0$"),
    ("circular", 1.0, "Circ. pol., $H_0$ applied"),
    ("linear", 0.0, "Lin. pol., no $H_0$"),
    ("linear", 1.0, "Lin. pol., $H_0$ applied"),
]
REGIMES = ["LT", "WNL", "UR"]


def main():
    apply_style()
    fig, axes = plt.subplots(4, 3, figsize=(11.0, 9.2), sharex=True)
    export = {}

    for row, (pol, H0, row_label) in enumerate(SCENARIOS):
        for col, tag in enumerate(REGIMES):
            ax = axes[row, col]
            a0 = REGIME_A0[tag]
            cfg = PulseConfig(a0=a0, L=L_PULSE, polarization=pol, H0=H0,
                              plane_wave=True)
            tr = solve_trajectory(cfg, n_out=6000)
            eta = tr["eta"] / L_PULSE
            theta = np.arccosh(np.clip(tr["gamma"], 1.0, None))
            tmax = theta.max()
            ax.plot(eta, theta / tmax if tmax > 0 else theta,
                    color=REGIME_COLORS[tag], lw=1.2)

            # Same run with the Landau-Lifshitz radiation-reaction force, to
            # show where the classical, RR-free treatment stops being valid.
            cfg_rr = PulseConfig(a0=a0, L=L_PULSE, polarization=pol, H0=H0,
                                 plane_wave=True, radiation_reaction=True)
            tr_rr = solve_trajectory(cfg_rr, n_out=6000)
            theta_rr = np.arccosh(np.clip(tr_rr["gamma"], 1.0, None))
            ax.plot(tr_rr["eta"] / L_PULSE, theta_rr / tmax if tmax > 0
                    else theta_rr, color="0.25", lw=1.0, ls="--", alpha=0.85)
            drop = 100.0 * (theta_rr.max() - tmax) / tmax if tmax > 0 else 0.0
            # For the H0 = 0 (non-resonant) panels the radiation-reaction shift
            # of theta_max is a delicate second-order quantity (the electron
            # co-moves with the wave, gamma - P_z stays close to unity and the
            # light-front invariant only drifts by O(eps_rad); the reduced LL
            # force even induces a tiny *positive* net drift, the plane-wave
            # radiation-reaction effect of Di Piazza (2008)).  Reporting its
            # sign at the 1e-4 % level would over-state the model's resolution,
            # so below a threshold we label the effect "negligible" rather than
            # printing a spurious "+0.0 %".  The physically robust statement --
            # the locally radiated power beta.f_rad <= 0 -- holds throughout
            # (see tests/test_physics.py::test_rr_removes_energy_locally).
            if abs(drop) < 0.05:
                rr_txt = r"RR: negl."
            else:
                rr_txt = rf"RR: ${drop:+.1f}\%$"

            ax.axhline(0.0, color="0.8", lw=0.6, ls=":")
            ax.text(0.97, 0.93, rf"$\theta_{{\max}}={tmax:.2f}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color="0.35")
            ax.text(0.97, 0.80, rr_txt,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="0.45")
            panel_label(ax, f"({'abcdefghijkl'[row * 3 + col]})")
            ax.set_ylim(-0.08, 1.15)
            ax.set_xlim(eta[0], eta[-1])
            if row == 0:
                ax.set_title(rf"{tag}  ($I={REGIME_INTENSITY[tag]}$ W/cm$^2$)",
                             color=REGIME_COLORS[tag])
            if col == 0:
                ax.set_ylabel(f"{row_label}\n" r"$\theta/\theta_{\max}$",
                              fontsize=8.5)
            if row == 3:
                ax.set_xlabel(r"Retarded phase $\eta/L$")
            export[f"eta_{tag}_{pol}_H{H0:g}"] = eta
            export[f"theta_{tag}_{pol}_H{H0:g}"] = theta
            export[f"theta_rr_{tag}_{pol}_H{H0:g}"] = theta_rr

    handles = [plt.Line2D([], [], color="0.5", lw=1.4),
               plt.Line2D([], [], color="0.25", lw=1.0, ls="--")]
    fig.legend(handles, ["classical (no radiation reaction)",
                         "with Landau--Lifshitz radiation reaction"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save_data("fig_new4_multiregime_trajectories", **export)
    return save(fig, "fig_new4_multiregime_trajectories")


if __name__ == "__main__":
    main()

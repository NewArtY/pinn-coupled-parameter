"""Figure 8 -- angular distribution of the emitted radiation.

The energy radiated per unit solid angle,

    dW/dOmega = int dt | n x ((n - beta) x betadot) |^2 / (1 - n.beta)^5 ,

is evaluated on a grid of observation polar angles theta_obs measured from
the laser propagation axis (+z).

(a) the three intensity regimes: increasing a0 pulls the emission cone
    towards the axis, because the electron's own direction of motion makes an
    angle tan(theta_v) = 2/a_perp with +z, and the emission is beamed into
    1/gamma about that direction;
(b) the static-field series H0 = 0 ... 5 at fixed a0 = 8.5 -- the cyclotron
    lock is what actually reshapes the directionality, and it is the design
    knob for a laser synchrotron source;
(c) the pulse-width series s in {0.3 ... 2.0} with L -> sL.  The curves are
    nearly degenerate: in the plane-wave limit the emission angle is fixed by
    the *local* field amplitude and not by the envelope length, so the pulse
    width is not a directionality control parameter.  This is reported rather
    than hidden.

NOTE ON THE PUBLISHED VERSION.  The submitted Fig. 8 showed "directionality
functions" of order 1e20-1e21, non-zero only in a single spike near
theta = 2.1 and flat elsewhere, with only one of the six curves visible.
Those were divergences of a closed-form integrand rather than an angular
distribution.  This script computes the Lienard-Wiechert distribution from
the trajectory; it is non-negative and normalized.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (apply_style, save, save_data, panel_label,
                    REGIME_COLORS, REGIME_A0, REGIME_INTENSITY)
from dynamics import PulseConfig, solve_trajectory, TWO_PI
from spectra import angular_distribution

L_PULSE = 5.0 * TWO_PI
S_VALUES = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0]
H0_VALUES = [0.0, 1.0, 2.0, 3.5, 5.0]
H0_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
THETA_DEG = np.logspace(-1.3, np.log10(180.0), 500)


def _curve(**kw):
    cfg = PulseConfig(polarization="circular", plane_wave=True, **kw)
    tr = solve_trajectory(cfg, n_out=8000)
    dW = angular_distribution(tr, np.deg2rad(THETA_DEG))
    return dW / dW.max(), tr["gamma"].max()


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    export = {"theta_deg": THETA_DEG}

    # ------------------------------------------------------------- panel (a)
    for tag in ("LT", "WNL", "UR"):
        dW, gmax = _curve(a0=REGIME_A0[tag], L=L_PULSE)
        axes[0].semilogx(THETA_DEG, dW, color=REGIME_COLORS[tag],
                         label=rf"{tag} ($a_0={REGIME_A0[tag]:g}$)")
        axes[0].axvline(np.rad2deg(np.arctan(2.0 / (REGIME_A0[tag] / np.sqrt(2)))),
                        color=REGIME_COLORS[tag], ls=":", lw=0.9, alpha=0.7)
        export[f"dW_{tag}"] = dW
    axes[0].set_title("intensity regimes", fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------------- panel (b)
    for H0, color in zip(H0_VALUES, H0_COLORS):
        dW, _ = _curve(a0=8.5, L=L_PULSE, H0=H0)
        axes[1].semilogx(THETA_DEG, dW, color=color, label=rf"$H_0={H0:g}$")
        export[f"dW_H{H0:g}"] = dW
    axes[1].set_title(r"static-field series ($a_0=8.5$)", fontsize=9.5)
    axes[1].legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------------- panel (c)
    cmap = plt.get_cmap("plasma")
    for i, s in enumerate(S_VALUES):
        dW, _ = _curve(a0=8.5, L=s * L_PULSE)
        axes[2].semilogx(THETA_DEG, dW, color=cmap(i / (len(S_VALUES) - 1)),
                         label=rf"$s={s}$")
        export[f"dW_s{s}"] = dW
    axes[2].set_title(r"pulse-width series ($L\rightarrow sL$, $a_0=8.5$)",
                      fontsize=9.5)
    axes[2].legend(loc="upper left", fontsize=7.5, ncol=2)

    for i, ax in enumerate(axes):
        ax.set_xlabel(r"Observation angle $\theta_{\rm obs}$ (deg)")
        ax.set_xlim(THETA_DEG[0], 180)
        ax.set_ylim(0, 1.12)
        panel_label(ax, f"({'abc'[i]})", loc=(0.03, 0.99))
    axes[0].set_ylabel(r"$dW/d\Omega$  (normalized)")

    fig.tight_layout()
    save_data("fig_new8_directionality_l_series", **export)
    return save(fig, "fig_new8_directionality_l_series")


if __name__ == "__main__":
    main()

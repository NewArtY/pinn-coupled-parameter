"""Figure 5 -- transverse phase portraits versus the static magnetic field.

Panel (a): forward channel -- electron initially at rest, pushed by the
co-propagating pulse.
Panel (b): backward channel -- electron injected counter-propagating with
gamma0 = 10, the configuration that produces the Thomson back-scattered
X-ray output of a laser synchrotron source.

Both panels show the transverse orbit (x/lambda, y/lambda) for
H0 = 0, 1, 2, 3.5, 5 in a circularly polarized pulse at the WNL amplitude
a0 = 8.5.

NOTE ON THE PUBLISHED VERSION.  Panel (b) of the submitted figure contained a
singular spike reaching y/lambda = -100 at x = 0, a divergence of the
closed-form expression rather than a feature of the motion; the remaining
curves were flat on that scale and therefore unreadable.  This script obtains
both panels by direct integration instead.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import apply_style, save, save_data, panel_label
from dynamics import PulseConfig, solve_trajectory, TWO_PI

H0_VALUES = [0.0, 1.0, 2.0, 3.5, 5.0]
H0_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
A0 = 8.5
L_PULSE = 2.0 * TWO_PI
GAMMA0_BACK = 10.0


def _orbit(H0, counter=False):
    P0 = np.zeros(3)
    if counter:
        P0 = np.array([0.0, 0.0, -np.sqrt(GAMMA0_BACK ** 2 - 1.0)])
    cfg = PulseConfig(a0=A0, L=L_PULSE, polarization="circular", H0=H0,
                      plane_wave=True)
    tr = solve_trajectory(cfg, n_out=4000, P0=P0)
    return tr["r"][:, 0] / TWO_PI, tr["r"][:, 1] / TWO_PI


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    export = {}

    for ax, counter, title in ((axes[0], False, r"Forward channel ($\xi^{+}$)"),
                               (axes[1], True,
                                r"Backward channel ($\xi^{-}$, "
                                rf"$\gamma_0={GAMMA0_BACK:g}$)")):
        for H0, color in zip(H0_VALUES, H0_COLORS):
            x, y = _orbit(H0, counter)
            ax.plot(x, y, color=color, lw=1.3, label=rf"$H_0={H0:g}$")
            key = f"{'back' if counter else 'fwd'}_H{H0:g}"
            export[f"x_{key}"], export[f"y_{key}"] = x, y
        ax.axhline(0.0, color="0.85", lw=0.6, ls=":")
        ax.axvline(0.0, color="0.85", lw=0.6, ls=":")
        ax.set_xlabel(r"$x/\lambda$")
        ax.set_ylabel(r"$y/\lambda$")
        ax.set_title(title, fontsize=9.5)
        ax.legend(loc="best", fontsize=8)

    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    save_data("fig_new5_phase_diagrams_H0_series", **export)
    return save(fig, "fig_new5_phase_diagrams_H0_series")


if __name__ == "__main__":
    main()

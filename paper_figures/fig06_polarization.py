"""Figures 6a / 6b -- transverse phase portraits for five polarization types.

Fig. 6a: linear, circular, elliptical (eps = 0.6).
Fig. 6b: Laguerre-Gauss OAM (l = 1) and a mixed LP+CP superposition.

Left column of each pair: no static field.  Right column: resonant H0 = 1.
Line brightness encodes the local pulse intensity, so that the bright part of
each orbit is the segment traversed near the envelope peak -- the behaviour
the published caption describes but which the submitted figure did not
actually show (its curves were drawn in a single flat colour).

The OAM panels need the transverse beam structure, so those two runs use the
full Gaussian beam with a waist reduced to b0 = 3 lambda in order to place the
electron in the region where the LG_{01} amplitude varies appreciably; all
other panels use the plane-wave limit.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from _style import apply_style, save, save_data, panel_label
from dynamics import PulseConfig, solve_trajectory, TWO_PI

A0 = 8.5
L_PULSE = 2.0 * TWO_PI

GROUP_A = [
    ("linear", "Linear", "#1f77b4", {}),
    ("circular", "Circular", "#ff7f0e", {}),
    ("elliptical", r"Elliptical ($\varepsilon=0.6$)", "#2ca02c",
     {"ellipticity": 0.6}),
]
GROUP_B = [
    ("oam", r"OAM ($\ell=1$)", "#9467bd", {"oam_l": 1}),
    ("mixed", "Mixed LP+CP", "#d62728", {}),
]


def _orbit(pol, H0, extra):
    kw = dict(a0=A0, L=L_PULSE, polarization=pol, H0=H0, plane_wave=True)
    kw.update(extra)
    r0 = np.zeros(3)
    if pol == "oam":
        # LG_{01} vanishes on axis: start the electron off-axis inside a
        # tightly focused beam so that it samples the vortex structure.
        kw["plane_wave"] = False
        kw["b0"] = 3.0 * TWO_PI
        r0 = np.array([0.35 * TWO_PI, 0.0, 0.0])
    cfg = PulseConfig(**kw)
    tr = solve_trajectory(cfg, n_out=4000, r0=r0)
    env = np.exp(-(tr["eta"] / cfg.L) ** 2)
    return tr["r"][:, 0] / TWO_PI, tr["r"][:, 1] / TWO_PI, env


def _draw(ax, x, y, env, color):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    alpha = 0.18 + 0.82 * (env[:-1] / max(env.max(), 1e-30))
    colors = np.concatenate([np.tile(rgb, (len(segs), 1)), alpha[:, None]],
                            axis=1)
    # Rasterized: a per-segment coloured path with thousands of segments makes
    # the vector PDF ~1.3 MB and slows the LaTeX build; the axes, labels and
    # ticks stay vector.
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=1.4,
                                     rasterized=True))
    pad = 0.08 * max(np.ptp(x), np.ptp(y), 1e-6)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)


def _make(group, stem, letters):
    apply_style()
    n = len(group)
    fig, axes = plt.subplots(n, 2, figsize=(8.6, 3.1 * n), squeeze=False)
    export = {}
    k = 0
    for row, (pol, label, color, extra) in enumerate(group):
        for col, H0 in enumerate((0.0, 1.0)):
            ax = axes[row, col]
            x, y, env = _orbit(pol, H0, extra)
            _draw(ax, x, y, env, color)
            ax.axhline(0.0, color="0.9", lw=0.6, ls=":")
            ax.axvline(0.0, color="0.9", lw=0.6, ls=":")
            ax.set_xlabel(r"$x/\lambda$")
            ax.set_ylabel(r"$y/\lambda$")
            panel_label(ax, f"({letters[k]})")
            ax.set_title("no $H_0$" if H0 == 0 else "resonant $H_0=1$",
                         fontsize=9)
            if col == 0:
                ax.text(-0.28, 0.5, label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9.5)
            export[f"x_{pol}_H{H0:g}"], export[f"y_{pol}_H{H0:g}"] = x, y
            k += 1
    fig.tight_layout()
    save_data(stem, **export)
    return save(fig, stem)


def main():
    _make(GROUP_A, "fig_new6a_polarization_p1", "abcdef")
    _make(GROUP_B, "fig_new6b_polarization_p2", "ghij")


if __name__ == "__main__":
    main()

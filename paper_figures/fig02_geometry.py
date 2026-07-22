"""Figure 2 -- interaction geometry of the Gaussian beam.

(a) side view of the beam envelope b(z) = b0 sqrt(1 + z^2/z_f^2) with the
    gyrovector triad (k, s, n) at the focus O';
(b) on-axis intensity  I/I_max = exp(-z^2/L^2) * (b0/b(z))^2;
(c) polarization ellipses in the transverse (Ex/E0, Ey/E0) plane.

The panel-(a) and panel-(b) length scales are shown in units of the laser
wavelength with b0 = 1 lambda and z_f = 5 lambda, chosen for legibility; the
physical runs of Table 3 use b0 = 40 lambda (z_f = 800 lambda).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import apply_style, save, save_data, panel_label

B0 = 1.0        # beam waist, in units of lambda (schematic)
Z_F = 5.0       # Rayleigh length, in units of lambda (schematic)
L_PULSE = 5.0   # pulse half-width, in units of lambda (Table 3)


def beam_radius(z):
    return B0 * np.sqrt(1.0 + (z / Z_F) ** 2)


def main():
    apply_style()
    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.45,
                          wspace=0.22)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------- panel (a)
    z = np.linspace(-12.0, 12.0, 800)
    b = beam_radius(z)
    ax_a.plot(z, b, color="#1f77b4")
    ax_a.plot(z, -b, color="#1f77b4")
    ax_a.fill_between(z, -b, b, color="#1f77b4", alpha=0.10)

    ax_a.axhline(0.0, color="0.8", lw=0.7, ls=":")
    for zz in (-Z_F, 0.0, Z_F):
        ax_a.axvline(zz, color="0.8", lw=0.7, ls="--")
    ax_a.text(-Z_F, 2.85, r"$z=-z_f$", ha="center", color="0.45", fontsize=8)
    ax_a.text(Z_F, 2.85, r"$z=+z_f$", ha="center", color="0.45", fontsize=8)

    ax_a.plot([0], [0], "o", color="k", ms=5)
    ax_a.text(0.15, -0.30, r"$O'$", fontsize=10)

    ax_a.annotate("", xy=(0, 1.0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2))
    ax_a.text(0.12, 1.05, r"$\mathbf{k}$", color="#ff7f0e", fontsize=12)
    ax_a.annotate("", xy=(1.7, 0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))
    ax_a.text(1.85, 0.10, r"$\mathbf{n}$", color="#d62728", fontsize=12)
    ax_a.annotate("", xy=(-1.25, 0.72), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="#9467bd", lw=2))
    ax_a.text(-1.55, 0.85, r"$\mathbf{s}$", color="#9467bd", fontsize=12)

    ax_a.annotate("", xy=(9.5, 0.0), xytext=(5.5, 0.0),
                  arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.5))
    ax_a.text(7.5, 0.28, r"Laser $(+z)$", color="#d62728", ha="center")
    ax_a.annotate("", xy=(-9.5, -0.55), xytext=(-3.2, -0.55),
                  arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.5))
    ax_a.text(-6.3, -0.95, r"$e^{-}\ (-z)$", color="#2ca02c", ha="center")

    ax_a.annotate("", xy=(Z_F, -1.75), xytext=(0.0, -1.75),
                  arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.2))
    ax_a.text(Z_F / 2, -2.0, r"$z_f$", color="0.35", ha="center")
    ax_a.annotate("", xy=(0.0, 1.0), xytext=(0.0, 0.0),
                  arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9))
    ax_a.text(0.22, 0.5, r"$w_0$", color="0.25")

    ax_a.set_xlim(-12.5, 12.5)
    ax_a.set_ylim(-3.2, 3.2)
    ax_a.set_xlabel(r"$z/\lambda$")
    ax_a.set_ylabel("Transverse coordinate")
    panel_label(ax_a, "(a)")

    # ------------------------------------------------------------- panel (b)
    zi = np.linspace(-15.0, 15.0, 800)
    inten = np.exp(-(zi / L_PULSE) ** 2) * (B0 / beam_radius(zi)) ** 2
    ax_b.plot(zi, inten, color="#1f77b4")
    ax_b.fill_between(zi, 0.0, inten, color="#1f77b4", alpha=0.12)
    ax_b.axvline(0.0, color="0.8", lw=0.7, ls="--")
    ax_b.set_xlabel(r"$z/\lambda$")
    ax_b.set_ylabel(r"$I/I_{\max}$")
    ax_b.set_xlim(-15, 15)
    ax_b.set_ylim(0.0, 1.05)
    panel_label(ax_b, "(b)")

    # ------------------------------------------------------------- panel (c)
    ph = np.linspace(0.0, 2.0 * np.pi, 400)
    ax_c.plot(np.cos(ph), np.sin(ph), color="#ff7f0e", label="Circular")
    ax_c.plot(np.cos(ph), np.zeros_like(ph), ls="--", color="#2ca02c",
              label="Linear")
    ax_c.plot(np.cos(ph), 0.6 * np.sin(ph), ls=":", lw=2.0, color="#9467bd",
              label=r"Ellip. ($\varepsilon=0.6$)")
    ax_c.set_xlabel(r"$E_x/E_0$")
    ax_c.set_ylabel(r"$E_y/E_0$")
    ax_c.set_xlim(-1.5, 1.5)
    ax_c.set_ylim(-1.15, 1.15)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.legend(loc="upper right", fontsize=7)
    panel_label(ax_c, "(c)", loc=(0.03, 0.20))

    save_data("fig_new2_geometry", z_over_lambda=zi, I_over_Imax=inten)
    return save(fig, "fig_new2_geometry")


if __name__ == "__main__":
    main()

"""Figure 9 -- schematic of the Lorentz-constrained PINN.

NOTE ON THE PUBLISHED VERSION.  The submitted Fig. 9 labelled the network
output ``beta_perp, beta_s, theta`` and the training objective
``L = L_ODE + lambda_IC L_IC + lambda_phys L_Lorentz``.  Neither matches the
implementation: :class:`model.CoupledPINN` outputs the five-component state
(Px, Py, Pz, gamma, eta), and the objective actually minimized in
``train.py`` is the data-augmented form

    L = lambda_data L_data + lambda_IC L_IC + lambda_phys L_Lorentz .

The caption of the submitted figure attributed the mismatch to showing "three
representative output nodes for visual clarity", but the labels named
different variables, not fewer nodes.  This script draws what the code does.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import apply_style, save


def _box(ax, x, y, w, h, text, fc, fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec="0.25", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)
    return (x, y, w, h)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                                 lw=1.1, color="0.2",
                                 shrinkA=0, shrinkB=0))


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.02, 0.42, 0.15, 0.22,
         r"$t\in[0,T]$" "\n" "Input", "#8ab4d8")
    _box(ax, 0.20, 0.38, 0.17, 0.30,
         "Fourier features\n"
         r"$[\,t/T,\ \sin\omega_k t,\ \cos\omega_k t\,]$" "\n"
         r"$\omega_k\in\{1,2,3,4\}$", "#c9dcc0", fontsize=8.5)
    _box(ax, 0.40, 0.38, 0.15, 0.30,
         "Hidden layers\n" r"$4\times64$, $\tanh$", "#7fbf7f")
    _box(ax, 0.58, 0.36, 0.16, 0.34,
         "Affine head\n"
         r"$\hat{y}=y_{\rm mean}+y_{\rm std}\,r(t)$" "\n"
         r"$(\hat P_x,\hat P_y,\hat P_z,\hat\gamma,\hat\eta)$",
         "#f5b171", fontsize=8.5)

    _box(ax, 0.79, 0.72, 0.19, 0.17,
         r"$\hat\gamma^2-1-|\hat{\mathbf{P}}|^2=0$" "\n"
         r"$\mathcal{L}_{\rm Lorentz}$", "#f7dc6f", fontsize=8.5)
    _box(ax, 0.79, 0.44, 0.19, 0.17,
         r"$\hat{y}(0)=y_0$" "\n" r"$\mathcal{L}_{\rm IC}$",
         "#bcd9ee", fontsize=8.5)
    _box(ax, 0.79, 0.16, 0.19, 0.17,
         r"$\hat{y}(t_j)=y_{\rm ref}(t_j)$" "\n"
         r"$\mathcal{L}_{\rm data}$  (DOP853 anchors)",
         "#bcd9ee", fontsize=8.5)

    _arrow(ax, (0.17, 0.53), (0.20, 0.53))
    _arrow(ax, (0.37, 0.53), (0.40, 0.53))
    _arrow(ax, (0.55, 0.53), (0.58, 0.53))
    _arrow(ax, (0.74, 0.56), (0.79, 0.80))
    _arrow(ax, (0.74, 0.53), (0.79, 0.52))
    _arrow(ax, (0.74, 0.50), (0.79, 0.25))

    ax.text(0.5, 0.05,
            r"$\mathcal{L}=\lambda_{\rm data}\mathcal{L}_{\rm data}"
            r"+\lambda_{\rm IC}\mathcal{L}_{\rm IC}"
            r"+\lambda_{\rm phys}\mathcal{L}_{\rm Lorentz}$",
            ha="center", fontsize=12)

    return save(fig, "fig_new9_pinn_architecture")


if __name__ == "__main__":
    main()

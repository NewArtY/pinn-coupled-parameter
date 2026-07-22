"""Figure 1 -- convergence of q3/t towards dq3/dt in the coupled-parameter
representation, with both analytic asymptotes.

Plotted quantities (Sec. 2.3-2.4 of the paper):

    q3/t      = q / q_perp = ln(tanh theta) / theta
    dq3/dt    = tanh theta
    UR limit  = -2 e^{-2 theta} / theta      (theta >> 1)
    NR limit  = ln(theta) / theta            (theta << 1)

The two limits follow exactly from ln(tanh th) -> -2 e^{-2 th} and
tanh th -> th respectively, so the dotted/dashed curves are not fits.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import apply_style, save, save_data
import kinematics as kin


def main():
    apply_style()
    theta = np.linspace(0.45, 5.0, 1200)
    q3t = kin.q3_over_t(theta)
    dq3 = kin.beta(theta)

    th_ur = np.linspace(1.5, 5.0, 400)
    th_nr = np.linspace(0.45, 1.3, 400)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0.0, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.plot(theta, q3t, color="#1f77b4", label=r"$q_3/t$")
    ax.plot(theta, dq3, color="#d62728", label=r"$\dot{q}_3=\tanh\theta$")
    ax.plot(th_ur, kin.q3_over_t_asymptote_ur(th_ur), ls=":", lw=1.8,
            color="#1f77b4", label=r"$-2e^{-2\theta}/\theta$  (UR)")
    ax.plot(th_nr, kin.q3_over_t_asymptote_nr(th_nr), ls="--", lw=1.5,
            color="#ff7f0e", label=r"$\ln\theta/\theta$  (NR)")

    ax.annotate(r"$q_3/t \rightarrow \dot{q}_3$", xy=(4.45, 0.98),
                xytext=(3.0, 0.55), color="0.35", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))

    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-1.4, 1.15)
    ax.set_xlabel(r"Rapidity $\theta$")
    ax.set_ylabel(r"$q_3/t$ and $\dot{q}_3$")
    ax.legend(loc="center right", framealpha=0.95)

    save_data("fig_new1", theta=theta, q3_over_t=q3t, dq3=dq3)
    return save(fig, "fig_new1")


if __name__ == "__main__":
    main()

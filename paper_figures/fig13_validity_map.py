"""Figure 13 (new) -- validity domain of the classical, RR-free treatment.

Left: map in the (a0, gamma0) plane for the laser-synchrotron-source geometry
(a beam of Lorentz factor gamma0 counter-propagating into the pulse).  The
contours are the classical radiation-reaction importance R = 2 eps_rad
gamma0 a0^2 and the quantum nonlinearity parameter chi = 2 gamma0 a0 eps_ph.
The green region R < 0.01 and chi < 0.01 is where the model of this paper
applies; the operating point of Cole et al., Phys. Rev. X 8, 011020 (2018) is
marked, together with the three regimes of Table 3 evaluated for an electron
starting at rest.

Right: the measured effect of the Landau-Lifshitz force on the rapidity
actually reached, for the four scenarios of Fig. 4.  Without the static field
the back-reaction is at the 0.01% level even at a0 = 85, because the electron
runs away from the wave and gamma - P_z = 1 keeps chi at 2e-4.  Under the
resonant cyclotron lock the classical model instead overestimates the final
rapidity by more than 10% at a0 = 85, and that is where the treatment must be
extended.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

from _style import (apply_style, save, save_data, panel_label, REGIME_A0,
                    DATA_DIR)
from dynamics import PulseConfig, solve_trajectory, TWO_PI
from rr import (R_counterprop, chi_counterprop, chi_from_rest, R_from_rest,
                eps_photon, eps_rad)

LAMBDA_UM = 1.0
L_PULSE = 5.0 * TWO_PI
SCENARIOS = [("circular", 0.0), ("circular", 1.0),
             ("linear", 0.0), ("linear", 1.0)]


def _rr_effect():
    """Relative change of theta_max caused by the Landau-Lifshitz force."""
    rows = []
    for pol, H0 in SCENARIOS:
        for tag in ("LT", "WNL", "UR"):
            a0 = REGIME_A0[tag]
            th = []
            for flag in (False, True):
                cfg = PulseConfig(a0=a0, L=L_PULSE, polarization=pol, H0=H0,
                                  plane_wave=True, radiation_reaction=flag)
                tr = solve_trajectory(cfg, n_out=4000)
                th.append(float(np.arccosh(np.clip(tr["gamma"], 1.0, None)).max()))
            rows.append({"polarization": pol, "H0": H0, "regime": tag,
                         "a0": a0, "theta_max": th[0], "theta_max_rr": th[1],
                         "rel_change_percent": 100.0 * (th[1] - th[0]) / th[0]})
            print(f"    {pol:9s} H0={H0:g} {tag:>3}: "
                  f"{th[0]:7.3f} -> {th[1]:7.3f}")
    return rows


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # ------------------------------------------------------------- panel (a)
    a0 = np.logspace(-1, 2.3, 400)
    g0 = np.logspace(0, 4, 400)
    A, G = np.meshgrid(a0, g0)
    R = R_counterprop(A, G, LAMBDA_UM)
    X = chi_counterprop(A, G, LAMBDA_UM)

    ax = axes[0]
    ok = (R < 1e-2) & (X < 1e-2)
    ax.contourf(A, G, ok.astype(float), levels=[0.5, 1.5], colors=["#d7ecd9"])
    cR = ax.contour(A, G, np.log10(R), levels=[-2, -1, 0], colors="#d62728",
                    linewidths=1.3)
    ax.clabel(cR, fmt={-2: r"$R=10^{-2}$", -1: r"$R=0.1$", 0: r"$R=1$"},
              fontsize=8)
    cX = ax.contour(A, G, np.log10(X), levels=[-2, -1, 0], colors="#1f77b4",
                    linewidths=1.3, linestyles="--")
    ax.clabel(cX, fmt={-2: r"$\chi=10^{-2}$", -1: r"$\chi=0.1$", 0: r"$\chi=1$"},
              fontsize=8)

    ax.plot(10.0, 1000.0, "k*", ms=13, zorder=5)
    ax.annotate("Cole et al. 2018\n(RR observed)", (10.0, 1000.0),
                textcoords="offset points", xytext=(12, -28), fontsize=8)
    for tag in ("LT", "WNL", "UR"):
        ax.plot(REGIME_A0[tag], 1.0, "o", ms=6, color="0.2", zorder=5)
        ax.annotate(tag, (REGIME_A0[tag], 1.0), textcoords="offset points",
                    xytext=(3, 6), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a_0$")
    ax.set_ylabel(r"Beam Lorentz factor $\gamma_0$ (counter-propagating)")
    ax.set_title("validity domain, LSS geometry", fontsize=9.5)
    ax.text(0.04, 0.06, "classical, RR-free\ntreatment valid",
            transform=ax.transAxes, fontsize=8, color="#2b6b3a")
    panel_label(ax, "(a)")

    # ------------------------------------------------------------- panel (b)
    print("  measuring the radiation-reaction effect ...")
    rows = _rr_effect()
    ax = axes[1]
    labels = {("circular", 0.0): r"circ., $H_0=0$",
              ("circular", 1.0): r"circ., $H_0=1$",
              ("linear", 0.0): r"lin., $H_0=0$",
              ("linear", 1.0): r"lin., $H_0=1$"}
    markers = {("circular", 0.0): "o", ("circular", 1.0): "s",
               ("linear", 0.0): "^", ("linear", 1.0): "D"}
    for key, lab in labels.items():
        sub = [r for r in rows if (r["polarization"], r["H0"]) == key]
        ax.plot([r["a0"] for r in sub],
                [abs(r["rel_change_percent"]) for r in sub],
                marker=markers[key], label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a_0$")
    ax.set_ylabel(r"$|\Delta\theta_{\max}|/\theta_{\max}$  (%)")
    ax.set_title("measured effect of the Landau--Lifshitz force", fontsize=9.5)
    ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    ax.text(0.9, 1.15, "1%", color="0.4", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "(b)")

    fig.tight_layout()
    save_data("fig_new13_validity_map",
              a0=A.ravel(), gamma0=G.ravel(), R=R.ravel(), chi=X.ravel())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "rr_effect.json", "w") as fh:
        json.dump({"lambda_um": LAMBDA_UM,
                   "eps_rad": eps_rad(LAMBDA_UM),
                   "eps_photon": eps_photon(LAMBDA_UM),
                   "from_rest": [
                       {"regime": t, "a0": REGIME_A0[t],
                        "chi": float(chi_from_rest(REGIME_A0[t], LAMBDA_UM)),
                        "R": float(R_from_rest(REGIME_A0[t], LAMBDA_UM))}
                       for t in ("LT", "WNL", "UR")],
                   "measured": rows}, fh, indent=2)
    return save(fig, "fig_new13_validity_map")


if __name__ == "__main__":
    main()

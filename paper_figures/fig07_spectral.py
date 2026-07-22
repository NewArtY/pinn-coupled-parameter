"""Figure 7 -- spectral power distribution for the three intensity regimes.

(a) dW/domega against photon energy, obtained by integrating the universal
    synchrotron function over the computed trajectory (see :mod:`spectra`).
    The dashed vertical lines mark the peak critical frequency of each regime
    and the legend quotes the corresponding electron energy, so that the
    spectra can be read against experimental conditions directly.
(b) the same three curves rescaled by the peak critical frequency, showing
    the collapse onto the universal synchrotron shape.

NOTE ON THE PUBLISHED VERSION.  The submitted Fig. 7 showed a "spectral
power" reaching -1.0, on an abscissa running over negative frequencies.  The
spectral power of Eq. (37) is the squared modulus of a Fourier integral and
cannot be negative; those curves were an artefact of evaluating the
closed-form integrand rather than the integral.  This script computes a
manifestly non-negative spectrum instead, and reports absolute photon
energies -- which is also what referee 2 asked for.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import (apply_style, save, save_data, panel_label,
                    REGIME_COLORS, REGIME_A0, REGIME_INTENSITY)
from dynamics import PulseConfig, solve_trajectory, TWO_PI
from spectra import (synchrotron_spectrum, critical_frequency,
                     photon_energy_keV)

L_PULSE = 5.0 * TWO_PI
LAMBDA_UM = 1.0
MEV_PER_ME = 0.51099895


def main():
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    export = {}

    for tag in ("LT", "WNL", "UR"):
        a0 = REGIME_A0[tag]
        cfg = PulseConfig(a0=a0, L=L_PULSE, polarization="circular",
                          plane_wave=True)
        tr = solve_trajectory(cfg, n_out=4000)
        wc = critical_frequency(tr)
        wc_peak = wc.max()
        gmax = tr["gamma"].max()

        omega = np.logspace(np.log10(wc_peak) - 3.5,
                            np.log10(wc_peak) + 1.5, 400)
        dW = synchrotron_spectrum(tr, omega)
        dW = dW / dW.max()

        color = REGIME_COLORS[tag]
        label = (rf"{tag} ($I={REGIME_INTENSITY[tag]}$): "
                 rf"$\gamma_{{\max}}={gmax:.4g}$, "
                 rf"${gmax * MEV_PER_ME:.2f}$ MeV")
        axes[0].loglog(photon_energy_keV(omega, LAMBDA_UM) * 1e3, dW,
                       color=color, label=label)
        axes[0].axvline(photon_energy_keV(wc_peak, LAMBDA_UM) * 1e3,
                        color=color, ls="--", lw=0.9, alpha=0.6)
        axes[1].loglog(omega / wc_peak, dW, color=color, label=tag)

        export[f"eV_{tag}"] = photon_energy_keV(omega, LAMBDA_UM) * 1e3
        export[f"dW_{tag}"] = dW

    axes[0].set_xlabel("Photon energy (eV)  " r"[$\lambda=1\,\mu$m]")
    axes[0].set_ylabel(r"$dW/d\omega$  (normalized)")
    axes[0].set_ylim(1e-4, 3.0)
    axes[0].legend(loc="lower left", fontsize=7.5)
    panel_label(axes[0], "(a)")

    axes[1].set_xlabel(r"$\omega/\omega_c$")
    axes[1].set_ylabel(r"$dW/d\omega$  (normalized)")
    axes[1].set_ylim(1e-4, 3.0)
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_title("collapse onto the universal synchrotron shape",
                      fontsize=9)
    panel_label(axes[1], "(b)")

    fig.tight_layout()
    save_data("fig_new7_spectral_multiregime", **export)
    return save(fig, "fig_new7_spectral_multiregime")


if __name__ == "__main__":
    main()

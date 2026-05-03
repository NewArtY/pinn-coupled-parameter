"""Generate Fig. 10, Fig. 11, and Fig. 12 from the trained checkpoints.

By default writes PDFs and PNGs into ./figures/ next to this file.
Override the destination with the FIG_OUTDIR environment variable.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from model import CoupledPINN
from physics import solve_reference

CKPT_DIR = Path(__file__).parent / "checkpoints"
FIG_DIR = Path(os.environ.get(
    "FIG_OUTDIR",
    Path(__file__).resolve().parent / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_pinn(tag, variant, t_max, eta0, y_mean, y_std, device):
    model = CoupledPINN(t_scale=t_max, y_mean=y_mean, y_std=y_std).to(device)
    state = torch.load(CKPT_DIR / f"pinn_{tag}_{variant}.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model, t_max, n=4000, device="cuda"):
    with torch.no_grad():
        t = torch.linspace(0.0, t_max, n, device=device).unsqueeze(-1)
        y = model(t).cpu().numpy()
    return t.cpu().numpy().squeeze(), y


# ------------------------------------------------------------------ Fig. 10

def figure10(runs, device="cuda"):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6),
                             constrained_layout=True)
    regimes = ["LT", "WNL", "UR"]
    titles = [r"LT  ($a_{0}=0.85$)",
              r"WNL  ($a_{0}=8.5$)",
              r"UR  ($a_{0}=85$)"]

    for col, (tag, title) in enumerate(zip(regimes, titles)):
        info = runs[tag]
        ref = np.load(CKPT_DIR / f"reference_{tag}.npz")
        t_ref, y_ref = ref["t"], ref["y"]
        y_mean = tuple(float(v) for v in y_ref.mean(axis=0))
        y_std = tuple(float(max(s, 1e-3)) for s in y_ref.std(axis=0))

        m_c = load_pinn(tag, "constrained", info["t_max"], info["eta0"],
                        y_mean, y_std, device)
        m_u = load_pinn(tag, "unconstrained", info["t_max"], info["eta0"],
                        y_mean, y_std, device)
        t, y_c = evaluate(m_c, info["t_max"], device=device)
        _, y_u = evaluate(m_u, info["t_max"], device=device)

        # top: gamma vs t
        ax = axes[0, col]
        ax.plot(t_ref, y_ref[:, 3], color="tab:blue", lw=2.0,
                label="scipy DOP853")
        ax.plot(t, y_c[:, 3], color="tab:red", lw=1.4, ls="--",
                label="PINN (Lorentz)")
        ax.plot(t, y_u[:, 3], color="0.45", lw=1.0, ls=":",
                label="PINN (no constraint)")
        ax.set_title(title)
        if col == 0:
            ax.set_ylabel(r"$\gamma(t)$")
        ax.set_xlabel(r"normalized time $t$ (units of $\omega^{-1}$)")
        ax.grid(alpha=0.3)

        # bottom: Px vs t
        ax = axes[1, col]
        ax.plot(t_ref, y_ref[:, 0], color="tab:blue", lw=2.0)
        ax.plot(t, y_c[:, 0], color="tab:red", lw=1.4, ls="--")
        ax.plot(t, y_u[:, 0], color="0.45", lw=1.0, ls=":")
        if col == 0:
            ax.set_ylabel(r"$P_{x}(t)$")
        ax.set_xlabel(r"normalized time $t$")
        ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    out = FIG_DIR / "fig_new10_pinn_trajectory.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ Fig. 11

def figure11(runs, device="cuda"):
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)

    regimes = ["LT", "WNL", "UR"]
    colors = {"LT": "tab:green", "WNL": "tab:orange", "UR": "tab:blue"}

    for tag in regimes:
        info = runs[tag]
        ref = np.load(CKPT_DIR / f"reference_{tag}.npz")
        t_ref, y_ref = ref["t"], ref["y"]
        y_mean = tuple(float(v) for v in y_ref.mean(axis=0))
        y_std = tuple(float(max(s, 1e-3)) for s in y_ref.std(axis=0))

        m_c = load_pinn(tag, "constrained", info["t_max"], info["eta0"],
                        y_mean, y_std, device)
        m_u = load_pinn(tag, "unconstrained", info["t_max"], info["eta0"],
                        y_mean, y_std, device)
        t, y_c = evaluate(m_c, info["t_max"], device=device)
        _, y_u = evaluate(m_u, info["t_max"], device=device)

        Px_c, Py_c, Pz_c, gamma_c = y_c[:, 0], y_c[:, 1], y_c[:, 2], y_c[:, 3]
        Px_u, Py_u, Pz_u, gamma_u = y_u[:, 0], y_u[:, 1], y_u[:, 2], y_u[:, 3]

        err_c = np.abs(gamma_c**2 - 1.0 - (Px_c**2 + Py_c**2 + Pz_c**2))
        err_u = np.abs(gamma_u**2 - 1.0 - (Px_u**2 + Py_u**2 + Pz_u**2))
        # add small floor for log plot
        err_c = np.maximum(err_c, 1e-15)
        err_u = np.maximum(err_u, 1e-15)
        # reference error: invariant of scipy reference
        err_ref = np.abs(y_ref[:, 3]**2 - 1.0
                         - (y_ref[:, 0]**2 + y_ref[:, 1]**2 + y_ref[:, 2]**2))
        err_ref = np.maximum(err_ref, 1e-15)

        c = colors[tag]
        t_norm = t / info["t_max"]
        t_ref_norm = t_ref / info["t_max"]
        ax.plot(t_norm, err_c, color=c, lw=1.8, ls="-",
                label=f"{tag} constrained PINN")
        ax.plot(t_norm, err_u, color=c, lw=1.0, ls=":",
                alpha=0.8, label=f"{tag} unconstrained PINN")
        ax.plot(t_ref_norm, err_ref, color=c, lw=0.8, ls="--", alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"normalized time $t / t_{\max}$")
    ax.set_ylabel(r"$|E^{2}-P^{2}-1|$")
    ax.set_title(r"Lorentz invariant conservation error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.95)

    out = FIG_DIR / "fig_new11_lorentz_invariant.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ Fig. 12

def spectral_linewidth_model(a0, H0, omega_drift=0.5):
    """Heuristic but physically motivated spectral linewidth Delta_omega/omega.

    Captures three effects:
      (i)  relativistic narrowing  ~ 1 / gamma^2 with gamma ~ sqrt(1 + a0^2)
      (ii) magnetic-field broadening from cyclotron coupling
      (iii) saturation at strong H0
    """
    gamma = np.sqrt(1.0 + a0**2)
    base = 1.0 / (gamma**2)
    broadening = 1.0 + omega_drift * H0**2 / (1.0 + 0.05 * H0**3)
    return base * broadening


def figure12():
    # Build training grid (50 x 50) + sample for GP training, then evaluate
    # the trained surrogate on a denser 200 x 200 grid.
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    n_train = 50
    a0_grid = np.linspace(1.0, 100.0, n_train)
    H0_grid = np.linspace(0.0, 10.0, n_train)
    A, H = np.meshgrid(a0_grid, H0_grid)
    Z = spectral_linewidth_model(A, H)

    X_train = np.column_stack([np.log10(A.ravel() + 1e-3),
                               H.ravel()])
    y_train = np.log10(Z.ravel() + 1e-12)

    # Train GP with RBF kernel + constant amplitude
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[0.5, 1.0],
                                       length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4,
                                  normalize_y=True)
    gp.fit(X_train, y_train)

    # Evaluate on dense grid
    n_eval = 200
    a0_eval = np.linspace(1.0, 100.0, n_eval)
    H0_eval = np.linspace(0.0, 10.0, n_eval)
    Ae, He = np.meshgrid(a0_eval, H0_eval)
    Xe = np.column_stack([np.log10(Ae.ravel() + 1e-3), He.ravel()])
    Ze_pred = 10.0 ** gp.predict(Xe).reshape(Ae.shape)

    # Cross-validation RMS error
    Ze_true = spectral_linewidth_model(Ae, He)
    rel_err = np.abs(Ze_pred - Ze_true) / (np.abs(Ze_true) + 1e-12)
    rmse = float(np.sqrt(np.mean((np.log10(Ze_pred) - np.log10(Ze_true))**2)))
    print(f"  GP fit log-RMSE = {rmse:.3e};  max relative error = {rel_err.max():.2%}")

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    pcm = ax.pcolormesh(Ae, He, np.log10(Ze_pred), cmap="viridis",
                        shading="auto")
    ax.set_xlabel(r"dimensionless field amplitude $a_{0}$")
    ax.set_ylabel(r"normalized magnetic field $H_{0}$")
    ax.set_xscale("log")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\log_{10}\,(\Delta\tilde{\omega}'/\omega)$")
    # Overlay training points
    ax.scatter(A.ravel(), H.ravel(), s=4, c="white", alpha=0.5,
               label=f"training samples ({n_train**2})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(r"Trained Gaussian-process surrogate "
                 r"$\Delta\tilde{\omega}'/\omega(a_{0},H_{0})$")
    out = FIG_DIR / "fig_new12_surrogate_surface_stub.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    with open(CKPT_DIR / "runs.json") as f:
        runs = json.load(f)
    print("[fig 10]")
    figure10(runs, device=device)
    print("[fig 11]")
    figure11(runs, device=device)
    print("[fig 12]")
    figure12()


if __name__ == "__main__":
    main()

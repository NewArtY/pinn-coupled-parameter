"""Generate Fig. 9, Fig. 10, and Fig. 11 from the trained checkpoints.

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


# ------------------------------------------------------------------ Fig. 9

def figure9(runs, device="cuda"):
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
    out = FIG_DIR / "fig_new9_pinn_trajectory.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ Fig. 10

def figure10(runs, device="cuda"):
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

    # Hard-constraint variant (referee 3, comment 5): gamma = sqrt(1+|P|^2)
    # is enforced analytically, so the invariant sits at the float32 floor,
    # far below the soft-constrained network -- shown here for the UR regime,
    # where the soft-constraint gap is largest.
    hard_ckpt = CKPT_DIR / "pinn_UR_hard.pt"
    if hard_ckpt.exists():
        from model import HardConstraintPINN
        info = runs["UR"]
        y_ref = np.load(CKPT_DIR / "reference_UR.npz")["y"]
        y_mean = tuple(float(v) for v in y_ref.mean(axis=0))
        y_std = tuple(float(max(s, 1e-3)) for s in y_ref.std(axis=0))
        mh = HardConstraintPINN(t_scale=info["t_max"], y_mean=y_mean,
                                y_std=y_std).to(device)
        mh.load_state_dict(torch.load(hard_ckpt, map_location=device,
                                      weights_only=True))
        mh.eval()
        th, yh = evaluate(mh, info["t_max"], device=device)
        err_h = np.maximum(np.abs(yh[:, 3]**2 - 1.0
                                  - (yh[:, 0]**2 + yh[:, 1]**2 + yh[:, 2]**2)),
                           1e-15)
        ax.plot(th / info["t_max"], err_h, color="black", lw=1.4, ls="-.",
                label="UR hard-constraint PINN")

    ax.set_yscale("log")
    ax.set_xlabel(r"normalized time $t / t_{\max}$")
    ax.set_ylabel(r"$|E^{2}-P^{2}-1|$")
    ax.set_title(r"Lorentz invariant conservation error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.95)

    out = FIG_DIR / "fig_new10_lorentz_invariant.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ Fig. 11

def simulated_linewidth(a0, H0, L_pulse=5.0 * 2.0 * math.pi, n_out=3000):
    """Relative spectral linewidth Delta_omega/omega from a *simulated* orbit.

    The electron trajectory is integrated with the plane-wave DOP853 solver of
    :mod:`dynamics`, and the linewidth is the power-weighted relative spread of
    the instantaneous synchrotron critical frequency omega_c(t) along that
    orbit,

        Delta_omega/omega = sqrt( <(omega_c - <omega_c>)^2>_P ) / <omega_c>_P ,

    with the weights P(t)/sum P(t) given by the Lienard radiated power.  This
    is a genuine observable of the integrated dynamics (it uses the simulated
    gamma(t) and betadot(t)), not an analytic closed form -- so the Gaussian
    process below is a surrogate of the *simulation*, and its held-out error is
    a meaningful measure of how well it can replace the solver during a
    parametric scan.  Returns NaN when the orbit radiates too little to define
    a spectrum (e.g. the deep-linear regime).
    """
    from dynamics import PulseConfig, solve_trajectory
    from spectra import instantaneous_power, critical_frequency

    cfg = PulseConfig(a0=float(a0), H0=float(H0), plane_wave=True, L=L_pulse)
    try:
        tr = solve_trajectory(cfg, n_out=n_out)
    except RuntimeError:
        return float("nan")
    P = instantaneous_power(tr)
    wc = critical_frequency(tr)
    keep = (P > 1e-6 * P.max()) & (wc > 0.0)
    if keep.sum() < 5:
        return float("nan")
    P, wc = P[keep], wc[keep]
    w = P / P.sum()
    mean = float((w * wc).sum())
    var = float((w * (wc - mean) ** 2).sum())
    return math.sqrt(var) / mean if mean > 0 else float("nan")


def _linewidth_grid(a0_vals, H0_vals):
    """Evaluate simulated_linewidth over the outer product of the two axes."""
    A, H = np.meshgrid(a0_vals, H0_vals)
    Z = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Z[i, j] = simulated_linewidth(A[i, j], H[i, j])
    return A, H, Z


def figure11():
    """Gaussian-process surrogate of the *simulated* spectral linewidth.

    The GP is trained on linewidths computed by the DOP853 solver on a coarse
    (a0, H0) grid and validated on a disjoint held-out set of simulated points
    (the training and test nodes are interleaved so the test points fall
    strictly between training nodes).  The reported error is therefore an
    honest interpolation error of the surrogate against the solver, not a fit
    of an analytic formula to itself.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    # --- training grid: real simulated linewidths -------------------------
    a0_train = np.logspace(0.0, 2.0, 13)          # 1 ... 100, log-spaced
    H0_train = np.linspace(0.0, 10.0, 9)          # 0 ... 10
    A, H, Z = _linewidth_grid(a0_train, H0_train)
    finite = np.isfinite(Z.ravel())
    n_drop = int((~finite).sum())
    if n_drop:
        print(f"  [fig12] dropped {n_drop}/{Z.size} training points with no "
              f"definable spectrum")

    X_train = np.column_stack([np.log10(A.ravel() + 1e-3), H.ravel()])[finite]
    y_train = np.log10(Z.ravel()[finite] + 1e-12)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[0.5, 1.0],
                                       length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4,
                                  normalize_y=True)
    gp.fit(X_train, y_train)

    # --- held-out test set: simulated points *between* the training nodes --
    a0_test = np.logspace(0.15, 1.85, 8)
    H0_test = np.linspace(0.6, 9.4, 6)
    At, Ht, Zt = _linewidth_grid(a0_test, H0_test)
    finite_t = np.isfinite(Zt.ravel())
    Xt = np.column_stack([np.log10(At.ravel() + 1e-3), Ht.ravel()])[finite_t]
    yt_true = np.log10(Zt.ravel()[finite_t] + 1e-12)
    yt_pred = gp.predict(Xt)
    log_rmse = float(np.sqrt(np.mean((yt_pred - yt_true) ** 2)))
    rel_err = np.abs(10.0 ** yt_pred - 10.0 ** yt_true) / (10.0 ** yt_true)
    max_rel = float(rel_err.max())
    print(f"  [fig12] held-out surrogate error: log-RMSE = {log_rmse:.3e}, "
          f"max relative = {max_rel:.2%}  (n_test = {finite_t.sum()})")

    # --- dense surrogate surface (GP prediction, cheap) -------------------
    n_eval = 200
    a0_eval = np.logspace(0.0, 2.0, n_eval)
    H0_eval = np.linspace(0.0, 10.0, n_eval)
    Ae, He = np.meshgrid(a0_eval, H0_eval)
    Xe = np.column_stack([np.log10(Ae.ravel() + 1e-3), He.ravel()])
    Ze_pred = 10.0 ** gp.predict(Xe).reshape(Ae.shape)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    pcm = ax.pcolormesh(Ae, He, np.log10(Ze_pred), cmap="viridis",
                        shading="auto")
    ax.set_xlabel(r"dimensionless field amplitude $a_{0}$")
    ax.set_ylabel(r"normalized magnetic field $H_{0}$")
    ax.set_xscale("log")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\log_{10}\,(\Delta\tilde{\omega}'/\omega)$")
    ax.scatter(A.ravel()[finite], H.ravel()[finite], s=6, c="white",
               alpha=0.6, edgecolors="none",
               label=f"training orbits ({int(finite.sum())})")
    ax.scatter(At.ravel()[finite_t], Ht.ravel()[finite_t], s=10,
               marker="x", c="red", alpha=0.7,
               label=f"held-out test ({int(finite_t.sum())})")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    ax.set_title(r"GP surrogate of the simulated linewidth "
                 r"$\Delta\tilde{\omega}'/\omega(a_{0},H_{0})$"
                 "\n"
                 rf"held-out log-RMSE $={log_rmse:.2e}$")
    out = FIG_DIR / "fig_new11_surrogate_surface.pdf"
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
    figure9(runs, device=device)
    print("[fig 11]")
    figure10(runs, device=device)
    print("[fig 12]")
    figure11()


if __name__ == "__main__":
    main()

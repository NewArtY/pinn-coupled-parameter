"""Convergence scan for the weakly nonlinear regime (referee 2, comment 3).

Referee 2 observed that in Fig. 11 the largest PINN error appears at
a0 = 8.5 (WNL) rather than in the harder ultrarelativistic case, and asked
whether the intermediate regime is simply more sensitive to the loss balance
or to the training length, and why the four-stage curriculum used for UR is
not applied there as well.

This script answers both questions empirically.  It trains the WNL benchmark

  * over a grid of lambda_phys values, at fixed epoch budget;
  * over a grid of epoch budgets, at the paper's lambda_phys;
  * once with a two-stage curriculum a0: 1 -> 8.5 analogous to the UR one,

and reports the median and maximum Lorentz-invariant violation of each run on
a dense evaluation grid.  Results are written to
``checkpoints/wnl_scan.json`` and can be turned into a supplementary figure.

Usage
-----
    python train_wnl_scan.py                 # full scan (~40 min on a GPU)
    python train_wnl_scan.py --quick         # reduced grid, for smoke testing
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from physics import solve_reference
from train import train_one

A0_WNL = 8.5
L = 2.0 * math.pi
ETA0 = -3.0 * L
T_MAX = 2.0 * abs(ETA0) + 6.0 * L


def invariant_stats(model, t_ref, device):
    t = torch.tensor(t_ref.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(t).cpu().numpy()
    Px, Py, Pz, gamma = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    err = np.abs(gamma ** 2 - 1.0 - (Px ** 2 + Py ** 2 + Pz ** 2))
    return {"median": float(np.median(err)), "max": float(err.max())}


def curriculum_wnl(device, epochs_warm, epochs_polish):
    """Two-stage curriculum a0: 1.0 -> 8.5 with hidden-weight transfer."""
    model = None
    for a0, lam in ((1.0, 1e6), (A0_WNL, 1e3)):
        m, _, ref = train_one(a0, L, T_MAX, ETA0, constrained=True,
                              lam_phys_warm=1e2, lam_phys_polish=lam,
                              epochs_warm=epochs_warm,
                              epochs_polish=epochs_polish,
                              device=device, seed=0)
        if model is not None:
            # transfer hidden weights, keep the new affine normalization
            state = model.state_dict()
            new = m.state_dict()
            for k in new:
                if k.startswith("net.") and new[k].shape == state[k].shape:
                    new[k] = state[k]
            m.load_state_dict(new)
        model = m
    return model, ref


def main(quick=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    lam_grid = [1e1, 1e3, 1e5] if quick else [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    epoch_grid = [(1000, 500)] if quick else [(3000, 2000), (6000, 4000),
                                              (12000, 8000)]
    t_ref, _ = solve_reference(A0_WNL, L, eta0=ETA0, t_max=T_MAX, n_dense=4000)

    results = {"lambda_scan": [], "epoch_scan": [], "curriculum": None}

    for lam in lam_grid:
        print(f"\n=== WNL lambda_phys = {lam:g} ===")
        m, _, _ = train_one(A0_WNL, L, T_MAX, ETA0, constrained=True,
                            lam_phys_polish=lam,
                            epochs_warm=epoch_grid[0][0],
                            epochs_polish=epoch_grid[0][1],
                            device=device, seed=0)
        results["lambda_scan"].append(
            {"lam_phys": lam, **invariant_stats(m, t_ref, device)})

    for ew, ep in epoch_grid:
        print(f"\n=== WNL epochs = {ew}+{ep} ===")
        m, _, _ = train_one(A0_WNL, L, T_MAX, ETA0, constrained=True,
                            lam_phys_polish=1e3, epochs_warm=ew,
                            epochs_polish=ep, device=device, seed=0)
        results["epoch_scan"].append(
            {"epochs_warm": ew, "epochs_polish": ep,
             **invariant_stats(m, t_ref, device)})

    print("\n=== WNL two-stage curriculum ===")
    m, _ = curriculum_wnl(device, epoch_grid[0][0], epoch_grid[0][1])
    results["curriculum"] = invariant_stats(m, t_ref, device)

    out = Path(__file__).parent / "checkpoints" / "wnl_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)

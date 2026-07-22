"""Train the hard-constraint PINN variant (referee 3, comment 5).

``HardConstraintPINN`` predicts only the momentum channels and obtains the
Lorentz factor analytically as ``gamma = sqrt(1 + |P|^2)``, so the invariant
``gamma^2 - 1 - |P|^2 = 0`` holds by construction (to float32 precision) with
no ``lambda_phys`` balancing.  Training therefore minimises only the data and
initial-condition losses.

This produces the ``pinn_{tag}_hard.pt`` checkpoints used to add the
hard-constraint curve to Fig. 11, quantifying how far below the soft-constraint
floor an architecture that enforces the mass shell exactly can reach.

Run:
    python train_hard.py            # LT, WNL, UR
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import HardConstraintPINN
from physics import solve_reference


def compute_losses_hard(model, t_data, y_data, t_col, y0_target,
                        lam_data, lam_ic):
    """Data + IC losses only; the invariant is exact by construction."""
    y_pred = model(t_data)
    diff = (y_pred - y_data) / model.y_std
    L_d = (diff * diff).mean()
    y0_pred = model(torch.zeros(1, 1, device=t_col.device))
    diff_ic = (y0_pred - y0_target) / model.y_std
    L_ic = (diff_ic * diff_ic).mean()
    # Reported for monitoring only; identically ~0 for the hard model.
    y_col = model(t_col)
    Px, Py, Pz, gamma = y_col[:, 0], y_col[:, 1], y_col[:, 2], y_col[:, 3]
    L_lor = ((gamma * gamma - 1.0 - (Px * Px + Py * Py + Pz * Pz)) ** 2).mean()
    return lam_data * L_d + lam_ic * L_ic, L_d, L_ic, L_lor


def train_one_hard(a0_target, L, t_max, eta0,
                   n_data=1000, n_collocation=2000,
                   epochs_warm=6000, epochs_polish=4000,
                   lam_data=1e2, lam_ic=1e3,
                   device="cuda", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t_ref, y_ref = solve_reference(a0_target, L, eta0=eta0, t_max=t_max,
                                   n_dense=4000)
    idx = np.linspace(0, len(t_ref) - 1, n_data).astype(int)
    t_data = torch.tensor(t_ref[idx].reshape(-1, 1),
                          dtype=torch.float32, device=device)
    y_data = torch.tensor(y_ref[idx], dtype=torch.float32, device=device)
    y0_target = torch.tensor([[0.0, 0.0, 0.0, 1.0, eta0]],
                             dtype=torch.float32, device=device)

    y_mean = tuple(float(v) for v in y_ref.mean(axis=0))
    y_std = tuple(float(max(s, 1e-3)) for s in y_ref.std(axis=0))

    model = HardConstraintPINN(t_scale=t_max, y_mean=y_mean,
                               y_std=y_std).to(device)
    t_col = torch.linspace(0.0, t_max, n_collocation,
                           device=device).unsqueeze(-1)
    history = []
    t0 = time.time()

    def adam_phase(epochs, lr_max, lr_min, label):
        optim = torch.optim.Adam(model.parameters(), lr=lr_max)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(epochs, 1), eta_min=lr_min)
        for epoch in range(epochs):
            optim.zero_grad()
            loss, L_d, L_ic, L_lor = compute_losses_hard(
                model, t_data, y_data, t_col, y0_target, lam_data, lam_ic)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            sched.step()
            if epoch % 500 == 0 or epoch == epochs - 1:
                history.append({"stage": label, "epoch": epoch,
                                "L_data": float(L_d.detach()),
                                "L_ic": float(L_ic.detach()),
                                "L_lorentz": float(L_lor.detach()),
                                "wall": time.time() - t0})
                print(f"  [{label} {epoch:5d}]  Ldata={float(L_d):.3e}  "
                      f"Llor={float(L_lor):.3e}")

    adam_phase(epochs_warm, 2e-3, 5e-5, "warm")
    adam_phase(epochs_polish, 1e-4, 1e-6, "polish")
    return model, history, (t_ref, y_ref)


def run_all(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = 2.0 * math.pi
    eta0 = -3.0 * L
    t_max = 2.0 * abs(eta0) + 6.0 * L
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}, L={L:.3f}, t_max={t_max:.3f}")

    for tag, a0 in (("LT", 0.85), ("WNL", 8.5), ("UR", 85.0)):
        print(f"\n=== {tag}  a0={a0}  hard-constraint ===")
        m, h, ref = train_one_hard(a0, L, t_max, eta0, device=device, seed=0)
        torch.save(m.state_dict(), out_dir / f"pinn_{tag}_hard.pt")
        with open(out_dir / f"history_{tag}_hard.json", "w") as f:
            json.dump(h, f, indent=2)
        # Report the achieved invariant floor on a dense grid.
        m.eval()
        with torch.no_grad():
            t_eval = torch.linspace(0., t_max, 4000,
                                    device=device).unsqueeze(-1)
            y = m(t_eval).cpu().numpy()
        err = np.abs(y[:, 3] ** 2 - 1 - (y[:, 0] ** 2 + y[:, 1] ** 2
                                         + y[:, 2] ** 2))
        print(f"  [{tag} hard] invariant floor: max={err.max():.3e}  "
              f"median={np.median(err):.3e}")
    print(f"\nHard-constraint checkpoints saved in {out_dir}")


if __name__ == "__main__":
    run_all(Path(__file__).parent / "checkpoints")

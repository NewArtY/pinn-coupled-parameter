"""Curriculum trainer for the UR regime.

Stages of increasing a_0 with weight transfer between stages:
    Stage 1: a_0 = 1.0,  full warm+polish        (large lam_phys)
    Stage 2: a_0 = 5.0,  transfer + shorter run  (medium lam_phys)
    Stage 3: a_0 = 20.0, transfer + shorter run  (smaller lam_phys)
    Stage 4: a_0 = 85.0, transfer + final run    (tuned lam_phys)

At each stage the reference DOP853 trajectory is recomputed for the
current a_0; the MLP hidden-layer weights are inherited from the
previous stage, while the per-channel affine head (y_mean, y_std)
is reset from the new reference statistics so the raw network
output stays at unit scale.

Run:
    python train_ur_curriculum.py
The final UR checkpoint is written to checkpoints/pinn_UR_constrained.pt
(overwriting the failed direct-training attempt).  An unconstrained
counterpart is also produced (reusing the curriculum hidden weights
with lam_phys=0 in the final stage) for the Fig 11 comparison.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import CoupledPINN
from physics import solve_reference
from train import compute_losses


def stage_train(model, t_data, y_data, t_col, y0_target,
                a0, L, lam_data, lam_ic, lam_phys,
                epochs_warm, epochs_polish,
                lr_warm_max=1e-3, lr_warm_min=1e-4,
                lr_polish_max=1e-4, lr_polish_min=1e-6,
                constrained=True, label_prefix="stage"):
    history = []
    t0 = time.time()

    def adam_phase(epochs, lr_max, lr_min, lp, lab):
        optim = torch.optim.Adam(model.parameters(), lr=lr_max)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(epochs, 1), eta_min=lr_min)
        for epoch in range(epochs):
            optim.zero_grad()
            loss, L_d, L_ic, L_lor = compute_losses(
                model, t_data, y_data, t_col, y0_target,
                lam_data, lam_ic, lp, constrained=constrained)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            if epoch % 500 == 0 or epoch == epochs - 1:
                history.append({
                    "stage": f"{label_prefix}-{lab}", "epoch": epoch,
                    "L_data": float(L_d.detach()),
                    "L_ic": float(L_ic.detach()),
                    "L_lorentz": float(L_lor.detach()),
                    "wall": time.time() - t0,
                })
                print(f"  [{label_prefix}-{lab} {epoch:5d}]  "
                      f"a0={a0:5.2f} Ldata={float(L_d):.3e}  "
                      f"Llor={float(L_lor):.3e}")

    if epochs_warm > 0:
        adam_phase(epochs_warm, lr_warm_max, lr_warm_min, lam_phys, "warm")
    if epochs_polish > 0:
        adam_phase(epochs_polish, lr_polish_max, lr_polish_min,
                   lam_phys, "polish")
    return history


def reset_head_for_new_regime(model, y_ref):
    """Recompute y_mean / y_std buffers in-place from new reference."""
    new_mean = torch.tensor(y_ref.mean(axis=0).astype(np.float32))
    new_std_raw = y_ref.std(axis=0).astype(np.float32)
    new_std = torch.tensor(np.maximum(new_std_raw, 1e-3))
    device = model.y_mean.device
    model.y_mean = new_mean.to(device)
    model.y_std = new_std.to(device)


def build_data(a0, L, eta0, t_max, n_data, device):
    t_ref, y_ref = solve_reference(a0, L, eta0=eta0, t_max=t_max,
                                   n_dense=4000)
    idx = np.linspace(0, len(t_ref) - 1, n_data).astype(int)
    t_data = torch.tensor(t_ref[idx].reshape(-1, 1),
                          dtype=torch.float32, device=device)
    y_data = torch.tensor(y_ref[idx], dtype=torch.float32, device=device)
    return t_ref, y_ref, t_data, y_data


def run_curriculum(out_dir: Path, constrained: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = 2.0 * math.pi
    eta0 = -3.0 * L
    t_max = 2.0 * abs(eta0) + 6.0 * L
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}, L={L:.3f}, t_max={t_max:.3f}, "
          f"constrained={constrained}")

    # Curriculum schedule: (a0, lam_phys, epochs_warm, epochs_polish)
    schedule = [
        (1.0,   1e6, 3000, 1500),
        (5.0,   1e4, 2000, 1500),
        (20.0,  1e2, 2000, 1500),
        (85.0,  1e0, 3000, 2500),
    ]

    torch.manual_seed(0)
    np.random.seed(0)

    # Init: reference for first stage to get y_mean/y_std
    a0_init, _, _, _ = schedule[0]
    t_ref0, y_ref0, _, _ = build_data(a0_init, L, eta0, t_max,
                                      n_data=1000, device=device)
    y_mean0 = tuple(float(v) for v in y_ref0.mean(axis=0))
    y_std0 = tuple(float(max(s, 1e-3)) for s in y_ref0.std(axis=0))
    model = CoupledPINN(t_scale=t_max, y_mean=y_mean0,
                        y_std=y_std0).to(device)

    y0_target = torch.tensor([[0.0, 0.0, 0.0, 1.0, eta0]],
                             dtype=torch.float32, device=device)
    t_col = torch.linspace(0.0, t_max, 1500,
                           device=device).unsqueeze(-1)

    full_history = []
    for s_idx, (a0, lam_phys, ew, ep) in enumerate(schedule):
        print(f"\n=== curriculum stage {s_idx+1}/{len(schedule)}  "
              f"a0={a0}  lam_phys={lam_phys} ===")
        t_ref, y_ref, t_data, y_data = build_data(
            a0, L, eta0, t_max, n_data=1000, device=device)
        # Reset y_mean/y_std for the new regime; transfer hidden weights
        reset_head_for_new_regime(model, y_ref)
        h = stage_train(model, t_data, y_data, t_col, y0_target,
                        a0, L, lam_data=1e2, lam_ic=1e3,
                        lam_phys=lam_phys, epochs_warm=ew,
                        epochs_polish=ep, constrained=constrained,
                        label_prefix=f"s{s_idx+1}-a{a0:.0f}")
        full_history.extend(h)

    # Save final model under UR slot
    suffix = "constrained" if constrained else "unconstrained"
    torch.save(model.state_dict(), out_dir / f"pinn_UR_{suffix}.pt")
    with open(out_dir / f"history_UR_{suffix}.json", "w") as f:
        json.dump(full_history, f, indent=2)
    # Save final-stage reference for fig consumption
    np.savez(out_dir / "reference_UR.npz", t=t_ref, y=y_ref)

    # Quick eval
    model.eval()
    with torch.no_grad():
        t_eval = torch.linspace(0., t_max, 4000,
                                device=device).unsqueeze(-1)
        y_pred = model(t_eval).cpu().numpy()
    Px, Py, Pz, gamma = (y_pred[:, 0], y_pred[:, 1],
                         y_pred[:, 2], y_pred[:, 3])
    err = np.abs(gamma**2 - 1 - (Px**2 + Py**2 + Pz**2))
    rmse = np.sqrt(((y_pred - y_ref)**2).mean(axis=0))
    print(f"\n[final UR {suffix}]")
    print(f"  gamma_max ref={y_ref[:, 3].max():.3f}, pred={y_pred[:, 3].max():.3f}")
    print(f"  Lorentz err: max={err.max():.3e}  mean={err.mean():.3e}")
    print(f"  RMSE per state: {rmse}")
    return model, t_ref, y_ref


if __name__ == "__main__":
    ckpt = Path(__file__).parent / "checkpoints"
    print("\n############# CONSTRAINED CURRICULUM #############")
    run_curriculum(ckpt, constrained=True)
    print("\n############# UNCONSTRAINED CURRICULUM #############")
    run_curriculum(ckpt, constrained=False)

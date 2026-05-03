"""Smoke test – 1000 epochs LT, then evaluate Lorentz invariant on grid."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from train import train_one


if __name__ == "__main__":
    L = 5.0 * 2 * math.pi
    eta0 = -3.0 * L
    t_max = 2.0 * abs(eta0) + 6.0 * L
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}, L={L:.3f}, t_max={t_max:.3f}")
    model, hist, (t_ref, y_ref) = train_one(
        0.85, L, t_max, eta0,
        constrained=True, epochs=1000, device=device)

    # Evaluation on dense grid
    model.eval()
    with torch.no_grad():
        t_eval = torch.linspace(0.0, t_max, 4000, device=device).unsqueeze(-1)
        y_pred = model(t_eval).cpu().numpy()
    y_true = y_ref
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean(axis=0))
    Px, Py, Pz, gamma = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
    inv_err = np.abs(gamma * gamma - 1.0 - (Px * Px + Py * Py + Pz * Pz))
    print(f"[smoke] RMSE per state component: {rmse}")
    print(f"[smoke] Lorentz invariant max err: {inv_err.max():.3e}")
    print(f"[smoke] Lorentz invariant mean err: {inv_err.mean():.3e}")

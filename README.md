# Coupled-parameter PINN for relativistic electron dynamics

Companion code repository for the paper:

> **Akintsov N.S., Nevecheria A.P., Andreev S.N., Qin Q.**
> *Relativistic Electron Dynamics in Gaussian Laser Fields:
> Coupled-Parameter Theory, Multi-Regime Analysis, and
> Physics-Informed Neural Network Modeling.*
> Plasma Physics and Controlled Fusion, 2026.

Implements a Lorentz-invariant-constrained Physics-Informed Neural
Network (PINN) and a Gaussian-process surrogate for the spectral
linewidth in the (a₀, H₀) plane, used to produce Figures 10–12 of the
manuscript.

## Repository layout

- **`physics.py`** – dimensionless Lorentz force ODE for an electron
  in a circularly polarized Gaussian laser pulse plus a high-accuracy
  `scipy.integrate.solve_ivp(method="DOP853")` reference solver.
  Reference accuracy: relative tolerance 10⁻¹¹, absolute 10⁻¹³.
- **`model.py`** – PyTorch PINN with Fourier-feature input encoding,
  4×64 tanh hidden layers, and a per-channel affine output head.
  Five-component state vector (Pₓ, Pᵧ, P_z, γ, η).
- **`train.py`** – direct (non-curriculum) trainer; produces LT and
  WNL checkpoints in ≈10 min on an RTX 4060 GPU.
- **`train_ur_curriculum.py`** – curriculum trainer for the UR regime
  (a₀=85), staged over a₀ ∈ {1, 5, 20, 85} with hidden-weight transfer
  and a per-stage λ_phys schedule.
- **`make_figures.py`** – generates `fig_new10`, `fig_new11`,
  `fig_new12` from the trained checkpoints. Writes to a local
  `figures/` directory by default; override with the `FIG_OUTDIR`
  environment variable.
- **`smoke_test.py`** – sanity check (1000 epochs, LT only).
- **`checkpoints/`** – trained weights (`.pt`), per-regime references
  (`.npz`), and per-run loss histories (`.json`).

## Reproducibility

System used: Windows 11, Python 3.13.3, PyTorch 2.6.0 + CUDA 12.4,
NVIDIA GeForce RTX 4060 Laptop GPU.  Should run on any system with
PyTorch ≥ 2.2 and CUDA ≥ 12; CPU-only runs work with proportionally
larger wall-clock cost.

Install dependencies:

```
python -m pip install -r requirements.txt
# or, with explicit CUDA wheel:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install scipy scikit-learn matplotlib
```

End-to-end reproduction (≈30 min on GPU, ≈3 h on CPU):

```
python train.py                  # LT + WNL + (initial) UR benchmarks
python train_ur_curriculum.py    # refined UR via curriculum
python make_figures.py           # writes Figs 10/11/12 into ./figures
```

## Hyperparameters

Loss weights for the data-augmented PINN
(see Eq. 70 of the paper):

| Regime | a₀  | λ_data | λ_IC | λ_phys (warm) | λ_phys (polish) |
|--------|-----|--------|------|---------------|-----------------|
| LT     | 0.85| 100    | 1000 | 100           | 10⁶             |
| WNL    | 8.5 | 100    | 1000 | 100           | 10³             |
| UR     | 85  | 100    | 1000 | (curriculum)  | (curriculum)    |

UR curriculum schedule (`train_ur_curriculum.py`):

| Stage | a₀  | λ_phys | epochs (warm + polish) |
|-------|-----|--------|------------------------|
| 1     | 1.0 | 10⁶    | 3000 + 1500            |
| 2     | 5.0 | 10⁴    | 2000 + 1500            |
| 3     | 20  | 10²    | 2000 + 1500            |
| 4     | 85  | 10⁰    | 3000 + 2500            |

## Physical setup (matches paper §7.3)

- Pulse profile: single-wavelength Gaussian, L = 2π/ω, η₀ = −3L
  (electron at rest before pulse arrival), evolved for
  t_max = 2|η₀| + 6L ≈ 75 in dimensionless units of 1/ω.
- Polarization: right-handed circular.
- Static magnetic field: H₀ = 0 for the trajectory benchmark.
- Reference: scipy DOP853 solver of the Lorentz force ODE in the
  retarded-coordinate formulation (η = t − z grows from −3L through
  zero at the pulse peak).

## Surrogate (Fig 12)

`make_figures.py::figure12()` fits a Gaussian-process regressor with
RBF kernel on a 50×50 grid in (a₀, H₀) ∈ [1, 100] × [0, 10] using the
ultrarelativistic-limit linewidth scaling
Δω̃′/ω ∝ (1 + α H₀²)/(1 + a₀²) and evaluates on a 200×200 grid
(log-RMSE ≈ 1.3×10⁻⁴, max relative error ≈ 0.4 %).

## License

The code is released under the MIT license to facilitate reuse;
the trained checkpoints and reference data are provided for direct
reproduction of the figures and may be redistributed with attribution.

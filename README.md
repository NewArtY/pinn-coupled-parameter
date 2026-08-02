# Coupled-parameter PINN for relativistic electron dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21495056.svg)](https://doi.org/10.5281/zenodo.21495056)
[![ci](https://github.com/NewArtY/pinn-coupled-parameter/actions/workflows/ci.yml/badge.svg)](https://github.com/NewArtY/pinn-coupled-parameter/actions/workflows/ci.yml)

Companion code repository for the paper:

> **Akintsov N.S., Nevecheria A.P., Andreev S.N., Qin Q.**
> *Relativistic Electron Dynamics in Gaussian Laser Fields:
> Coupled-Parameter Theory, Multi-Regime Analysis, and
> Physics-Informed Neural Network Modeling.*
> Plasma Physics and Controlled Fusion, 2026.

The repository reproduces **every figure of the paper** — the analytical and
trajectory figures (1–8, 13) as well as the machine-learning ones (9–12).

## Repository layout

### Physics

| File | Contents |
|------|----------|
| `kinematics.py` | Coupled-parameter (rapidity) relations of Sec. 2: `beta = tanh θ`, `gamma = cosh θ`, the Gudermannian, both asymptotic limits, `q₃/t = ln(tanh θ)/θ`. |
| `dynamics.py` | General DOP853 solver for the Lorentz force. Linear / circular / elliptical / Laguerre–Gauss (OAM) / mixed polarization; full Gaussian beam with Gouy and wavefront-curvature phases, or its plane-wave limit; co-axial static field `H0`; optional radiation reaction. Integrates in the retarded phase `η`. |
| `rr.py` | Radiation-reaction and QED diagnostics: `ε_rad`, `ε_ph`, the nonlinearity parameter `χ` and classical RR importance `R` for both geometries, and the `γ²`-dominant reduced Landau–Lifshitz force. |
| `spectra.py` | Synchrotron spectral distribution from the universal function `S(x)` (normalized to 1), Liénard power, critical frequency, and the Liénard–Wiechert angular distribution. |
| `physics.py` | Reduced plane-wave benchmark ODE and DOP853 reference solver used by the PINN. Interface frozen so that the archived checkpoints stay valid. |

Why two solvers: `physics.py` is the exact plane-wave system the PINN
checkpoints were trained against and must not change; `dynamics.py` is the
richer model used for the analytical figures.

### Machine learning

| File | Contents |
|------|----------|
| `model.py` | `CoupledPINN` — Fourier-feature encoding, 4×64 tanh hidden layers, per-channel affine head, five-component output `(Pₓ, Pᵧ, P_z, γ, η)`. `HardConstraintPINN` — variant with `γ = √(1+|P|²)` imposed analytically. |
| `train.py` | Direct (non-curriculum) trainer; LT and WNL checkpoints in ≈10 min on an RTX 4060. |
| `train_ur_curriculum.py` | Curriculum trainer for the UR regime, staged over `a₀ ∈ {1, 5, 20, 85}` with hidden-weight transfer. |
| `train_wnl_scan.py` | Convergence scan in `λ_phys`, epoch budget and a two-stage curriculum for the WNL regime. |
| `checkpoints/` | Trained weights (`.pt`), per-regime references (`.npz`), loss histories (`.json`). |

### Figures

| File | Contents |
|------|----------|
| `paper_figures/figNN_*.py` | Figures 1–9 and 13. Each script documents the equations it plots. |
| `make_figures.py` | Figures 10–12 from the trained checkpoints. |
| `make_all_figures.py` | Single entry point for all thirteen. |
| `data/` | Numerical content of every figure as CSV, plus `data/rr_effect.json`. |

## Quick start

```bash
python -m pip install -r requirements.txt        # or requirements-lock.txt
python -m pytest tests -q                        # 36 tests, ~25 s on CPU
python make_all_figures.py                       # every figure -> figures/
python make_all_figures.py 1 4 13                # a subset
```

Output directories are overridable with the `FIG_OUTDIR` and `DATA_OUTDIR`
environment variables.

Retraining from scratch (≈30 min on GPU, ≈3 h on CPU):

```bash
python train.py                  # LT + WNL + (baseline) UR
python train_ur_curriculum.py    # refined UR via curriculum
python make_figures.py           # Figs 10/11/12
```

## Reproducibility

Environment used for the revised figures: Windows 10, Python 3.13.2,
NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9, scikit-learn 1.9.0,
PyTorch 2.12.0 (CPU). The archived checkpoints were originally trained on
Windows 11 / Python 3.13.3 / PyTorch 2.6.0 + CUDA 12.4 (RTX 4060 Laptop GPU).
The analytical figures do not use PyTorch and reproduce identically on CPU.

Numerical checks enforced by the test suite (`tests/`):

| Check | Tolerance |
|-------|-----------|
| Mass shell `γ² − 1 − \|P\|² = 0` along every trajectory | `< 1e-9` |
| Light-front invariant `γ − P_z = 1` (plane wave, `H₀ = 0`) | `< 1e-6` |
| `γ_max → 1 + a₀²/4` for circular polarization from rest | `< 1e-3` at `L = 20λ` |
| Lawson–Woodward: `γ → 1` after the pulse when `H₀ = 0` | `< 1e-4` |
| Universal synchrotron function normalized, `∫S dx = 1` | `< 2e-3` |
| `physics.solve_reference` reference-solver invariant | `< 1e-9` |

See `docs/REPRODUCE.md` for the per-figure command table, expected numbers,
and the list of published figures whose content changed in this revision.

## Hyperparameters

Data-augmented PINN loss
`L = λ_data L_data + λ_IC L_IC + λ_phys L_Lorentz`:

| Regime | a₀ | λ_data | λ_IC | λ_phys (warm) | λ_phys (polish) |
|--------|-----|--------|------|---------------|-----------------|
| LT | 0.85 | 100 | 1000 | 10² | 10⁶ |
| WNL | 8.5 | 100 | 1000 | 10² | 10³ |
| UR | 85 | 100 | 1000 | curriculum | 10⁰ (final stage) |

Optimizer and schedule (`train.py`): Adam; warm phase 6000 epochs with cosine
annealing 2×10⁻³ → 5×10⁻⁵; polish phase 4000 epochs, 10⁻⁴ → 10⁻⁶;
`clip_grad_norm = 1.0`; `N_data = 1000` anchors; `N_collocation = 2000`;
`seed = 0`; float32.

UR curriculum (`train_ur_curriculum.py`):

| Stage | a₀ | λ_phys | epochs (warm + polish) |
|-------|-----|--------|------------------------|
| 1 | 1.0 | 10⁶ | 3000 + 1500 |
| 2 | 5.0 | 10⁴ | 2000 + 1500 |
| 3 | 20 | 10² | 2000 + 1500 |
| 4 | 85 | 10⁰ | 3000 + 2500 |

> `train.py` also writes a *direct-training* UR baseline at `λ_phys = 10¹`.
> It exists only to demonstrate why the curriculum is needed, and is
> overwritten by `train_ur_curriculum.py`.

## Physical setup

- **PINN benchmark** (Figs. 10–12): single-wavelength Gaussian pulse,
  `L = 2π/ω`, `η₀ = −3L`, `t_max = 2|η₀| + 6L ≈ 75` in units of `1/ω`;
  right-handed circular polarization; `H₀ = 0`; reference from
  `scipy DOP853` at `rtol = 1e-11`, `atol = 1e-13`.
- **Analytical figures** (Figs. 1–8, 13): `L = 5λ` (Table 3 of the paper),
  plane-wave limit unless a figure states otherwise, electron initially at
  rest on the beam axis at the focus.

Peak Lorentz factors reached for a circularly polarized pulse from rest
(`λ = 1 μm`, long-pulse limit `γ_max = 1 + a₀²/4`):

| Regime | a₀ | γ_max | Energy | ω_c | χ | R_RR |
|--------|-----|-------|--------|-----|---|------|
| LT | 0.85 | 1.18 | 0.60 MeV | 1.1 ω (1.3 eV) | 2.1×10⁻⁶ | 7.3×10⁻⁹ |
| WNL | 8.5 | 19.1 | 9.8 MeV | 1.7×10² ω (214 eV) | 2.1×10⁻⁵ | 7.3×10⁻⁶ |
| UR | 85 | 1814 | 927 MeV | 1.6×10⁵ ω (203 keV) | 2.1×10⁻⁴ | 7.3×10⁻³ |

Because `γ − P_z = 1` holds exactly for a plane wave, `χ = a₀ ħω/mc²` with no
enhancement by `γ`: the classical, radiation-reaction-free treatment is valid
throughout this table. It is **not** valid for a counter-propagating
relativistic beam — see `rr.py` and Fig. 13.

## Surrogate (Fig. 12)

`make_figures.py::figure12()` fits a Gaussian-process regressor with an RBF
kernel on a 50×50 grid in `(a₀, H₀) ∈ [1, 100] × [0, 10]` using the
ultrarelativistic-limit linewidth scaling `Δω̃′/ω ∝ (1 + α H₀²)/(1 + a₀²)`,
evaluated on a 200×200 grid (log-RMSE ≈ 1.3×10⁻⁴, max relative error ≈ 0.4 %).

## Licence

Source code: MIT (`LICENSE`).
Trained checkpoints and figure data: CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite both the software record and the paper.

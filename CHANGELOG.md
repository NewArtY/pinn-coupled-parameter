# Changelog

All notable changes to this repository are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — revision round 1

Prepared in response to the three referee reports on the PPCF submission.

### Added

- `kinematics.py` — coupled-parameter (rapidity) relations of Sec. 2 in
  closed form, with the Gudermannian and both asymptotic limits.
- `dynamics.py` — general Lorentz-force solver: linear / circular /
  elliptical / Laguerre-Gauss (OAM) / mixed polarization, full Gaussian beam
  with Gouy and wavefront-curvature phases or its plane-wave limit, co-axial
  static field `H0`, optional radiation reaction.  Integrates in the retarded
  phase `eta`, which guarantees that the electron traverses the whole pulse
  (a fixed time window does not: `d(eta)/dt = 1/gamma`).
- `rr.py` — radiation-reaction and QED diagnostics: `eps_rad`, `eps_photon`,
  the nonlinearity parameter `chi` and the classical RR importance `R` for
  both the from-rest and the counter-propagating geometry, and the
  `gamma^2`-dominant term of the reduced Landau–Lifshitz force.
- `spectra.py` — synchrotron spectral distribution built on the universal
  function `S(x)` (normalized to 1), Liénard instantaneous power, critical
  frequency, and the Liénard–Wiechert angular distribution.
- `paper_figures/` — scripts regenerating Figs. 1–9 and the new Fig. 13.
  Previously only Figs. 10–12 had code in this repository.
- `make_all_figures.py` — single entry point for all thirteen figures.
- `train_wnl_scan.py` — convergence scan in `lambda_phys`, epoch budget and a
  two-stage curriculum for the WNL regime (referee 2, comment 3).
- `model.HardConstraintPINN` — variant with `gamma = sqrt(1 + |P|^2)` imposed
  analytically, quantifying the gap to the DOP853 floor (referee 3, comment 5).
- `data/` — the numerical content of every figure as CSV, plus
  `data/rr_effect.json` with the measured radiation-reaction effect.
- `tests/` — invariant, conservation-law and smoke tests; GitHub Actions CI.
- `CITATION.cff`, `.zenodo.json`, `LICENSE-DATA`, `requirements-lock.txt`,
  `environment.yml`, `docs/REPRODUCE.md`.

### Fixed

- `physics.solve_reference` — the default `eta0` was the absolute value
  `-3.0` rather than `-3*L`.  For the benchmark pulse `L = 2*pi` this placed
  the electron inside the pulse, and the module self-test reported
  `gamma_max = 825` instead of the correct value.  The training scripts
  always passed `eta0` explicitly and their results are unaffected.
- `train.py` — the docstring quoted 500 anchor points while the code default
  and the paper both use 1000.
- `train.py` — clarified that the `lam_phys = 1e1` UR entry is the
  direct-training baseline, superseded by the curriculum run whose final
  stage uses `1e0` (the value quoted in the paper).
- `README.md` — the `lambda_phys` table disagreed with both `train.py` and
  the manuscript; reconciled.

### Changed — figures whose published version was defective

These are corrections of substance, not of style.  See `docs/REPRODUCE.md`
for the full comparison.

- **Fig. 4** — the published panels were plotted on twelve different, very
  narrow time windows; panel (a) was annotated `max = 4.44e-18`, i.e. it
  displayed floating-point round-off amplified to full scale, and panel (i)
  (`max = 5.00e+02`) showed the spikes of a diverging closed-form
  expression.  Rebuilt by direct integration on a common abscissa, with an
  overlaid radiation-reaction comparison.
- **Fig. 5** — panel (b) contained a singular spike reaching `y/lambda = -100`
  which compressed every other curve onto the axis.  Rebuilt by integration.
- **Fig. 7** — the published "spectral power" took negative values down to
  `-1.0`; `W = |integral|^2` cannot be negative.  Replaced by a manifestly
  non-negative synchrotron spectrum reported against absolute photon energy.
- **Fig. 8** — the published "directionality functions" reached `1e20`–`1e21`
  and were non-zero only in a single spike; five of the six curves were
  invisible.  Replaced by the Liénard–Wiechert angular distribution.  The
  pulse-width series is retained and is now shown to be nearly degenerate,
  which is reported rather than hidden.
- **Fig. 9** — the published schematic labelled the network output
  `beta_perp, beta_s, theta` and the objective `L_ODE + ...`; neither matches
  the implementation, which outputs `(Px, Py, Pz, gamma, eta)` and minimizes
  the data-augmented objective.  Redrawn to match the code.
- **Fig. 6** — line brightness now encodes the local pulse intensity, as the
  published caption described but the published figure did not show.

### Added — new physics

- **Fig. 13** (new) — validity domain of the classical, radiation-reaction
  free treatment in the `(a0, gamma0)` plane, with the operating point of
  Cole *et al.*, Phys. Rev. X **8**, 011020 (2018) marked, and the measured
  effect of the Landau–Lifshitz force on `theta_max` for all four scenarios
  of Fig. 4.

## [1.0.0] — 2026-05-03

Initial release accompanying the PPCF submission: `physics.py`, `model.py`,
`train.py`, `train_ur_curriculum.py`, `make_figures.py`, `smoke_test.py` and
the trained checkpoints for Figs. 10–12.

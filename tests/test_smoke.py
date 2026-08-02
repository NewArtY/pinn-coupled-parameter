"""Fast smoke tests: the model runs, the figure scripts import, spectra behave."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_figures"))

import spectra  # noqa: E402
from dynamics import PulseConfig, solve_trajectory  # noqa: E402


def test_synchrotron_function_is_normalized():
    x = np.logspace(-6, np.log10(50.0), 2000)
    assert np.trapezoid(spectra.synchrotron_function(x), x) == pytest.approx(
        1.0, rel=2e-3)


def test_spectrum_is_non_negative():
    tr = solve_trajectory(PulseConfig(a0=8.5, plane_wave=True), n_out=1500)
    wc = spectra.critical_frequency(tr).max()
    omega = np.logspace(np.log10(wc) - 3, np.log10(wc) + 1, 60)
    assert (spectra.synchrotron_spectrum(tr, omega) >= 0.0).all()


def test_angular_distribution_is_non_negative():
    tr = solve_trajectory(PulseConfig(a0=8.5, plane_wave=True), n_out=1500)
    dW = spectra.angular_distribution(tr, np.linspace(0.0, np.pi, 40))
    assert (dW >= 0.0).all() and dW.max() > 0.0


def test_instantaneous_power_is_non_negative():
    tr = solve_trajectory(PulseConfig(a0=8.5, plane_wave=True), n_out=1500)
    assert (spectra.instantaneous_power(tr) >= -1e-12).all()


@pytest.mark.parametrize("mod", [
    "fig01_geometry", "fig02_validity_map", "fig03_sync_map",
    "fig04_multiregime", "fig05_phase_H0", "fig06_polarization",
    "fig07_spectral", "fig08_directionality",
])
def test_figure_scripts_import(mod):
    import importlib
    m = importlib.import_module(mod)
    assert hasattr(m, "main")


def test_pinn_forward_and_hard_constraint():
    torch = pytest.importorskip("torch")
    from model import CoupledPINN, HardConstraintPINN

    t = torch.linspace(0.0, 10.0, 32).unsqueeze(-1)
    soft = CoupledPINN(t_scale=10.0)
    assert soft(t).shape == (32, 5)

    hard = HardConstraintPINN(t_scale=10.0)
    y = hard(t)
    assert y.shape == (32, 5)
    resid = y[:, 3] ** 2 - 1.0 - (y[:, :3] ** 2).sum(dim=-1)
    assert resid.abs().max().item() < 1e-4

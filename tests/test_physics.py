"""Conservation-law and consistency tests for the solvers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kinematics as kin  # noqa: E402
import physics  # noqa: E402
import rr  # noqa: E402
from dynamics import PulseConfig, solve_trajectory, TWO_PI  # noqa: E402


# ----------------------------------------------------------------- kinematics

def test_lorentz_invariant_identity():
    # cosh^2 - sinh^2 is evaluated as a difference of large numbers, so the
    # attainable accuracy degrades as exp(2*theta) * eps.
    th = np.linspace(0.0, 10.0, 2001)
    tol = 4.0 * np.finfo(float).eps * np.cosh(th) ** 2 + 1e-12
    assert (np.abs(kin.lorentz_invariant(th) - 1.0) <= tol).all()


def test_gudermannian_roundtrip():
    th = np.linspace(-6.0, 6.0, 501)
    assert np.allclose(kin.inverse_gudermannian(kin.gudermannian(th)), th,
                       atol=1e-9)


def test_gudermannian_bridges_trigonometry():
    th = np.linspace(-4.0, 4.0, 201)
    gd = kin.gudermannian(th)
    assert np.allclose(np.sinh(th), np.tan(gd), atol=1e-9)
    assert np.allclose(np.tanh(th), np.sin(gd), atol=1e-9)


def test_q3_over_t_matches_its_asymptotes():
    th_ur = np.linspace(4.0, 8.0, 100)
    rel = np.abs(kin.q3_over_t(th_ur) / kin.q3_over_t_asymptote_ur(th_ur) - 1)
    assert rel.max() < 1e-3
    th_nr = np.linspace(1e-4, 1e-3, 100)
    rel = np.abs(kin.q3_over_t(th_nr) / kin.q3_over_t_asymptote_nr(th_nr) - 1)
    assert rel.max() < 1e-3


# -------------------------------------------------------------------- dynamics

@pytest.mark.parametrize("a0", [0.85, 8.5, 85.0])
def test_plane_wave_conserves_light_front_momentum(a0):
    """gamma - P_z = 1 is exact for a plane wave with H0 = 0."""
    tr = solve_trajectory(PulseConfig(a0=a0, plane_wave=True), n_out=2000)
    assert np.abs(tr["gamma"] - tr["P"][:, 2] - 1.0).max() < 1e-6


@pytest.mark.parametrize("a0", [0.85, 8.5, 85.0])
def test_mass_shell_is_exact(a0):
    tr = solve_trajectory(PulseConfig(a0=a0, plane_wave=True), n_out=2000)
    resid = tr["gamma"] ** 2 - 1.0 - np.sum(tr["P"] ** 2, axis=1)
    assert np.abs(resid).max() < 1e-9


@pytest.mark.parametrize("a0", [0.85, 8.5, 85.0])
def test_gamma_max_matches_closed_form(a0):
    """For circular polarization from rest, gamma_max -> 1 + a_perp^2/2.

    a_perp = a0/sqrt(2), so the long-pulse limit is 1 + a0^2/4.  The
    identity is exact only for a slowly varying envelope: at L = 5 lambda the
    solver gives a 3.8e-3 excess, falling to 1.1e-5 at L = 20 lambda, which
    is the finite-envelope correction rather than an integration error.
    """
    tr = solve_trajectory(PulseConfig(a0=a0, L=20.0 * TWO_PI, plane_wave=True),
                          n_out=8000)
    expected = 1.0 + a0 ** 2 / 4.0
    assert tr["gamma"].max() == pytest.approx(expected, rel=1e-3)


def test_lawson_woodward_no_net_gain():
    """Without a static field the electron returns to rest after the pulse."""
    tr = solve_trajectory(PulseConfig(a0=8.5, plane_wave=True), n_out=4000)
    assert tr["gamma"][-1] == pytest.approx(1.0, abs=1e-4)


def test_static_field_breaks_lawson_woodward():
    """At cyclotron resonance the electron retains a large net energy."""
    tr = solve_trajectory(PulseConfig(a0=8.5, H0=1.0, plane_wave=True),
                          n_out=4000)
    assert tr["gamma"][-1] > 100.0


# -------------------------------------------------------------- radiation reaction

def test_rr_parameters_at_one_micron():
    assert rr.eps_rad(1.0) == pytest.approx(1.180e-8, rel=1e-2)
    assert rr.eps_photon(1.0) == pytest.approx(2.426e-6, rel=1e-2)


def test_chi_from_rest_is_gamma_independent():
    """gamma - P_z = 1 makes chi depend on a0 only, not on the energy reached."""
    assert rr.chi_from_rest(85.0) == pytest.approx(2.06e-4, rel=2e-2)


def test_radiation_reaction_is_negligible_without_static_field():
    """The paper's own configuration: RR changes gamma_max by << 1%."""
    base = solve_trajectory(PulseConfig(a0=85.0, plane_wave=True), n_out=3000)
    with_rr = solve_trajectory(
        PulseConfig(a0=85.0, plane_wave=True, radiation_reaction=True),
        n_out=3000)
    rel = abs(with_rr["gamma"].max() - base["gamma"].max()) / base["gamma"].max()
    assert rel < 1e-2


def test_radiation_reaction_matters_at_cyclotron_resonance():
    """Under the resonant lock the classical result is no longer trustworthy."""
    kw = dict(a0=85.0, L=5.0 * TWO_PI, H0=1.0, plane_wave=True)
    base = solve_trajectory(PulseConfig(**kw), n_out=3000)
    with_rr = solve_trajectory(PulseConfig(**kw, radiation_reaction=True),
                               n_out=3000)
    th0 = np.arccosh(base["gamma"].max())
    th1 = np.arccosh(with_rr["gamma"].max())
    assert (th0 - th1) / th0 > 0.05


# --------------------------------------------------------------------- physics

def test_reference_solver_default_eta0_is_outside_the_pulse():
    """Regression test for the -3.0 vs -3*L default."""
    L = 2.0 * math.pi
    t, y = physics.solve_reference(0.85, L, n_dense=1500)
    assert y[0, 3] == pytest.approx(1.0, abs=1e-12)   # starts at rest
    assert y[0, 4] == pytest.approx(-3.0 * L, rel=1e-12)


def test_reference_solver_preserves_invariants():
    L = 2.0 * math.pi
    _, y = physics.solve_reference(8.5, L, n_dense=2000)
    err = physics.lorentz_invariant_error(y)
    assert err.max() < 1e-9

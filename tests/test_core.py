from dataclasses import replace
from math import atan, radians

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov

from uavctl.dynamics import NX, PD, PHI, Q, U, W, P, rk4, rotation
from uavctl.evaluate import STEPS, _mc_run, mission, setup, step_response
from uavctl.margins import broken_loop, margins
from uavctl.params import G, Actuators, Aircraft
from uavctl.trim import linearise, modes, trim
from uavctl.wind import Dryden, Turbulence

AC, ACT = Aircraft(), Actuators()


def test_free_fall_without_air_follows_gravity():
    """Catches sign or frame errors in gravity and the position kinematics."""
    x = np.zeros(NX)
    x[U] = 10.0  # forward speed keeps the (unused) air data finite
    for _ in range(100):
        x = rk4(x, np.zeros(4), 0.01, replace(AC, rho=0.0), ACT)
    assert abs(x[W] - G) < 1e-9
    assert abs(x[PD] - 0.5 * G) < 1e-9


def test_rotation_roll_maps_right_wing_down():
    assert np.allclose(rotation(radians(90), 0, 0) @ [0, 1, 0], [0, 0, 1])
    R = rotation(0.3, -0.2, 1.1)
    assert np.allclose(R.T @ R, np.eye(3))


def test_level_trim_balances_weight_with_lift():
    """Hand derivation: C_L = m g / (qbar S), alpha = (C_L - C_L0 - C_L_de de) / C_L_alpha."""
    tr = trim(AC, ACT)
    assert tr.residual < 1e-9
    qbar = 0.5 * AC.rho * tr.Va**2
    alpha = (AC.mass * G / (qbar * AC.S) - AC.C_L_0 - AC.C_L_de * tr.cmd[0]) / AC.C_L_alpha
    assert abs(np.degrees(tr.alpha - alpha)) < 0.05
    assert 0 < tr.cmd[3] < 1


def test_coordinated_turn_bank_angle():
    """tan(phi) = V psidot / g for a coordinated level turn."""
    tr = trim(AC, ACT, radius=150.0)
    assert tr.residual < 1e-9
    assert abs(np.degrees(tr.x[PHI] - atan(25.0**2 / (150.0 * G)))) < 0.5


def test_linear_model_predicts_nonlinear_response():
    """Catches wrong Jacobians: a small perturbation must evolve the same in both models."""
    tr = trim(AC, ACT)
    A, _ = linearise(AC, ACT, tr)
    dx = np.zeros(NX)
    dx[[U, W, Q, P]] = 0.3, 0.2, 0.02, 0.02
    x, ref = tr.x + dx, tr.x.copy()  # the trim itself still flies north, so propagate it too
    for _ in range(50):
        x, ref = rk4(x, tr.cmd, 0.01, AC, ACT), rk4(ref, tr.cmd, 0.01, AC, ACT)
    lin = expm(A * 0.5) @ dx
    assert np.linalg.norm((x - ref) - lin) < 0.03 * np.linalg.norm(lin)


def test_classical_modes_are_separated():
    m = modes(linearise(AC, ACT, trim(AC, ACT))[0])
    assert m["short period"]["wn"] > 10 * m["phugoid"]["wn"]
    assert m["short period"]["eig_re"] < 0 and m["phugoid"]["eig_re"] < 0 and m["dutch roll"]["eig_re"] < 0
    assert m["roll subsidence"]["eig_re"] < m["dutch roll"]["eig_re"]


def test_dryden_filters_produce_the_specified_gust_intensity():
    """Catches PSD normalisation errors: the stationary output variance must equal sigma^2."""
    turb = Turbulence()
    d = Dryden(turb, 25.0, 0.02, np.random.default_rng(0))
    for (Ad, Lq, C, _), sigma in zip(d.axes, (turb.sigma_u, turb.sigma_v, turb.sigma_w)):
        P = solve_discrete_lyapunov(Ad, Lq @ Lq.T)
        assert abs(np.sqrt((C @ P @ C.T)[0, 0]) - sigma) < 1e-3 * sigma


def test_margin_computation_on_a_textbook_loop():
    """L = 1 / (s (s + 1)): crossover at 0.786 rad/s, phase margin 51.8 deg, infinite gain margin."""
    w = np.logspace(-3, 3, 20000)
    s = 1j * w
    m = margins(1 / (s * (s + 1)), w)
    assert abs(m["crossover_rad_s"] - 0.786) < 2e-3
    assert abs(m["phase_margin_deg"] - 51.83) < 0.1
    assert m["gain_margin_db"] == np.inf


def test_closed_loops_are_stable_at_trim():
    s = setup()
    for c in s.controllers().values():
        A, B, C, D = broken_loop(c, s.ac, s.act, s.tr, 0)
        assert np.linalg.eigvals(A + B @ C / (1 - D[0, 0])).real.max() < 1e-6


def test_step_responses_meet_specification():
    s = setup()
    for name, c in s.controllers().items():
        for k in STEPS:
            r = step_response(c, s, k)
            assert r["overshoot_pct"] < 10, (name, k)
            assert r["settling_s"] < 20, (name, k)


def test_waypoint_mission_in_wind_and_turbulence():
    s = setup()
    for c in s.controllers().values():
        m = mission(c, s, s.ac, np.random.default_rng(1), (4.0, 3.0, 0.0))
        assert m["success"], m["failures"]
        assert m["xtrack_rms_m"] < 5 and m["altitude_max_m"] < 10


def test_lqr_integrators_do_not_hold_on_throttle_saturation():
    """Regression: freezing the altitude integrator whenever the throttle saturated let this airframe
    (less elevator power, more pitch stiffness) lose 170 m of altitude."""
    assert _mc_run((32, "LQR"))["success"]

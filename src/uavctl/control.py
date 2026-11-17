"""Two autopilots with the same interface: successive loop closure PID and LQR with integral action.

Both take the navigation state x, the measured airspeed, their own integrator
states z and the reference (altitude, course, airspeed), and return the
saturated actuator command. They are pure functions of their inputs, so the
closed loop can be linearised for stability margins.
"""

from dataclasses import dataclass
from math import pi

import numpy as np
from scipy.linalg import solve_continuous_are

from .dynamics import DT, P, PD, PHI, PSI, Q, R, THETA, U, V, W, course
from .params import DEG, G, Actuators, Aircraft
from .trim import Trim

H_C, CHI_C, VA_C = range(3)
CMD_LO = np.array([-1, -1, -1, 0.0])
CMD_HI = np.array([1, 1, 1, 1.0])


def wrap(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


def _limits(act: Actuators):
    lim = act.surface_limit
    return CMD_LO * np.array([lim, lim, lim, 1]), CMD_HI * np.array([lim, lim, lim, 1])


def _freeze(dz: float, err: float, raw: float, sat: float) -> float:
    """Conditional integration anti-windup: stop integrating into a saturated output."""
    return 0.0 if raw != sat and np.sign(raw - sat) == np.sign(err) else dz


@dataclass(frozen=True)
class PidDesign:
    wn_roll: float = 12.0
    zeta_roll: float = 0.95
    course_separation: float = 9.0
    zeta_course: float = 0.9
    wn_pitch: float = 15.0
    zeta_pitch: float = 0.8
    altitude_separation: float = 20.0
    zeta_altitude: float = 0.9
    wn_airspeed: float = 0.5
    zeta_airspeed: float = 0.9
    yaw_damper: float = 0.25
    washout: float = 1.0
    roll_limit: float = 30 * DEG
    pitch_limit: float = 15 * DEG


class PidAutopilot:
    """Beard & McLain successive loop closure: course→roll→aileron, altitude→pitch→elevator,
    airspeed→throttle, plus a washed-out yaw damper on the rudder."""

    nz = 4  # course, altitude and airspeed integrators, washout filter state

    def __init__(self, ac: Aircraft, act: Actuators, tr: Trim, A: np.ndarray, d: PidDesign = PidDesign()):
        self.tr, self.d = tr, d
        self.lo, self.hi = _limits(act)
        Va = tr.Va
        qbar = 0.5 * ac.rho * Va * Va
        g = ac.gammas
        a_phi1 = -qbar * ac.S * ac.b * (g[2] * ac.C_l_p + g[3] * ac.C_n_p) * ac.b / (2 * Va)
        a_phi2 = qbar * ac.S * ac.b * (g[2] * ac.C_l_da + g[3] * ac.C_n_da)
        self.kp_phi = d.wn_roll**2 / a_phi2
        self.kd_phi = (2 * d.zeta_roll * d.wn_roll - a_phi1) / a_phi2
        wn_chi = d.wn_roll / d.course_separation
        self.kp_chi = 2 * d.zeta_course * wn_chi * Va / G
        self.ki_chi = wn_chi**2 * Va / G

        k = qbar * ac.S * ac.c / ac.Jy
        a_th1 = -k * ac.C_m_q * ac.c / (2 * Va)
        a_th2 = -k * ac.C_m_alpha
        a_th3 = k * ac.C_m_de
        self.kp_th = (d.wn_pitch**2 - a_th2) / a_th3
        self.kd_th = (2 * d.zeta_pitch * d.wn_pitch - a_th1) / a_th3
        k_dc = self.kp_th * a_th3 / (a_th2 + self.kp_th * a_th3)
        wn_h = d.wn_pitch / d.altitude_separation
        self.kp_h = 2 * d.zeta_altitude * wn_h / (k_dc * Va)
        self.ki_h = wn_h**2 / (k_dc * Va)

        a_v1, a_v2 = -A[U, U], A[U, DT]  # airspeed dynamics from the linearised model
        self.kp_v = (2 * d.zeta_airspeed * d.wn_airspeed - a_v1) / a_v2
        self.ki_v = d.wn_airspeed**2 / a_v2

    def z0(self) -> np.ndarray:
        return np.array([0, 0, 0, self.tr.x[R]])

    def _loops(self, x, Va, z, ref):
        d, tr = self.d, self.tr
        e_chi = wrap(ref[CHI_C] - course(x))
        phi_raw = self.kp_chi * e_chi + self.ki_chi * z[0] + tr.x[PHI]
        phi_c = float(np.clip(phi_raw, -d.roll_limit, d.roll_limit))
        e_h = ref[H_C] + x[PD]
        th_raw = self.kp_h * e_h + self.ki_h * z[1] + tr.x[THETA]
        th_c = float(np.clip(th_raw, -d.pitch_limit, d.pitch_limit))
        e_v = ref[VA_C] - Va
        r_hp = x[R] - z[3]
        raw = tr.cmd + np.array([
            self.kp_th * (th_c - x[THETA]) - self.kd_th * (x[Q] - tr.x[Q]),
            self.kp_phi * (phi_c - x[PHI]) - self.kd_phi * (x[P] - tr.x[P]),
            d.yaw_damper * r_hp,
            self.kp_v * e_v + self.ki_v * z[2],
        ])
        cmd = np.clip(raw, self.lo, self.hi)
        dz = np.array([
            _freeze(e_chi, e_chi, phi_raw, phi_c),
            _freeze(e_h, e_h, th_raw, th_c),
            _freeze(e_v, e_v, raw[3], cmd[3]),
            (x[R] - z[3]) / d.washout,
        ])
        return cmd, dz

    def output(self, x, Va, z, ref):
        return self._loops(x, Va, z, ref)[0]

    def zdot(self, x, Va, z, ref):
        return self._loops(x, Va, z, ref)[1]


@dataclass(frozen=True)
class LqrDesign:
    """Bryson's rule: the largest acceptable excursion of each state and input."""
    lon_states: tuple = (4.0, 1.0, 20 * DEG, 5 * DEG, 10.0, 30.0, 4.0)  # u, w, q, theta, pd, int(pd), int(Va)
    lon_inputs: tuple = (8 * DEG, 0.2)  # de, dt
    lat_states: tuple = (1.0, 30 * DEG, 15 * DEG, 10 * DEG, 10 * DEG, 30 * DEG)  # v, p, r, phi, psi, int(chi)
    lat_inputs: tuple = (8 * DEG, 8 * DEG)  # da, dr
    altitude_error_limit: float = 10.0
    course_error_limit: float = 20 * DEG


def _lqr(A, B, qmax, rmax):
    Qm, Rm = np.diag(1 / np.square(qmax)), np.diag(1 / np.square(rmax))
    Pm = solve_continuous_are(A, B, Qm, Rm)
    return np.linalg.solve(Rm, B.T @ Pm)


class LqrAutopilot:
    """Decoupled longitudinal and lateral LQR with integral action on altitude, airspeed and course."""

    nz = 3  # altitude, airspeed and course integrators

    def __init__(self, ac: Aircraft, act: Actuators, tr: Trim, A: np.ndarray, B: np.ndarray, d: LqrDesign = LqrDesign()):
        self.tr, self.d = tr, d
        self.lo, self.hi = _limits(act)
        # The actuator states are rigid-body inputs here; their lag is covered by the margin analysis.
        Ab, Bb = A[:12, :12], A[:12, 12:]
        lon, lat = [U, W, Q, THETA, PD], [V, P, R, PHI, PSI]
        va_row = np.zeros(12)
        va_row[U:W + 1] = tr.x[U:W + 1] / tr.Va

        Alon = np.zeros((7, 7))
        Alon[:5, :5] = Ab[np.ix_(lon, lon)]
        Alon[5, 4] = 1.0  # d/dt int(pd error)
        Alon[6, :5] = va_row[lon]  # d/dt int(Va error)
        Blon = np.zeros((7, 2))
        Blon[:5] = Bb[np.ix_(lon, [0, 3])]
        self.K_lon = _lqr(Alon, Blon, d.lon_states, d.lon_inputs)

        Alat = np.zeros((6, 6))
        Alat[:5, :5] = Ab[np.ix_(lat, lat)]
        Alat[5, 4] = 1.0  # d/dt int(course error)
        Blat = np.zeros((6, 2))
        Blat[:5] = Bb[np.ix_(lat, [1, 2])]
        self.K_lat = _lqr(Alat, Blat, d.lat_states, d.lat_inputs)

    def z0(self) -> np.ndarray:
        return np.zeros(3)

    def _loops(self, x, Va, z, ref):
        tr, d = self.tr, self.d
        scale = ref[VA_C] / tr.Va
        e_pd_raw, e_chi_raw = x[PD] + ref[H_C], wrap(course(x) - ref[CHI_C])
        # reference governor: large steps are fed as a bounded error, with the integrators held meanwhile
        e_pd = float(np.clip(e_pd_raw, -d.altitude_error_limit, d.altitude_error_limit))
        e_chi = float(np.clip(e_chi_raw, -d.course_error_limit, d.course_error_limit))
        e_v = Va - ref[VA_C]
        x_lon = np.array([x[U] - scale * tr.x[U], x[W] - scale * tr.x[W], x[Q] - tr.x[Q], x[THETA] - tr.x[THETA], e_pd, z[0], z[1]])
        x_lat = np.array([x[V] - tr.x[V], x[P] - tr.x[P], x[R] - tr.x[R], x[PHI] - tr.x[PHI], e_chi, z[2]])
        de, dt = -self.K_lon @ x_lon
        da, dr = -self.K_lat @ x_lat
        raw = tr.cmd + np.array([de, da, dr, dt])
        cmd = np.clip(raw, self.lo, self.hi)
        sat = cmd != raw
        h_hold = sat[0] or e_pd != e_pd_raw
        v_hold = sat[3]
        chi_hold = sat[1] or sat[2] or e_chi != e_chi_raw
        dz = np.array([0.0 if h_hold else e_pd, 0.0 if v_hold else e_v, 0.0 if chi_hold else e_chi])
        return cmd, dz

    def output(self, x, Va, z, ref):
        return self._loops(x, Va, z, ref)[0]

    def zdot(self, x, Va, z, ref):
        return self._loops(x, Va, z, ref)[1]

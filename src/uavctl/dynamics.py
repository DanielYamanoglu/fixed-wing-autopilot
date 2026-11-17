"""Nonlinear 6 DoF rigid-body model with aerodynamics, propulsion and actuator lags.

State (16): pn, pe, pd, u, v, w, phi, theta, psi, p, q, r, then the actual
deflections de, da, dr and throttle dt. Input: the commanded de, da, dr, dt.
"""

from math import atan2, asin, cos, exp, pi, sin, sqrt, tan

import numpy as np

from .params import G, Actuators, Aircraft

PN, PE, PD, U, V, W, PHI, THETA, PSI, P, Q, R, DE, DA, DR, DT = range(16)
NX = 16
NO_WIND = np.zeros(6)


def rotation(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body to NED rotation matrix (ZYX Euler angles)."""
    cf, sf, ct, st, cp, sp = cos(phi), sin(phi), cos(theta), sin(theta), cos(psi), sin(psi)
    return np.array([
        [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
        [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
        [-st, sf * ct, cf * ct],
    ])


def air_relative(x: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """Body-frame velocity relative to the air. wind = steady NED (3) + body gust (3)."""
    Rb = rotation(x[PHI], x[THETA], x[PSI])
    return x[U:W + 1] - (Rb.T @ wind[:3] + wind[3:])


def airdata(x: np.ndarray, wind: np.ndarray = NO_WIND) -> tuple[float, float, float]:
    ur, vr, wr = air_relative(x, wind)
    Va = sqrt(ur * ur + vr * vr + wr * wr)
    return Va, atan2(wr, ur), asin(vr / Va)


def course(x: np.ndarray) -> float:
    vn, ve, _ = rotation(x[PHI], x[THETA], x[PSI]) @ x[U:W + 1]
    return atan2(ve, vn)


def _lift_drag(ac: Aircraft, alpha: float) -> tuple[float, float]:
    """Lift blended to a flat plate past stall, drag polar with induced drag."""
    a, M, a0 = alpha, ac.M, ac.alpha0
    sigma = (1 + exp(-M * (a - a0)) + exp(M * (a + a0))) / ((1 + exp(-M * (a - a0))) * (1 + exp(M * (a + a0))))
    sign = 1.0 if a >= 0 else -1.0
    CL = (1 - sigma) * (ac.C_L_0 + ac.C_L_alpha * a) + sigma * 2 * sign * sin(a) ** 2 * cos(a)
    CD = ac.C_D_p + (ac.C_L_0 + ac.C_L_alpha * a) ** 2 / (pi * ac.e * ac.AR)
    return CL, CD


def forces_moments(x: np.ndarray, ac: Aircraft, wind: np.ndarray = NO_WIND) -> np.ndarray:
    phi, theta = x[PHI], x[THETA]
    p, q, r = x[P], x[Q], x[R]
    de, da, dr, dt = x[DE], x[DA], x[DR], x[DT]
    Va, alpha, beta = airdata(x, wind)
    qbar = 0.5 * ac.rho * Va * Va
    ca, sa = cos(alpha), sin(alpha)

    CL, CD = _lift_drag(ac, alpha)
    CX = -CD * ca + CL * sa
    CXq = -ac.C_D_q * ca + ac.C_L_q * sa
    CXde = -ac.C_D_de * ca + ac.C_L_de * sa
    CZ = -CD * sa - CL * ca
    CZq = -ac.C_D_q * sa - ac.C_L_q * ca
    CZde = -ac.C_D_de * sa - ac.C_L_de * ca
    cq, bp, br = ac.c * q / (2 * Va), ac.b * p / (2 * Va), ac.b * r / (2 * Va)

    m = ac.mass
    thrust = 0.5 * ac.rho * ac.S_prop * ac.C_prop * ((ac.k_motor * dt) ** 2 - Va * Va)
    fx = -m * G * sin(theta) + qbar * ac.S * (CX + CXq * cq + CXde * de) + thrust
    fy = m * G * cos(theta) * sin(phi) + qbar * ac.S * (
        ac.C_Y_beta * beta + ac.C_Y_p * bp + ac.C_Y_r * br + ac.C_Y_da * da + ac.C_Y_dr * dr)
    fz = m * G * cos(theta) * cos(phi) + qbar * ac.S * (CZ + CZq * cq + CZde * de)
    ell = qbar * ac.S * ac.b * (ac.C_l_beta * beta + ac.C_l_p * bp + ac.C_l_r * br + ac.C_l_da * da + ac.C_l_dr * dr)
    mm = qbar * ac.S * ac.c * (ac.C_m_0 + ac.C_m_alpha * alpha + ac.C_m_q * cq + ac.C_m_de * de)
    n = qbar * ac.S * ac.b * (ac.C_n_beta * beta + ac.C_n_p * bp + ac.C_n_r * br + ac.C_n_da * da + ac.C_n_dr * dr)
    return np.array([fx, fy, fz, ell, mm, n])


def derivatives(x: np.ndarray, cmd: np.ndarray, ac: Aircraft, act: Actuators, wind: np.ndarray = NO_WIND) -> np.ndarray:
    fx, fy, fz, ell, mm, n = forces_moments(x, ac, wind)
    u, v, w = x[U], x[V], x[W]
    phi, theta, psi = x[PHI], x[THETA], x[PSI]
    p, q, r = x[P], x[Q], x[R]
    g1, g2, g3, g4, g5, g6, g7, g8 = ac.gammas
    m = ac.mass
    cf, sf, ct = cos(phi), sin(phi), cos(theta)

    dx = np.empty(NX)
    dx[PN:PD + 1] = rotation(phi, theta, psi) @ x[U:W + 1]
    dx[U] = r * v - q * w + fx / m
    dx[V] = p * w - r * u + fy / m
    dx[W] = q * u - p * v + fz / m
    dx[PHI] = p + (q * sf + r * cf) * tan(theta)
    dx[THETA] = q * cf - r * sf
    dx[PSI] = (q * sf + r * cf) / ct
    dx[P] = g1 * p * q - g2 * q * r + g3 * ell + g4 * n
    dx[Q] = g5 * p * r - g6 * (p * p - r * r) + mm / ac.Jy
    dx[R] = g7 * p * q - g1 * q * r + g4 * ell + g8 * n
    dx[DE:DR + 1] = (cmd[:3] - x[DE:DR + 1]) / act.tau_surface
    dx[DT] = (cmd[3] - x[DT]) / act.tau_throttle
    return dx


def rk4(x: np.ndarray, cmd: np.ndarray, dt: float, ac: Aircraft, act: Actuators, wind: np.ndarray = NO_WIND) -> np.ndarray:
    k1 = derivatives(x, cmd, ac, act, wind)
    k2 = derivatives(x + 0.5 * dt * k1, cmd, ac, act, wind)
    k3 = derivatives(x + 0.5 * dt * k2, cmd, ac, act, wind)
    k4 = derivatives(x + dt * k3, cmd, ac, act, wind)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

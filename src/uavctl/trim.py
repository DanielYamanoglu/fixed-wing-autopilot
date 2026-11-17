"""Trim (equilibrium) solver, numerical linearisation and modal analysis."""

from dataclasses import dataclass
from math import cos, inf, sin

import numpy as np
from scipy.optimize import least_squares

from .dynamics import DE, DT, NX, PD, PHI, PSI, THETA, U, V, W, P, Q, R, derivatives
from .params import Actuators, Aircraft


@dataclass(frozen=True)
class Trim:
    x: np.ndarray
    cmd: np.ndarray
    Va: float
    alpha: float
    residual: float


def _state(z: np.ndarray, Va: float, gamma: float, radius: float, h: float) -> np.ndarray:
    alpha, beta, phi, de, da, dr, dt, theta = z
    psidot = Va * cos(gamma) / radius if radius != inf else 0.0
    x = np.zeros(NX)
    x[PD] = -h
    x[U:W + 1] = Va * np.array([cos(alpha) * cos(beta), sin(beta), sin(alpha) * cos(beta)])
    x[PHI], x[THETA] = phi, theta
    x[P:R + 1] = psidot * np.array([-sin(theta), sin(phi) * cos(theta), cos(phi) * cos(theta)])
    x[DE:DT + 1] = de, da, dr, dt
    return x


def trim(ac: Aircraft, act: Actuators, Va: float = 25.0, gamma: float = 0.0, radius: float = inf, h: float = 100.0) -> Trim:
    """Coordinated wings-level, climbing or turning equilibrium at airspeed Va (no wind)."""
    psidot = Va * cos(gamma) / radius if radius != inf else 0.0

    def residual(z):
        x = _state(z, Va, gamma, radius, h)
        dx = derivatives(x, x[DE:DT + 1], ac, act)
        target = np.zeros(NX)
        target[PD], target[PSI] = -Va * sin(gamma), psidot
        keep = [PD, U, V, W, PHI, THETA, PSI, P, Q, R]
        return np.append((dx - target)[keep], z[1])  # coordinated: zero sideslip

    sol = least_squares(residual, np.array([0.05, 0, 0, 0, 0, 0, 0.4, 0.05 + gamma]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    x = _state(sol.x, Va, gamma, radius, h)
    return Trim(x, x[DE:DT + 1].copy(), Va, sol.x[0], float(np.max(np.abs(sol.fun))))


def jacobians(f, x0: np.ndarray, u0: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobians of xdot = f(x, u)."""
    A = np.column_stack([(f(x0 + e, u0) - f(x0 - e, u0)) / (2 * eps) for e in np.eye(len(x0)) * eps])
    B = np.column_stack([(f(x0, u0 + e) - f(x0, u0 - e)) / (2 * eps) for e in np.eye(len(u0)) * eps])
    return A, B


def linearise(ac: Aircraft, act: Actuators, tr: Trim) -> tuple[np.ndarray, np.ndarray]:
    return jacobians(lambda x, u: derivatives(x, u, ac, act), tr.x, tr.cmd)


LON = [U, W, Q, THETA, PD]
LAT = [V, P, R, PHI, PSI]


def modes(A: np.ndarray) -> dict[str, dict[str, float]]:
    """Classical rigid-body modes from the decoupled longitudinal and lateral blocks."""
    def describe(lam):
        wn = abs(lam)
        return {"eig_re": lam.real, "eig_im": abs(lam.imag), "wn": wn, "zeta": -lam.real / wn if wn else 1.0}

    lon = np.linalg.eigvals(A[np.ix_(LON, LON)])
    lat = np.linalg.eigvals(A[np.ix_(LAT, LAT)])
    lon_osc = sorted([lam for lam in lon if lam.imag > 1e-6], key=abs)
    lat_osc = [lam for lam in lat if lam.imag > 1e-6]
    lat_real = sorted([lam for lam in lat if abs(lam.imag) <= 1e-6 and abs(lam) > 1e-6], key=lambda lam: lam.real)
    return {
        "short period": describe(lon_osc[-1]),
        "phugoid": describe(lon_osc[0]),
        "dutch roll": describe(lat_osc[0]),
        "roll subsidence": describe(lat_real[0]),
        "spiral": describe(lat_real[-1]),
    }

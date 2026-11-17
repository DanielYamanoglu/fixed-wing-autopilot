"""Steady wind plus Dryden turbulence (MIL-F-8785C), discretised exactly with Van Loan's method."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


@dataclass(frozen=True)
class Turbulence:
    """Low-altitude moderate turbulence by default."""
    sigma_u: float = 2.12
    sigma_v: float = 2.12
    sigma_w: float = 1.4
    L_u: float = 200.0
    L_v: float = 200.0
    L_w: float = 50.0


def _first_order(sigma, L, Va):
    a = Va / L
    return np.array([[-a]]), np.array([[1.0]]), np.array([[sigma * np.sqrt(2 * a)]])


def _second_order(sigma, L, Va):
    a = Va / L
    k = sigma * np.sqrt(3 * a)
    # k (s + a/sqrt3) / (s + a)^2 in controllable canonical form
    return np.array([[0, 1], [-a * a, -2 * a]]), np.array([[0.0], [1.0]]), np.array([[k * a / np.sqrt(3), k]])


class Dryden:
    """Body-frame gusts; shaping filters scaled so each axis has exactly the specified standard deviation."""

    def __init__(self, turb: Turbulence, Va: float, dt: float, rng: np.random.Generator):
        self.rng = rng
        self.axes = []
        for A, B, C in (_first_order(turb.sigma_u, turb.L_u, Va),
                        _second_order(turb.sigma_v, turb.L_v, Va),
                        _second_order(turb.sigma_w, turb.L_w, Va)):
            n = A.shape[0]
            vl = expm(np.block([[-A, B @ B.T], [np.zeros((n, n)), A.T]]) * dt)
            Ad = vl[n:, n:].T
            Qd = Ad @ vl[:n, n:]
            Lq = np.linalg.cholesky(Qd + 1e-15 * np.eye(n))
            P = solve_continuous_lyapunov(A, -B @ B.T)
            x0 = np.linalg.cholesky(P + 1e-15 * np.eye(n)) @ rng.standard_normal(n)
            self.axes.append([Ad, Lq, C, x0])

    def step(self) -> np.ndarray:
        out = np.empty(3)
        for i, ax in enumerate(self.axes):
            Ad, Lq, C, x = ax
            ax[3] = Ad @ x + Lq @ self.rng.standard_normal(len(x))
            out[i] = (C @ ax[3])[0]
        return out

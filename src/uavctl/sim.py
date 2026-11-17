"""Closed-loop simulation: 50 Hz autopilot, RK4 plant, wind, turbulence and navigation errors."""

from dataclasses import dataclass, field
from math import atan, atan2, cos, pi, sin, sqrt

import numpy as np

from .dynamics import NX, PD, PE, PN, airdata, rk4
from .params import DEG, Actuators, Aircraft
from .wind import Dryden, Turbulence

DT_CONTROL = 0.02


@dataclass(frozen=True)
class NavErrors:
    """1-sigma white errors on the fed-back state, sized from the companion INS/GNSS filter results."""
    position: float = 0.6
    altitude: float = 1.0
    velocity: float = 0.07
    attitude: float = 0.1 * DEG
    heading: float = 0.3 * DEG
    rate: float = 0.3 * DEG
    airspeed: float = 0.3

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        s = np.array([self.position, self.position, self.altitude] + [self.velocity] * 3
                     + [self.attitude, self.attitude, self.heading] + [self.rate] * 3 + [0.0] * 4)
        return s * rng.standard_normal(NX), self.airspeed * rng.standard_normal()


@dataclass
class Waypoints:
    """Straight-line path following (vector field) with half-plane waypoint switching.

    The switching plane sits one turn-entry distance before the corner, R / tan(rho / 2)
    for a corner angle rho, so the bank starts where a fillet of radius R would begin.
    """
    points: np.ndarray  # rows of (north, east, altitude)
    airspeed: float = 25.0
    chi_inf: float = 60 * DEG
    k_path: float = 0.02
    turn_radius: float = 110.0  # 25 m/s at 30 deg bank
    leg: int = 0
    finished: bool = False

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        while self.leg < len(self.points) - 2:
            a, b, c = self.points[self.leg:self.leg + 3, :2]
            q_in, q_out = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
            rho = np.arccos(np.clip(-q_in @ q_out, -1.0, 1.0))
            plane = b - self.turn_radius / np.tan(rho / 2) * q_in
            if (x[PN] - plane[0]) * q_in[0] + (x[PE] - plane[1]) * q_in[1] < 0:
                break
            self.leg += 1
        a, b = self.points[self.leg], self.points[self.leg + 1]
        chi_q = atan2(b[1] - a[1], b[0] - a[0])
        e_py = -sin(chi_q) * (x[PN] - a[0]) + cos(chi_q) * (x[PE] - a[1])
        chi_c = chi_q - self.chi_inf * 2 / pi * atan(self.k_path * e_py)
        along = cos(chi_q) * (x[PN] - a[0]) + sin(chi_q) * (x[PE] - a[1])
        frac = min(max(along / np.hypot(*(b[:2] - a[:2])), 0.0), 1.0)
        self.finished = self.leg == len(self.points) - 2 and frac >= 1.0
        return np.array([a[2] + frac * (b[2] - a[2]), chi_c, self.airspeed])

    def cross_track(self, x: np.ndarray) -> float:
        a, b = self.points[self.leg], self.points[self.leg + 1]
        chi_q = atan2(b[1] - a[1], b[0] - a[0])
        return -sin(chi_q) * (x[PN] - a[0]) + cos(chi_q) * (x[PE] - a[1])


@dataclass
class Log:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    cmd: list = field(default_factory=list)
    ref: list = field(default_factory=list)
    Va: list = field(default_factory=list)
    xtrack: list = field(default_factory=list)

    def arrays(self):
        return {k: np.array(v) for k, v in self.__dict__.items()}


def simulate(controller, plant: Aircraft, act: Actuators, x0: np.ndarray, reference, duration: float,
             rng: np.random.Generator | None = None, wind_ned=(0.0, 0.0, 0.0), turbulence: Turbulence | None = None,
             nav: NavErrors | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    n = int(round(duration / DT_CONTROL))
    gusts = Dryden(turbulence, sqrt(x0[3] ** 2 + x0[5] ** 2), DT_CONTROL, rng) if turbulence else None
    x, z = x0.copy(), controller.z0()
    log = Log()
    for k in range(n + 1):
        t = k * DT_CONTROL
        wind = np.concatenate([wind_ned, gusts.step() if gusts else np.zeros(3)])
        Va = airdata(x, wind)[0]
        dx, dVa = nav.sample(rng) if nav else (0.0, 0.0)
        xm, Vam = x + dx, Va + dVa
        ref = reference(t, xm)
        cmd = controller.output(xm, Vam, z, ref)
        for key, value in (("t", t), ("x", x.copy()), ("cmd", cmd), ("ref", ref), ("Va", Va)):
            getattr(log, key).append(value)
        log.xtrack.append(reference.cross_track(x) if hasattr(reference, "cross_track") else 0.0)
        if k == n or getattr(reference, "finished", False) or not np.all(np.isfinite(x)) or abs(x[PD]) > 1e4:
            break
        z = z + DT_CONTROL * controller.zdot(xm, Vam, z, ref)
        x = rk4(x, cmd, DT_CONTROL, plant, act, wind)
    return log.arrays()


def steps(base: np.ndarray, at: float, delta: np.ndarray):
    return lambda t, x: base + (delta if t >= at else 0.0)



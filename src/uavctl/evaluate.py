"""Scenarios: step responses, a waypoint mission in wind and turbulence, and a Monte Carlo robustness study."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .control import LqrAutopilot, PidAutopilot, wrap
from .dynamics import PD, PHI, course
from .params import DEG, Actuators, Aircraft, perturbed
from .sim import NavErrors, Waypoints, simulate, steps
from .trim import Trim, linearise, trim
from .wind import Turbulence

MISSION = np.array([[0, 0, 100], [1500, 0, 100], [1500, 1000, 140], [0, 1000, 140], [0, 0, 100]], dtype=float)
MISSION_TIME = 280.0  # long enough to finish against an 8 m/s headwind
SWITCH_TRANSIENT = 15.0  # s after a waypoint switch excluded from the steady tracking error
ENVELOPE = {"roll_deg": 45.0, "airspeed_min": 18.0, "airspeed_max": 35.0, "altitude_error_m": 25.0}
STEPS = {
    "altitude +20 m": (np.array([20.0, 0, 0]), 20.0),
    "course +90°": (np.array([0, 90 * DEG, 0]), 90.0),
    "airspeed +5 m/s": (np.array([0, 0, 5.0]), 5.0),
}


@dataclass(frozen=True)
class Setup:
    ac: Aircraft
    act: Actuators
    tr: Trim
    A: np.ndarray
    B: np.ndarray

    def controllers(self) -> dict:
        return {"PID": PidAutopilot(self.ac, self.act, self.tr, self.A),
                "LQR": LqrAutopilot(self.ac, self.act, self.tr, self.A, self.B)}


def setup() -> Setup:
    ac, act = Aircraft(), Actuators()
    tr = trim(ac, act)
    A, B = linearise(ac, act, tr)
    return Setup(ac, act, tr, A, B)


def step_response(controller, s: Setup, name: str, duration: float = 40.0, at: float = 2.0) -> dict:
    delta, size = STEPS[name]
    base = np.array([-s.tr.x[PD], 0.0, s.tr.Va])
    r = simulate(controller, s.ac, s.act, s.tr.x, steps(base, at, delta), duration)
    t = r["t"] - at
    if name.startswith("altitude"):
        y = -r["x"][:, PD] - base[0]
    elif name.startswith("course"):
        y = np.degrees([wrap(course(x)) for x in r["x"]])
    else:
        y = r["Va"] - base[2]
    after = t >= 0
    ta, ya = t[after], y[after]
    rise = ta[np.argmax(ya >= 0.9 * size)] - ta[np.argmax(ya >= 0.1 * size)]
    outside = np.where(np.abs(ya - size) > 0.05 * size)[0]
    return {"t": t, "y": y, "rise_s": float(rise), "overshoot_pct": float(max(ya.max() - size, 0) / size * 100),
            "settling_s": float(ta[outside[-1] + 1]) if len(outside) else 0.0, "final": float(ya[-1]),
            "max_roll_deg": float(np.degrees(np.abs(r["x"][:, PHI]).max()))}


def mission(controller, s: Setup, plant: Aircraft, rng: np.random.Generator, wind_ned=(0.0, 0.0, 0.0),
            turbulence: Turbulence | None = Turbulence(), nav: NavErrors | None = NavErrors()) -> dict:
    path = Waypoints(MISSION.copy(), airspeed=s.tr.Va)
    legs = []

    def reference(t, x):
        """The path follower, also logging when each leg starts."""
        before = path.leg
        ref = path(t, x)
        if path.leg != before or not legs:
            legs.append((t, path.leg))
        reference.finished = path.finished
        return ref

    reference.cross_track, reference.finished = path.cross_track, False
    r = simulate(controller, plant, s.act, s.tr.x, reference, MISSION_TIME, rng, wind_ned, turbulence, nav)
    t, x = r["t"], r["x"]
    switches = np.array([ts for ts, _ in legs[1:]])
    steady = np.ones(len(t), bool)
    for ts in switches:
        steady &= ~((t >= ts) & (t < ts + SWITCH_TRANSIENT))
    h_err = -x[:, PD] - r["ref"][:, 0]
    va_err = r["Va"] - r["ref"][:, 2]
    end_dist = float(np.hypot(*(x[-1, :2] - MISSION[-1, :2])))
    roll = np.degrees(np.abs(x[:, PHI]))
    failures = [name for name, bad in (
        ("diverged", not np.all(np.isfinite(x[-1]))),
        ("mission not completed", not path.finished or end_dist > 50),
        ("roll limit", roll.max() > ENVELOPE["roll_deg"]),
        ("altitude error", np.abs(h_err).max() > ENVELOPE["altitude_error_m"]),
        ("airspeed envelope", not ENVELOPE["airspeed_min"] < r["Va"].min() < r["Va"].max() < ENVELOPE["airspeed_max"]),
    ) if bad]
    return {
        "log": r, "switches": switches, "success": not failures, "failures": failures, "duration_s": float(t[-1]),
        "xtrack_rms_m": float(np.sqrt(np.mean(r["xtrack"][steady] ** 2))),
        "xtrack_max_m": float(np.abs(r["xtrack"][steady]).max()),
        "altitude_rms_m": float(np.sqrt(np.mean(h_err[steady] ** 2))),
        "altitude_max_m": float(np.abs(h_err).max()),
        "airspeed_rms_ms": float(np.sqrt(np.mean(va_err ** 2))),
        "max_roll_deg": float(roll.max()),
        "surface_activity_deg": float(np.degrees(r["cmd"][:, :3].std(axis=0).mean())),
    }


def _mc_run(args) -> dict:
    seed, name = args
    s = setup()
    rng = np.random.default_rng(seed)
    plant = perturbed(s.ac, rng)
    speed, heading = rng.uniform(0, 8), rng.uniform(-np.pi, np.pi)
    wind = (speed * np.cos(heading), speed * np.sin(heading), 0.0)
    out = mission(s.controllers()[name], s, plant, rng, wind)
    out.pop("log")
    out.pop("switches")
    return {"seed": seed, "controller": name, "wind_ms": speed, **out}


def monte_carlo(runs: int, workers: int | None = None) -> list[dict]:
    """Same random airframes, winds, gusts and navigation errors for both autopilots."""
    jobs = [(seed, name) for seed in range(runs) for name in ("PID", "LQR")]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_mc_run, jobs))

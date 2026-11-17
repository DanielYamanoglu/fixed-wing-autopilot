"""Loop-at-a-time stability margins at the actuator inputs of the linearised closed loop.

The loop is broken at one actuator command while the other three stay closed,
so the result includes the servo lags and every cross-coupling of the 6 DoF model.
"""

import numpy as np

from .dynamics import NX, airdata, derivatives
from .params import Actuators, Aircraft
from .trim import Trim, jacobians

CHANNELS = ("elevator", "aileron", "rudder", "throttle")


def broken_loop(controller, ac: Aircraft, act: Actuators, tr: Trim, channel: int):
    """State space (A, B, C, D) from the injected command to the controller's command on that channel."""
    ref = np.array([-tr.x[2], 0.0, tr.Va])
    nz = controller.nz

    def split(X):
        return X[:NX], X[NX:]

    def closed(X, v):
        x, z = split(X)
        Va = airdata(x)[0]
        cmd = controller.output(x, Va, z, ref).copy()
        cmd[channel] = v[0]
        return np.concatenate([derivatives(x, cmd, ac, act), controller.zdot(x, Va, z, ref)])

    def measured(X, v):
        x, z = split(X)
        return np.array([controller.output(x, airdata(x)[0], z, ref)[channel]])

    X0 = np.concatenate([tr.x, controller.z0()])
    v0 = tr.cmd[channel:channel + 1]
    A, B = jacobians(closed, X0, v0)
    C, D = jacobians(measured, X0, v0)
    assert A.shape == (NX + nz, NX + nz)
    return A, B, C, D


def loop_gain(A, B, C, D, w: np.ndarray) -> np.ndarray:
    """L(jw) = -(C (jwI - A)^-1 B + D): negative feedback convention, closing v = u."""
    n = A.shape[0]
    return np.array([-(C @ np.linalg.solve(1j * wi * np.eye(n) - A, B) + D)[0, 0] for wi in w])


def margins(L: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Smallest gain margin (dB), phase margin (deg) and delay margin (ms) over all crossovers,
    plus the vector margin min|1 + L|, which bounds simultaneous gain and phase variations."""
    mag, ph = np.abs(L), np.unwrap(np.angle(L))
    gm, pm, dm, wc = np.inf, np.inf, np.inf, np.nan
    for i in range(len(w) - 1):
        # phase crossover: phase passes an odd multiple of -180 deg
        k0, k1 = np.floor((ph[i] + np.pi) / (2 * np.pi)), np.floor((ph[i + 1] + np.pi) / (2 * np.pi))
        if k0 != k1:
            m = 0.5 * (mag[i] + mag[i + 1])
            if m < 1:
                gm = min(gm, -20 * np.log10(m))
        if (mag[i] - 1) * (mag[i + 1] - 1) <= 0 and mag[i] != mag[i + 1]:
            f = (1 - mag[i]) / (mag[i + 1] - mag[i])
            p = ph[i] + f * (ph[i + 1] - ph[i])
            wx = w[i] + f * (w[i + 1] - w[i])
            phase = np.degrees((p + np.pi) % (2 * np.pi)) - 180  # wrapped to [-180, 180)
            margin = 180 - abs(phase)
            if margin < pm:
                pm, wc = margin, wx
            # a pure delay only lowers the phase, so it must travel down to -180 deg
            dm = min(dm, np.radians(phase + 180) / wx * 1e3)
    vm = float(np.min(np.abs(1 + L)))
    return {"gain_margin_db": gm, "phase_margin_deg": pm, "delay_margin_ms": dm, "crossover_rad_s": wc,
            "vector_margin": vm, "disk_phase_margin_deg": float(np.degrees(2 * np.arcsin(min(vm, 2.0) / 2)))}


def all_margins(controller, ac: Aircraft, act: Actuators, tr: Trim, w: np.ndarray | None = None) -> dict:
    w = np.logspace(-3, 3, 6000) if w is None else w
    out = {}
    for ch, name in enumerate(CHANNELS):
        A, B, C, D = broken_loop(controller, ac, act, tr, ch)
        out[name] = margins(loop_gain(A, B, C, D, w), w)
    return out

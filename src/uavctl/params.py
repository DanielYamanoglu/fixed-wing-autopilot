"""Aerosonde class small fixed-wing UAV (Beard & McLain, Small Unmanned Aircraft, 2012)."""

from dataclasses import dataclass, fields, replace

import numpy as np

G = 9.81
DEG = np.pi / 180


@dataclass(frozen=True)
class Aircraft:
    mass: float = 11.0
    Jx: float = 0.824
    Jy: float = 1.135
    Jz: float = 1.759
    Jxz: float = 0.120
    S: float = 0.55
    b: float = 2.8956
    c: float = 0.18994
    rho: float = 1.2682
    e: float = 0.9
    # longitudinal
    C_L_0: float = 0.23
    C_L_alpha: float = 5.61
    C_L_q: float = 7.95
    C_L_de: float = 0.13
    C_D_p: float = 0.03
    C_D_q: float = 0.0
    C_D_de: float = 0.0135
    C_m_0: float = 0.0135
    C_m_alpha: float = -2.74
    C_m_q: float = -38.21
    C_m_de: float = -0.99
    M: float = 50.0
    alpha0: float = 0.47
    # lateral
    C_Y_beta: float = -0.98
    C_Y_p: float = 0.0
    C_Y_r: float = 0.0
    C_Y_da: float = 0.075
    C_Y_dr: float = 0.19
    C_l_beta: float = -0.13
    C_l_p: float = -0.51
    C_l_r: float = 0.25
    C_l_da: float = 0.17
    C_l_dr: float = 0.0024
    C_n_beta: float = 0.073
    C_n_p: float = -0.069
    C_n_r: float = -0.095
    C_n_da: float = -0.011
    C_n_dr: float = -0.069
    # propulsion
    S_prop: float = 0.2027
    k_motor: float = 80.0
    C_prop: float = 1.0

    @property
    def AR(self) -> float:
        return self.b**2 / self.S

    @property
    def gammas(self) -> tuple[float, ...]:
        Jx, Jy, Jz, Jxz = self.Jx, self.Jy, self.Jz, self.Jxz
        g = Jx * Jz - Jxz**2
        return (
            Jxz * (Jx - Jy + Jz) / g,
            (Jz * (Jz - Jy) + Jxz**2) / g,
            Jz / g,
            Jxz / g,
            (Jz - Jx) / Jy,
            Jxz / Jy,
            ((Jx - Jy) * Jx + Jxz**2) / g,
            Jx / g,
        )


# 1-sigma relative uncertainty used in the Monte Carlo robustness study.
UNCERTAINTY = {
    "mass": 0.10, "Jx": 0.15, "Jy": 0.15, "Jz": 0.15,
    "C_D_p": 0.20, "C_L_alpha": 0.10, "C_m_alpha": 0.20, "C_m_q": 0.20, "C_m_de": 0.15,
    "C_l_p": 0.20, "C_l_da": 0.15, "C_n_beta": 0.20, "C_n_r": 0.20, "C_n_dr": 0.15,
    "k_motor": 0.05,
}


def perturbed(ac: Aircraft, rng: np.random.Generator, scale: float = 1.0) -> Aircraft:
    """Aircraft with every uncertain parameter scaled by an independent Gaussian factor."""
    names = {f.name for f in fields(ac)}
    changes = {k: getattr(ac, k) * (1 + scale * s * rng.standard_normal()) for k, s in UNCERTAINTY.items() if k in names}
    return replace(ac, **changes)


@dataclass(frozen=True)
class Actuators:
    tau_surface: float = 0.05  # s, servo first-order lag
    tau_throttle: float = 0.20  # s, motor and propeller spool-up
    surface_limit: float = 30 * DEG

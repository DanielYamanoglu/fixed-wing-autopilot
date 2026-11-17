"""Run every analysis, write reports/results.json and the README figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dynamics import PD, PHI
from .evaluate import MISSION, STEPS, mission, monte_carlo, setup, step_response
from .margins import CHANNELS, broken_loop, loop_gain, margins
from .trim import modes

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
COLORS = {"PID": "#2563eb", "LQR": "#ea580c"}
MISSION_WIND = (4.0, 3.0, 0.0)  # 5 m/s from the south-west


def _round(d):
    if isinstance(d, dict):
        return {k: _round(v) for k, v in d.items()}
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else round(float(d), 3)
    return d


def run(runs: int = 100, figures: bool = True) -> dict:
    s = setup()
    ctrls = s.controllers()
    w = np.logspace(-3, 3, 6000)
    results = {"trim": {"airspeed_ms": s.tr.Va, "alpha_deg": np.degrees(s.tr.alpha),
                        "elevator_deg": np.degrees(s.tr.cmd[0]), "throttle": s.tr.cmd[3], "residual": s.tr.residual},
               "open_loop_modes": modes(s.A)}

    loops, steps_out, missions = {}, {}, {}
    for name, c in ctrls.items():
        loops[name] = {ch: loop_gain(*broken_loop(c, s.ac, s.act, s.tr, i), w) for i, ch in enumerate(CHANNELS)}
        steps_out[name] = {k: step_response(c, s, k) for k in STEPS}
        missions[name] = mission(c, s, s.ac, np.random.default_rng(1), MISSION_WIND)
    results["margins"] = {n: {ch: margins(L, w) for ch, L in lp.items()} for n, lp in loops.items()}
    results["steps"] = {n: {k: {m: v for m, v in r.items() if not isinstance(v, np.ndarray)} for k, r in st.items()}
                        for n, st in steps_out.items()}
    results["mission"] = {n: {k: v for k, v in m.items() if k not in ("log", "switches")} for n, m in missions.items()}

    mc = monte_carlo(runs)
    summary = {}
    for name in ctrls:
        rows = [r for r in mc if r["controller"] == name]
        def col(k):
            return np.array([r[k] for r in rows])
        summary[name] = {
            "runs": len(rows), "success_rate": float(col("success").mean()),
            "xtrack_rms_mean_m": col("xtrack_rms_m").mean(), "xtrack_rms_p95_m": np.percentile(col("xtrack_rms_m"), 95),
            "altitude_rms_mean_m": col("altitude_rms_m").mean(), "altitude_max_p95_m": np.percentile(col("altitude_max_m"), 95),
            "airspeed_rms_mean_ms": col("airspeed_rms_ms").mean(), "max_roll_p95_deg": np.percentile(col("max_roll_deg"), 95),
            "surface_activity_mean_deg": col("surface_activity_deg").mean(),
            "failures": sorted({f"seed {r['seed']}: {', '.join(r['failures'])}" for r in rows if r["failures"]}),
        }
    results["monte_carlo"] = summary
    results = _round(results)

    out = ROOT / "reports" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if figures:
        FIG.mkdir(parents=True, exist_ok=True)
        _steps_figure(steps_out)
        _mission_figure(missions)
        _bode_figure(loops, w)
        _mc_figure(mc)
    return results


def _steps_figure(steps_out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    units = {"altitude +20 m": "Δ altitude [m]", "course +90°": "course [deg]", "airspeed +5 m/s": "Δ airspeed [m/s]"}
    for ax, (k, (_, size)) in zip(axes, STEPS.items()):
        ax.axhspan(0.95 * size, 1.05 * size, color="0.9", lw=0)
        ax.axhline(size, color="0.4", lw=0.8, ls="--")
        for name, st in steps_out.items():
            r = st[k]
            ax.plot(r["t"], r["y"], color=COLORS[name], lw=1.6,
                    label=f"{name}: {r['overshoot_pct']:.0f}% overshoot, {r['settling_s']:.1f} s to 5%")
        ax.set(title=k, xlabel="time after step [s]", ylabel=units[k], xlim=(-1, 25))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "steps.png", dpi=130)
    plt.close(fig)


def _mission_figure(missions):
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(MISSION[:, 1], MISSION[:, 0], "k--", lw=0.8, label="waypoint path")
    ax.plot(MISSION[:, 1], MISSION[:, 0], "ks", ms=5)
    for name, m in missions.items():
        x = m["log"]["x"]
        ax.plot(x[:, 1], x[:, 0], color=COLORS[name], lw=1.4, label=name)
    ax.annotate("", xy=(620, 800), xytext=(380, 480), arrowprops=dict(arrowstyle="->", lw=1.5, color="0.3"))
    ax.text(250, 330, "wind 5 m/s\n+ moderate\nturbulence", fontsize=8, color="0.3")
    ax.set(xlabel="east [m]", ylabel="north [m]", title="Ground track")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.7))
    ax.grid(alpha=0.3)
    ah, ar = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for name, m in missions.items():
        r = m["log"]
        ah.plot(r["t"], -r["x"][:, PD] - r["ref"][:, 0], color=COLORS[name], lw=1, label=name)
        ar.plot(r["t"], np.degrees(r["x"][:, PHI]), color=COLORS[name], lw=1, label=name)
    for a in (ah, ar):
        for ts in missions["PID"]["switches"]:
            a.axvline(ts, color="0.6", lw=0.7, ls=":")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")
    ah.set(ylabel="altitude error [m]", title="Tracking along the mission (dotted: waypoint switches)")
    ar.set(ylabel="roll [deg]", xlabel="time [s]")
    fig.tight_layout()
    fig.savefig(FIG / "mission.png", dpi=130)
    plt.close(fig)


def _bode_figure(loops, w):
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.4), sharex=True)
    for j, ch in enumerate(CHANNELS):
        for name, lp in loops.items():
            L = lp[ch]
            m = margins(L, w)
            axes[0, j].semilogx(w, 20 * np.log10(np.abs(L)), color=COLORS[name], lw=1.3,
                                label=f"{name}: PM {m['phase_margin_deg']:.0f}°, |1+L|min {m['vector_margin']:.2f}")
            phase = np.degrees(np.angle(L))
            phase[1:][np.abs(np.diff(phase)) > 180] = np.nan  # break the line at the +-180 deg wrap
            axes[1, j].semilogx(w, phase, color=COLORS[name], lw=1.3)
        axes[0, j].axhline(0, color="0.4", lw=0.8)
        axes[1, j].axhline(-180, color="0.4", lw=0.8)
        axes[1, j].axhline(180, color="0.4", lw=0.8)
        axes[0, j].set(title=f"loop broken at {ch}", ylim=(-60, 60))
        axes[0, j].legend(fontsize=7.5, loc="lower left")
        axes[1, j].set(xlabel="frequency [rad/s]", xlim=(1e-2, 1e2))
        for a in axes[:, j]:
            a.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("|L| [dB]")
    axes[1, 0].set_ylabel("phase L [deg]")
    fig.tight_layout()
    fig.savefig(FIG / "margins.png", dpi=130)
    plt.close(fig)


def _mc_figure(mc):
    keys = [("xtrack_rms_m", "cross-track RMS [m]"), ("altitude_max_m", "max altitude error [m]"),
            ("airspeed_rms_ms", "airspeed RMS error [m/s]"), ("max_roll_deg", "max roll [deg]")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (k, label) in zip(axes, keys):
        data = [[r[k] for r in mc if r["controller"] == n] for n in COLORS]
        bp = ax.boxplot(data, tick_labels=list(COLORS), patch_artist=True, widths=0.5)
        for patch, n in zip(bp["boxes"], COLORS):
            patch.set_facecolor(COLORS[n])
            patch.set_alpha(0.35)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{len(mc) // 2} perturbed airframes, random wind up to 8 m/s, moderate turbulence, navigation errors", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "monte_carlo.png", dpi=130)
    plt.close(fig)

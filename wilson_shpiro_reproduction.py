"""Reproduce the Wilson-model results analyzed by Shpiro et al. (2007).

Paper:
    Shpiro, Curtu, Rinzel & Rubin (2007), Journal of Neurophysiology 97:462-473.
    DOI: 10.1152/jn.00604.2006

This script recreates the Wilson-model results most useful for the binocular-rivalry
project:
  1. The five deterministic dynamical regimes shown in their Fig. 2.
  2. The rivalry period vs. common input for g = 0.44 (their Fig. 3A idea).
  3. The effect of inhibition strength g = 0.44, 0.42, 0.34 (their Fig. 3C idea).
  4. An optional stochastic extension using their exponentially filtered input-noise
     equation (Eq. 5), useful for testing whether noise destroys the WTA regime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


@dataclass(frozen=True)
class WilsonParams:
    tau_e: float = 20.0
    tau_h: float = 900.0
    tau_i: float = 11.0
    h: float = 0.47
    g: float = 0.44


DEFAULT_Y0 = np.array([1.0, 0.0, 0.0, 1.001, 0.0, 0.0], dtype=float)


def naka_rushton(x: np.ndarray | float, H: np.ndarray | float) -> np.ndarray | float:
    """Wilson gain function: 100*x_+^2 / ((10+H)^2 + x_+^2)."""
    xp = np.maximum(x, 0.0)
    return 100.0 * xp**2 / ((10.0 + H) ** 2 + xp**2)


def wilson_rhs(t: float, y: np.ndarray, V1: float, V2: float, p: WilsonParams) -> np.ndarray:
    """Six ODEs in Eq. 1 of Shpiro et al. for Wilson's model."""
    E1, H1, I1, E2, H2, I2 = y

    F1 = naka_rushton(V1 - p.g * I2, H1)
    F2 = naka_rushton(V2 - p.g * I1, H2)

    return np.array(
        [
            (-E1 + F1) / p.tau_e,
            (-H1 + p.h * E1) / p.tau_h,
            (-I1 + E1) / p.tau_i,
            (-E2 + F2) / p.tau_e,
            (-H2 + p.h * E2) / p.tau_h,
            (-I2 + E2) / p.tau_i,
        ]
    )


def simulate_deterministic(
    V: float,
    g: float = 0.44,
    t_end: float = 12000.0,
    dt_out: float = 2.0,
    y0: np.ndarray = DEFAULT_Y0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one symmetric-input deterministic condition."""
    p = WilsonParams(g=g)
    t_eval = np.arange(0.0, t_end + dt_out, dt_out)
    sol = solve_ivp(
        lambda t, y: wilson_rhs(t, y, V, V, p),
        (0.0, t_end),
        np.asarray(y0, dtype=float),
        t_eval=t_eval,
        max_step=2.0,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y


def _rhs_vectorized(y: np.ndarray, V: np.ndarray, g: float) -> np.ndarray:
    """Vectorized Wilson RHS for many input strengths at once."""
    E1, H1, I1, E2, H2, I2 = [y[:, k] for k in range(6)]
    x1 = np.maximum(V - g * I2, 0.0)
    x2 = np.maximum(V - g * I1, 0.0)
    F1 = 100.0 * x1**2 / ((10.0 + H1) ** 2 + x1**2)
    F2 = 100.0 * x2**2 / ((10.0 + H2) ** 2 + x2**2)

    return np.column_stack(
        [
            (-E1 + F1) / 20.0,
            (-H1 + 0.47 * E1) / 900.0,
            (-I1 + E1) / 11.0,
            (-E2 + F2) / 20.0,
            (-H2 + 0.47 * E2) / 900.0,
            (-I2 + E2) / 11.0,
        ]
    )


def simulate_input_sweep(
    V_values: np.ndarray,
    g: float,
    t_end: float = 60000.0,
    transient: float = 25000.0,
    dt: float = 2.0,
    sample_dt: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns sampled times and E1(t) after the discarded transient. 
    """
    V_values = np.asarray(V_values, dtype=float)
    y = np.zeros((len(V_values), 6), dtype=float)
    y[:, 0] = 1.0
    y[:, 3] = 1.001  # tiny asymmetry breaks the exactly symmetric invariant state

    n_steps = int(round(t_end / dt))
    save_every = max(1, int(round(sample_dt / dt)))
    saved_t: list[float] = []
    saved_e1: list[np.ndarray] = []

    for step in range(n_steps):
        k1 = _rhs_vectorized(y, V_values, g)
        k2 = _rhs_vectorized(y + 0.5 * dt * k1, V_values, g)
        k3 = _rhs_vectorized(y + 0.5 * dt * k2, V_values, g)
        k4 = _rhs_vectorized(y + dt * k3, V_values, g)
        y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        t = (step + 1) * dt
        if t >= transient and step % save_every == 0:
            saved_t.append(t)
            saved_e1.append(y[:, 0].copy())

    return np.asarray(saved_t), np.asarray(saved_e1)


def estimate_periods(
    V_values: np.ndarray,
    g: float,
    t_end: float = 60000.0,
    transient: float = 25000.0,
) -> np.ndarray:
    """Estimate sustained oscillation periods from late E1 peaks.

    NaN means the late-time trajectory is not classified as a sustained rivalry
    oscillation (fusion, WTA, high-activity simultaneous state, or unresolved near a
    bifurcation).
    """
    sample_dt = 10.0
    t, E1 = simulate_input_sweep(
        V_values,
        g=g,
        t_end=t_end,
        transient=transient,
        dt=2.0,
        sample_dt=sample_dt,
    )

    periods = np.full(len(V_values), np.nan)
    late_mask = t >= max(transient, t_end - 25000.0)
    t_late = t[late_mask]

    for j in range(len(V_values)):
        x = E1[late_mask, j]
        amplitude = np.ptp(x)
        if amplitude < 0.20:
            continue

        prominence = max(0.05, 0.10 * amplitude)
        peaks, _ = find_peaks(
            x,
            prominence=prominence,
            distance=max(1, int(round(300.0 / sample_dt))),
        )
        if len(peaks) < 3:
            continue

        intervals_s = np.diff(t_late[peaks]) / 1000.0
        periods[j] = float(np.median(intervals_s[-4:]))

    return periods


def simulate_stochastic(
    V: float,
    g: float = 0.44,
    sigma: float = 0.03,
    tau_n: float = 10.0,
    t_end: float = 120000.0,
    dt: float = 1.0,
    sample_dt: float = 5.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eq. 5:
        dn_i/dt = -n_i/tau_n + sigma*sqrt(2/tau_n)*eta_i(t)

   
    """
    rng = np.random.default_rng(seed)
    p = WilsonParams(g=g)
    y = DEFAULT_Y0.copy()
    n = np.zeros(2, dtype=float)

    n_steps = int(round(t_end / dt))
    save_every = max(1, int(round(sample_dt / dt)))
    t_save: list[float] = []
    y_save: list[np.ndarray] = []
    n_save: list[np.ndarray] = []

    diffusion = sigma * np.sqrt(2.0 / tau_n)

    for step in range(n_steps):
        n += (-n / tau_n) * dt + diffusion * np.sqrt(dt) * rng.normal(size=2)

        E1, H1, I1, E2, H2, I2 = y
        F1 = naka_rushton(V + n[0] - p.g * I2, H1)
        F2 = naka_rushton(V + n[1] - p.g * I1, H2)
        dy = np.array(
            [
                (-E1 + F1) / p.tau_e,
                (-H1 + p.h * E1) / p.tau_h,
                (-I1 + E1) / p.tau_i,
                (-E2 + F2) / p.tau_e,
                (-H2 + p.h * E2) / p.tau_h,
                (-I2 + E2) / p.tau_i,
            ]
        )
        y += dt * dy

        if step % save_every == 0:
            t_save.append(step * dt)
            y_save.append(y.copy())
            n_save.append(n.copy())

    return np.asarray(t_save), np.asarray(y_save), np.asarray(n_save)


def save_regime_figures(out_dir: Path) -> None:
    """Recreate the five Wilson regimes from Shpiro et al. Fig. 2."""
    regimes = [
        ("I", 38.0, "simultaneous high activity"),
        ("II", 15.0, "rivalry, decreasing-duration branch"),
        ("III", 6.0, "winner-take-all"),
        ("IV", 3.5, "rivalry, increasing-duration branch"),
        ("V", 2.0, "fusion / simultaneous low activity"),
    ]

    rows = []
    for roman, V, description in regimes:
        # Longer integration for regime IV because its period is about 4.7 s.
        t_end = 16000.0 if V == 3.5 else 12000.0
        t, y = simulate_deterministic(V=V, g=0.44, t_end=t_end)
        E1, E2 = y[0], y[3]

        fig = plt.figure(figsize=(7.2, 4.6))
        ax = fig.add_axes([0.12, 0.14, 0.83, 0.77])
        ax.plot(t / 1000.0, E1, label=r"$E_1$")
        ax.plot(t / 1000.0, E2, label=r"$E_2$")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("population activity")
        ax.set_title(f"Wilson regime {roman}: V1 = V2 = {V:g} ({description})")
        ax.legend(frameon=False)
        fig.savefig(out_dir / f"figure2_regime_{roman}.png", dpi=220)
        fig.savefig(out_dir / f"figure2_regime_{roman}.pdf")
        plt.close(fig)

        rows.append(
            pd.DataFrame(
                {
                    "time_s": t / 1000.0,
                    "E1": E1,
                    "E2": E2,
                    "V": V,
                    "regime": roman,
                }
            )
        )

    pd.concat(rows, ignore_index=True).to_csv(out_dir / "figure2_timecourses.csv", index=False)


def save_period_figures(out_dir: Path) -> pd.DataFrame:
    """Recreate period-vs-input results analogous to Fig. 3A and Fig. 3C."""
    V_values = np.arange(1.5, 38.01, 0.5)
    all_rows = []
    period_by_g: dict[float, np.ndarray] = {}

    for g in (0.44, 0.42, 0.34):
        periods = estimate_periods(V_values, g=g)
        period_by_g[g] = periods
        all_rows.append(pd.DataFrame({"V": V_values, "g": g, "period_s": periods}))

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out_dir / "figure3_period_sweep.csv", index=False)

    # Fig. 3A-like plot: g = 0.44 only.
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.12, 0.14, 0.83, 0.77])
    ax.plot(V_values, period_by_g[0.44], marker="o", markersize=3)
    ax.set_xlabel("common input V")
    ax.set_ylabel("rivalry period T (s)")
    ax.set_title("Wilson model: nonmonotonic rivalry period (g = 0.44)")
    ax.set_xlim(0, 40)
    fig.savefig(out_dir / "figure3A_period_g044.png", dpi=220)
    fig.savefig(out_dir / "figure3A_period_g044.pdf")
    plt.close(fig)

    # Fig. 3C-like plot: compare inhibition strengths.
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.12, 0.14, 0.83, 0.77])
    for g in (0.44, 0.42, 0.34):
        ax.plot(V_values, period_by_g[g], marker="o", markersize=2.5, label=f"g = {g:.2f}")
    ax.set_xlabel("common input V")
    ax.set_ylabel("rivalry period T (s)")
    ax.set_title("Wilson model: effect of cross-inhibition strength")
    ax.set_xlim(0, 40)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "figure3C_period_by_g.png", dpi=220)
    fig.savefig(out_dir / "figure3C_period_by_g.pdf")
    plt.close(fig)

    return df


def save_noise_figure(out_dir: Path) -> None:
    
    
    t, y, n = simulate_stochastic(V=4.5, sigma=0.03, tau_n=10.0, t_end=120000.0)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.12, 0.14, 0.83, 0.77])
    ax.plot(t / 1000.0, y[:, 0], label=r"$E_1$")
    ax.plot(t / 1000.0, y[:, 3], label=r"$E_2$")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("population activity")
    ax.set_title("Noisy Wilson model inside deterministic WTA region (V = 4.5)")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "noise_wta_example.png", dpi=220)
    fig.savefig(out_dir / "noise_wta_example.pdf")
    plt.close(fig)

    pd.DataFrame(
        {
            "time_s": t / 1000.0,
            "E1": y[:, 0],
            "E2": y[:, 3],
            "noise1": n[:, 0],
            "noise2": n[:, 1],
        }
    ).to_csv(out_dir / "noise_wta_example.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("wilson_results"))
    parser.add_argument(
        "--noise",
        action="store_true",
        help="also run the stochastic Wilson extension using Shpiro et al. Eq. 5",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    save_regime_figures(args.out)
    df = save_period_figures(args.out)
    if args.noise:
        save_noise_figure(args.out)

    print(f"Saved results to: {args.out.resolve()}")
    print("\nOscillatory V ranges detected numerically:")
    for g, group in df.groupby("g"):
        valid = group.dropna(subset=["period_s"])
        if len(valid):
            # Print contiguous-looking min/max only as a compact summary; g=.44 has
            # a NaN gap corresponding to the WTA region, visible in the CSV/figure.
            print(f"  g={g:.2f}: V from {valid.V.min():.2f} to {valid.V.max():.2f}")
        else:
            print(f"  g={g:.2f}: no sustained oscillations detected")


if __name__ == "__main__":
    main()

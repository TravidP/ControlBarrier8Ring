#!/usr/bin/env python3
"""
Compare distributed control (Barriercontrol) vs. Traffic light strategies
across vehicle densities N = 37, 47, 57.

Reads batch_summary CSVs from each (strategy, N) cell, computes per-cell
statistics (mean, std, median, 95% CI, n) for each KPI, runs Welch's t-test
between strategies at each N, and produces:

  - figs/summary_kpi_vs_N.png   : 2x2 panel, KPI vs N with 95% CI error bars
  - figs/boxplot_per_kpi.png    : 2x2 panel, boxplots per (strategy, N) cell
  - tables/results_table.csv    : per-cell stats + p-values + Cohen's d
  - tables/results_table.tex    : same, LaTeX-ready

Usage:
    python compare_strategies.py --base-dir /path/to/simulation_results
    # Or edit BASE_DIR below.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------- Configuration ----------

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR_DEFAULT = PROJECT_DIR / "simulation_results"
DENSITIES = [37, 47, 57]
STRATEGIES = {
    "Trafficlight":   "TrafficlightResults",
    "Barriercontrol": "BarrierControlResults",
}
STRATEGY_COLORS = {
    "Trafficlight":   "#08519C",  # blue
    "Barriercontrol": "#CB181D",  # red
}
STRATEGY_LABELS = {
    "Trafficlight":   "Traffic light",
    "Barriercontrol": "Distributed (barrier)",
}

# KPIs to analyze. (column_in_batch_summary, display_name, unit, lower_is_better)
# We also derive normalized KPIs below.
KPIS_RAW = [
    ("final_throughput",      "Throughput",            "veh/s",  False),
    ("total_delay_s",         "Total delay",           "s",      True),
    ("energy_kWh",            "Energy consumption",    "kWh",    True),
    ("safety_violation_rate", "Safety violation rate", "viol/s", True),
]
# Normalized KPIs (computed in load step). (key, display_name, unit, lower_is_better)
KPIS_NORM = [
    ("energy_per_km",         "Energy per distance",   "kWh/km",     True),
    ("delay_per_crossing",    "Delay per crossing",    "s/veh",      True),
    ("safety_per_veh_hour",   "Safety viol. per veh-h","viol/(veh·h)", True),
    ("throughput_per_vehicle","Throughput per vehicle","1/s",        False),
]
# Set which set to use for the main 2x2 summary figure
HEADLINE_KPIS = [
    ("final_throughput",      "Throughput",            "veh/s",  False),
    ("delay_per_crossing",    "Delay per crossing",    "s/veh",  True),
    ("energy_per_km",         "Energy per distance",   "kWh/km", True),
    ("safety_violation_rate", "Safety violation rate", "viol/s", True),
]


# ---------- Data loading ----------

BATCH_DIR_RE = re.compile(r"batch_\d{8}_\d{6}_N(?P<N>\d+)_T(?P<T>\d+)_R(?P<R>[\dp]+)")


@dataclass
class Cell:
    strategy: str
    N: int
    runs: pd.DataFrame   # rows = runs, columns = KPIs (raw + normalized)
    source_dir: Path

    @property
    def n_runs(self) -> int:
        return len(self.runs)


def find_batch_dirs(base_dir: Path, strategy_subdir: str, N: int) -> list[Path]:
    """Find all batch_* directories for a given strategy and N."""
    strategy_path = base_dir / strategy_subdir
    if not strategy_path.is_dir():
        return []
    matches = []
    for d in sorted(strategy_path.iterdir()):
        if not d.is_dir():
            continue
        m = BATCH_DIR_RE.match(d.name)
        if m and int(m.group("N")) == N:
            matches.append(d)
    return matches


def load_batch_summary(batch_dir: Path) -> pd.DataFrame | None:
    """Load the batch_summary CSV from a batch dir, or None if missing."""
    summaries = sorted(batch_dir.glob("batch_summary_*.csv"))
    if not summaries:
        return None
    # Use the most recent if multiple
    return pd.read_csv(summaries[-1])


def derive_normalized_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-vehicle / per-distance normalized KPIs to a batch summary."""
    df = df.copy()
    duration_h = df["duration_s"] / 3600.0
    N = df["vehicles"]

    df["energy_per_km"] = df["energy_kWh"] / df["total_distance_km"].replace(0, np.nan)
    df["delay_per_crossing"] = df["total_delay_s"] / df["total_crossings"].replace(0, np.nan)
    df["safety_per_veh_hour"] = df["total_safety_violations"] / (N * duration_h)
    df["throughput_per_vehicle"] = df["final_throughput"] / N
    return df


def load_cell(base_dir: Path, strategy: str, N: int) -> Cell | None:
    """Load all runs for one (strategy, N) cell."""
    subdir = STRATEGIES[strategy]
    batch_dirs = find_batch_dirs(base_dir, subdir, N)
    if not batch_dirs:
        print(f"  [{strategy}, N={N}] no batch dir found")
        return None

    # Combine all matching batch dirs (usually one per cell, but support multiple)
    frames = []
    used_dir = None
    for bd in batch_dirs:
        summary = load_batch_summary(bd)
        if summary is None or summary.empty:
            continue
        frames.append(summary)
        used_dir = bd

    if not frames:
        print(f"  [{strategy}, N={N}] batch dirs found but no summary CSVs")
        return None

    runs = pd.concat(frames, ignore_index=True)
    runs = derive_normalized_kpis(runs)
    print(f"  [{strategy}, N={N}] loaded {len(runs)} runs from {used_dir.name}")
    return Cell(strategy=strategy, N=N, runs=runs, source_dir=used_dir)


def load_all_cells(base_dir: Path) -> dict[tuple[str, int], Cell]:
    print(f"Loading from {base_dir}")
    cells = {}
    for strategy in STRATEGIES:
        for N in DENSITIES:
            cell = load_cell(base_dir, strategy, N)
            if cell is not None:
                cells[(strategy, N)] = cell
    return cells


# ---------- Statistics ----------

def ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, half-width of 95% CI, std). Uses t-distribution."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = values.mean()
    if n == 1:
        return (mean, np.nan, 0.0)
    std = values.std(ddof=1)
    sem = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (mean, t_crit * sem, std)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def compare_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch's t-test + Mann-Whitney U + Cohen's d between two samples."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    out = {"n_a": len(a), "n_b": len(b)}
    if len(a) < 2 or len(b) < 2:
        out.update({"t_p": np.nan, "mw_p": np.nan, "cohen_d": np.nan})
        return out
    try:
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
    except Exception:
        t_stat, t_p = np.nan, np.nan
    try:
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        u_stat, u_p = np.nan, np.nan
    out.update({"t_p": t_p, "mw_p": u_p, "cohen_d": cohens_d(a, b)})
    return out


# ---------- Results table ----------

def build_results_table(cells: dict[tuple[str, int], Cell],
                        kpis: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    rows = []
    for (strategy, N), cell in sorted(cells.items(), key=lambda x: (x[0][1], x[0][0])):
        for col, name, unit, _lower_better in kpis:
            if col not in cell.runs.columns:
                continue
            vals = cell.runs[col].to_numpy()
            mean, ci_hw, std = ci95(vals)
            rows.append({
                "N": N,
                "strategy": strategy,
                "kpi": name,
                "unit": unit,
                "n": int(np.sum(~np.isnan(vals))),
                "mean": mean,
                "std": std,
                "ci95_halfwidth": ci_hw,
                "median": float(np.nanmedian(vals)) if len(vals) else np.nan,
                "min": float(np.nanmin(vals)) if len(vals) else np.nan,
                "max": float(np.nanmax(vals)) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def build_comparison_table(cells: dict[tuple[str, int], Cell],
                           kpis: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    """For each (N, KPI), compare Trafficlight vs Barriercontrol."""
    rows = []
    for N in DENSITIES:
        cell_tl = cells.get(("Trafficlight", N))
        cell_bc = cells.get(("Barriercontrol", N))
        if cell_tl is None or cell_bc is None:
            continue
        for col, name, unit, lower_better in kpis:
            if col not in cell_tl.runs.columns or col not in cell_bc.runs.columns:
                continue
            a = cell_tl.runs[col].to_numpy()
            b = cell_bc.runs[col].to_numpy()
            mean_a = np.nanmean(a)
            mean_b = np.nanmean(b)
            cmp = compare_pair(a, b)
            # Percent change: Barriercontrol relative to Trafficlight
            pct_change = (mean_b - mean_a) / mean_a * 100.0 if mean_a not in (0, np.nan) else np.nan
            # Direction: does Barrier beat Trafficlight?
            if lower_better:
                winner = "Barrier" if mean_b < mean_a else "Trafficlight"
            else:
                winner = "Barrier" if mean_b > mean_a else "Trafficlight"
            rows.append({
                "N": N,
                "kpi": name,
                "unit": unit,
                "mean_TL": mean_a,
                "mean_BC": mean_b,
                "pct_change_BC_vs_TL": pct_change,
                "cohen_d": cmp["cohen_d"],
                "t_p": cmp["t_p"],
                "mw_p": cmp["mw_p"],
                "n_TL": cmp["n_a"],
                "n_BC": cmp["n_b"],
                "winner": winner,
            })
    return pd.DataFrame(rows)


def df_to_latex_safe(df: pd.DataFrame, float_format: str = "%.4g") -> str:
    """Produce a LaTeX table string without requiring jinja2."""
    return df.to_latex(index=False, float_format=lambda x: float_format % x if pd.notnull(x) else "")


# ---------- Plotting ----------

def plot_summary_panel(cells: dict[tuple[str, int], Cell],
                       kpis: list[tuple[str, str, str, bool]],
                       out_path: Path) -> None:
    """2x2 grid: KPI vs N with 95% CI error bars, one line per strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (col, name, unit, lower_better) in zip(axes, kpis):
        for strategy in STRATEGIES:
            xs, means, errs = [], [], []
            for N in DENSITIES:
                cell = cells.get((strategy, N))
                if cell is None or col not in cell.runs.columns:
                    continue
                vals = cell.runs[col].to_numpy()
                mean, ci_hw, _std = ci95(vals)
                if np.isnan(mean):
                    continue
                xs.append(N)
                means.append(mean)
                errs.append(ci_hw if not np.isnan(ci_hw) else 0.0)
            if not xs:
                continue
            ax.errorbar(
                xs, means, yerr=errs,
                marker="o", markersize=7, linewidth=2.0, capsize=5,
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
            )

        better_str = " (lower is better)" if lower_better else " (higher is better)"
        ax.set_title(f"{name}{better_str}")
        ax.set_xlabel("Number of vehicles N")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.set_xticks(DENSITIES)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("KPI comparison across vehicle densities (mean ± 95% CI, n=10 runs per cell)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_boxplots(cells: dict[tuple[str, int], Cell],
                  kpis: list[tuple[str, str, str, bool]],
                  out_path: Path) -> None:
    """2x2 grid of boxplots: for each KPI, 6 boxes (3 N x 2 strategies)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (col, name, unit, lower_better) in zip(axes, kpis):
        positions = []
        data = []
        colors = []
        tick_positions = []
        tick_labels = []
        pos = 0
        width = 0.8
        gap_within = 0.0
        gap_between = 1.2

        for N in DENSITIES:
            cell_positions = []
            for strategy in STRATEGIES:
                cell = cells.get((strategy, N))
                if cell is None or col not in cell.runs.columns:
                    pos += 1
                    continue
                vals = cell.runs[col].dropna().to_numpy()
                if len(vals) == 0:
                    pos += 1
                    continue
                positions.append(pos)
                cell_positions.append(pos)
                data.append(vals)
                colors.append(STRATEGY_COLORS[strategy])
                pos += 1
            if cell_positions:
                tick_positions.append(np.mean(cell_positions))
                tick_labels.append(f"N={N}")
            pos += gap_between

        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue

        bp = ax.boxplot(data, positions=positions, widths=width,
                        patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor(c)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.4)

        # Overlay raw points
        for vals, p, c in zip(data, positions, colors):
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.2
            ax.scatter(np.full_like(vals, p, dtype=float) + jitter, vals,
                       color=c, edgecolor="black", linewidth=0.4, s=22, alpha=0.85, zorder=3)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(f"{name} [{unit}]")
        better_str = " (lower is better)" if lower_better else " (higher is better)"
        ax.set_title(f"{name}{better_str}")
        ax.grid(True, axis="y", alpha=0.3)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=STRATEGY_COLORS[s], alpha=0.55, label=STRATEGY_LABELS[s])
               for s in STRATEGIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.00),
               ncol=2, frameon=False)
    fig.suptitle("Per-run KPI distributions across strategy and density",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ---------- Pretty print to console ----------

def format_pvalue(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "<0.001 ***"
    if p < 0.01:
        return f"{p:.3f} **"
    if p < 0.05:
        return f"{p:.3f} *"
    return f"{p:.3f}"


def print_summary(cells: dict[tuple[str, int], Cell],
                  comparison: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("DATA COVERAGE")
    print("=" * 78)
    print(f"{'Strategy':<18} {'N':>4}  {'n_runs':>7}")
    print("-" * 32)
    for strategy in STRATEGIES:
        for N in DENSITIES:
            cell = cells.get((strategy, N))
            n = cell.n_runs if cell else 0
            print(f"{strategy:<18} {N:>4}  {n:>7}")

    if comparison.empty:
        print("\nNo comparable cells (need both strategies at the same N).")
        return

    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD: Barriercontrol vs Trafficlight (per N, per KPI)")
    print("=" * 78)
    for N in DENSITIES:
        sub = comparison[comparison["N"] == N]
        if sub.empty:
            continue
        print(f"\n--- N = {N} (n_TL={sub['n_TL'].iloc[0]}, n_BC={sub['n_BC'].iloc[0]}) ---")
        for _, r in sub.iterrows():
            arrow = "↓" if r["pct_change_BC_vs_TL"] < 0 else "↑"
            print(f"  {r['kpi']:<24} | TL={r['mean_TL']:.4g} {r['unit']:<10} "
                  f"BC={r['mean_BC']:.4g} {r['unit']:<10} | "
                  f"Δ={arrow}{abs(r['pct_change_BC_vs_TL']):.1f}% | "
                  f"d={r['cohen_d']:+.2f} | "
                  f"t-p={format_pvalue(r['t_p'])} | "
                  f"MW-p={format_pvalue(r['mw_p'])} | "
                  f"winner: {r['winner']}")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR_DEFAULT,
                    help="Parent directory containing TrafficlightResults/ and BarrierControlResults/")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (defaults to <base-dir>/AnalysisOutputs)")
    args = ap.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    out_dir = (args.out_dir or (base_dir / "AnalysisOutputs")).expanduser().resolve()
    figs_dir = out_dir / "figs"
    tables_dir = out_dir / "tables"

    cells = load_all_cells(base_dir)
    if not cells:
        print("ERROR: No data loaded. Check --base-dir.")
        return

    # Per-cell summary stats
    headline_table = build_results_table(cells, HEADLINE_KPIS)
    raw_table = build_results_table(cells, KPIS_RAW + KPIS_NORM)
    comparison = build_comparison_table(cells, HEADLINE_KPIS)

    tables_dir.mkdir(parents=True, exist_ok=True)
    headline_table.to_csv(tables_dir / "results_headline.csv", index=False)
    raw_table.to_csv(tables_dir / "results_all_kpis.csv", index=False)
    comparison.to_csv(tables_dir / "comparison_per_N.csv", index=False)
    (tables_dir / "comparison_per_N.tex").write_text(df_to_latex_safe(comparison))
    print(f"Saved tables to {tables_dir}")

    # Plots
    plot_summary_panel(cells, HEADLINE_KPIS, figs_dir / "summary_kpi_vs_N.png")
    plot_boxplots(cells, HEADLINE_KPIS, figs_dir / "boxplot_per_kpi.png")

    print_summary(cells, comparison)


if __name__ == "__main__":
    main()

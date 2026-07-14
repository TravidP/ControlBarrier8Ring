import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Style -----------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------- Constants -------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "simulation_results"
BC_RESULTS_DIR = BASE_DIR / "BarrierControlResults"

BATCHES: dict[str, Path] = {
    "Barriercontrol_N57": BC_RESULTS_DIR / "Batch_30runs_N=57",
    "Barriercontrol_N67": BC_RESULTS_DIR / "Batch_30runs_N=67",
    "Barriercontrol_N77": BC_RESULTS_DIR / "Batch_30runs_N=77",
}

OUTPUT_DIR = BASE_DIR / "PlotsControllers" / "Batch_BC_N57_N67_N77"

TIME_COL       = "time_seconds"
SPEED_COL      = "avg_speed_kmh"
DIST_COL       = "total_distance_km"
ENERGY_COL     = "Energy_kWh"
THROUGHPUT_COL = "throughput"
DELAY_COL      = "total_delay_s"
SAFETY_COL     = "safety_violation_rate"
BARRIER_FIRINGS_COL = "u_barrier_firings"
BARRIER_AVG_COL     = "u_barrier_avg"
HEADWAY_AVG_COL     = "u_headway_avg"
STEADY_STATE_COL    = "steady_state"

TARGET_SPEED_KMH  = 30.0
TARGET_SPEED_MPS  = TARGET_SPEED_KMH / 3.6
TRACK_LENGTH_M    = 2357.36
RAMP_UP_TIME_S    = 17.5  # seconds to exclude as ramp-up in speed plots
HEADWAY_EVAL_START_S = 120.0

LAP_DURATION_S    = 281.78
STABILITY_START_S = 500.0
STABILITY_END_S   = STABILITY_START_S + LAP_DURATION_S  # 781.78 s
HEADWAY_TARGET_BAND_M = 0.5

PREFERRED_CONFLICT_N = 57
PREFERRED_CONFLICT_RUN = 27
CONFLICT_WINDOW_AFTER_S = 30.0
BARRIER_EPS = 1e-9

GROUP_N_COLORS: dict[int, str] = {
    57: "#E53935",  # red
    67: "#43A047",  # green
    77: "#1E88E5",  # blue
}

GAP_N_COLORS: dict[int, str] = {
    57: "#E53935",  # red
    67: "#43A047",  # green
    77: "#1E88E5",  # blue
}

BARRIER_SIGN_COLORS = {
    "negative": "#B23A48",
    "positive": "#2A9D8F",
}

CTRL_DISPLAY: dict[str, str] = {
    "Trafficlight":      "Traffic Light",
    "Barriercontrol":    "Barrier Control",
    "Barriercontrol_N57": "Barrier Control",
    "Barriercontrol_N67": "Barrier Control",
    "Barriercontrol_N77": "Barrier Control",
}


def _darken(hex_color: str, factor: float = 0.70) -> str:
    """Return a darkened version of a hex color for the mean line."""
    rgb = matplotlib.colors.to_rgb(hex_color)
    return matplotlib.colors.to_hex(tuple(c * factor for c in rgb))


# ---------- Loading ---------------------------------------------------------

def _resolve_throughput_csv(row: pd.Series, batch_dir: Path) -> Path | None:
    candidate = Path(row["throughput_csv"])
    if candidate.is_file():
        return candidate
    run_dir = batch_dir / f"run_{int(row['run_index']):02d}"
    hits = sorted(run_dir.glob("throughput_*.csv"))
    return hits[0] if hits else None


def _resolve_vehicle_csv(row: pd.Series, batch_dir: Path) -> Path | None:
    if pd.isna(row.get("vehicle_csv")):
        return None
    candidate = Path(row["vehicle_csv"])
    if candidate.is_file():
        return candidate
    run_dir = batch_dir / f"run_{int(row['run_index']):02d}"
    hits = sorted(run_dir.glob("vehicle_*.csv"))
    return hits[0] if hits else None


def _resolve_lap_csv(row: pd.Series, batch_dir: Path) -> Path | None:
    if pd.isna(row.get("lap_csv")):
        return None
    candidate = Path(row["lap_csv"])
    if candidate.is_file():
        return candidate
    run_dir = batch_dir / f"run_{int(row['run_index']):02d}"
    hits = sorted(run_dir.glob("lap_times_*.csv"))
    return hits[0] if hits else None


def load_batch_runs(group_name: str, batch_dir: Path) -> list:
    summary_files = sorted(batch_dir.glob("batch_summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No batch summary found in {batch_dir}")

    frames: list[pd.DataFrame] = []
    for sf in summary_files:
        try:
            df = pd.read_csv(sf)
            if not df.empty:
                frames.append(df)
        except Exception:
            pass

    summary_df = pd.concat(frames, ignore_index=True).sort_values("run_index")
    summary_df = summary_df[
        summary_df["final_time_seconds"] >= summary_df["duration_s"] * 0.99
    ].reset_index(drop=True)

    datasets: list = []
    for _, row in summary_df.iterrows():
        n_vehicles = int(row["vehicles"])
        csv_path = _resolve_throughput_csv(row, batch_dir)
        if csv_path is None:
            continue

        try:
            df = pd.read_csv(csv_path).sort_values(TIME_COL)
        except Exception:
            continue

        available_cols = [
            c for c in [
                TIME_COL, DIST_COL, ENERGY_COL, SPEED_COL, THROUGHPUT_COL, DELAY_COL, SAFETY_COL,
                BARRIER_FIRINGS_COL, BARRIER_AVG_COL, HEADWAY_AVG_COL, STEADY_STATE_COL,
            ]
            if c in df.columns
        ]
        df = df[available_cols].dropna()

        run_label = f"Run {int(row['run_index'])}"
        vehicle_csv = _resolve_vehicle_csv(row, batch_dir)
        lap_csv = _resolve_lap_csv(row, batch_dir)
        datasets.append((df, run_label, group_name, n_vehicles, vehicle_csv, lap_csv))

    return datasets


# ---------- Helpers ---------------------------------------------------------

def _run_index(run_label: str) -> int:
    return int(run_label.split()[1])


def _group_datasets(datasets: list) -> dict[tuple[str, int], list[pd.DataFrame]]:
    groups: dict[tuple[str, int], list] = {}
    for df, _, group_name, vehicle_count, *_ in datasets:
        groups.setdefault((group_name, vehicle_count), []).append(df)
    return groups


def _sorted_keys(groups: dict) -> list:
    return sorted(groups.keys(), key=lambda k: (k[1], k[0]))


def align_run_counts(datasets: list) -> list:
    by_group: dict = {}
    for item in datasets:
        df, run_label, group_name, vehicle_count, *_ = item
        by_group.setdefault((group_name, vehicle_count), {})[_run_index(run_label)] = item

    n_values = {n for _, n in by_group}
    for n in n_values:
        keys = [(g, n) for (g, nv) in by_group if nv == n]
        if len(keys) < 2:
            continue
        common = set.intersection(*[set(by_group[k]) for k in keys])
        for k in keys:
            for r in sorted(set(by_group[k]) - common):
                del by_group[k][r]

    result: list = []
    for key in sorted(by_group, key=lambda k: (k[1], k[0])):
        for run_idx in sorted(by_group[key]):
            result.append(by_group[key][run_idx])
    return result


def save_and_show(fig: plt.Figure, filename: str) -> None:
    output_dir = OUTPUT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    print(f"Saved {output_dir / stem}")
    plt.close(fig)


def theoretical_max_throughput(vehicle_count: int) -> float:
    return 2.0 * vehicle_count * TARGET_SPEED_MPS / TRACK_LENGTH_M



def _append_band_legend_bottom(ax: plt.Axes, ncol: int = 3) -> None:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        loc="best",
        ncol=ncol,
        frameon=False,
    )


# ---------- Throughput combined plot ----------------------------------------

def plot_throughput_combined(ax: plt.Axes, datasets: list) -> None:
    groups = _group_datasets(datasets)

    for (group_name, vehicle_count) in _sorted_keys(groups):
        run_dfs = [df for df in groups[(group_name, vehicle_count)] if THROUGHPUT_COL in df.columns]
        if not run_dfs:
            continue

        color  = GROUP_N_COLORS.get(vehicle_count, "#ef3b2c")
        ctrl   = CTRL_DISPLAY[group_name]
        n_runs = len(run_dfs)
        label  = f"{ctrl}, N={vehicle_count} (n={n_runs})"

        time_ref = run_dfs[0][TIME_COL].to_numpy()
        metric_matrix = np.vstack([
            np.interp(time_ref, df[TIME_COL].to_numpy(), df[THROUGHPUT_COL].to_numpy())
            for df in run_dfs
        ])

        mean_vals = metric_matrix.mean(axis=0)
        std_vals  = metric_matrix.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean_vals)
        
        sem_vals  = std_vals / np.sqrt(n_runs)
        ci_bound  = 1.96 * sem_vals

        ax.fill_between(time_ref, mean_vals - ci_bound, mean_vals + ci_bound,
                        color=color, alpha=0.20, edgecolor="none")
        ax.plot(time_ref, mean_vals, color=_darken(color), linewidth=2.2, label=label)

    n_values = sorted({vc for _, vc in groups})
    linestyles = ["--", "-.", ":"]
    for i, vc in enumerate(n_values):
        max_tp = theoretical_max_throughput(vc)
        ax.axhline(
            max_tp,
            color="#555555",
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.5,
            alpha=0.75,
            label=f"Theo. max N={vc}: {max_tp:.3f} veh/s",
        )

    ax.set_xlabel("Time, t, [s]")
    ax.set_ylabel("Throughput, I, [veh/s]")
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="best", ncol=2, frameon=False)


# ---------- Generic time-metric plot ----------------------------------------

def plot_time_metric_with_band(
    ax: plt.Axes,
    datasets: list,
    metric_col: str,
    title: str,
    ylabel: str,
    scale: float = 1.0,
    ylim_bottom: float | None = None,
    ylim_top: float | None = None,
    hline: float | None = None,
    hline_label: str = "",
) -> None:
    groups = _group_datasets(datasets)
    first_ci = True

    for (group_name, vehicle_count) in _sorted_keys(groups):
        run_dfs = [df for df in groups[(group_name, vehicle_count)] if metric_col in df.columns]
        if not run_dfs:
            continue

        color = GROUP_N_COLORS.get(vehicle_count, "#ef3b2c")
        ctrl = CTRL_DISPLAY[group_name]
        n_runs = len(run_dfs)
        label = f"{ctrl}, N={vehicle_count} (n={n_runs})"

        time_ref = run_dfs[0][TIME_COL].to_numpy()
        metric_matrix = np.vstack([
            np.interp(time_ref, df[TIME_COL].to_numpy(), df[metric_col].to_numpy() * scale)
            for df in run_dfs
        ])

        mean_vals = metric_matrix.mean(axis=0)
        std_vals = metric_matrix.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean_vals)

        sem_vals = std_vals / np.sqrt(n_runs)
        ci_bound = 1.96 * sem_vals

        ci_label = "95% CI" if first_ci else ""
        ax.fill_between(time_ref, mean_vals - ci_bound, mean_vals + ci_bound,
                        color=color, alpha=0.20, edgecolor="none", label=ci_label)
        ax.plot(time_ref, mean_vals, color=_darken(color), linewidth=2.8, label=label)
        first_ci = False

    if hline is not None:
        ax.axhline(hline, color="#222222", linewidth=1.5,
                   linestyle=(0, (4, 3)), alpha=0.8,
                   label=hline_label or f"{hline:.2f}")

    ax.set_xlabel("Time, t, [s]")
    ax.set_ylabel(ylabel)

    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    elif any(x in metric_col.lower() for x in ["safety", "energy", "throughput"]):
        ax.set_ylim(bottom=0)

    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)

    _append_band_legend_bottom(ax, ncol=3)


# ---------- Energy vs distance plot -----------------------------------------

def plot_energy_metric_with_band(
    ax: plt.Axes, datasets: list, title: str, ylabel: str
) -> None:
    groups = _group_datasets(datasets)
    first_ci = True

    for (group_name, vehicle_count) in _sorted_keys(groups):
        run_dfs = [df for df in groups[(group_name, vehicle_count)] if DIST_COL in df.columns and ENERGY_COL in df.columns]
        if not run_dfs:
            continue

        color  = GROUP_N_COLORS.get(vehicle_count, "#ef3b2c")
        ctrl   = CTRL_DISPLAY[group_name]
        n_runs = len(run_dfs)
        label  = f"{ctrl}, N={vehicle_count} (n={n_runs})"

        distance_max  = min(df[DIST_COL].max() for df in run_dfs)
        distance_grid = np.linspace(0.0, distance_max, 500)
        energy_curves = []

        for df in run_dfs:
            energy_curves.append(
                np.interp(distance_grid, df[DIST_COL].to_numpy(), df[ENERGY_COL].to_numpy())
            )

        energy_matrix = np.vstack(energy_curves)
        mean_vals = energy_matrix.mean(axis=0)
        std_vals  = energy_matrix.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean_vals)

        sem_vals  = std_vals / np.sqrt(n_runs)
        ci_bound  = 1.96 * sem_vals

        ci_label = "95% CI" if first_ci else ""
        ax.fill_between(distance_grid, mean_vals - ci_bound, mean_vals + ci_bound,
                        color=color, alpha=0.20, edgecolor="none", label=ci_label)
        ax.plot(distance_grid, mean_vals, color=_darken(color), linewidth=2.8, label=label)
        first_ci = False

    ax.set_xlabel("Distance Travelled, d, [km]")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    _append_band_legend_bottom(ax, ncol=3)


# ---------- Per-vehicle speed band (single run) -----------------------------

def plot_speed_vehicle_band(ax: plt.Axes, dataset_item: tuple) -> None:
    _, run_label, group_name, vehicle_count, vehicle_csv, *_ = dataset_item

    if vehicle_csv is None or not Path(vehicle_csv).is_file():
        return

    vdf = pd.read_csv(vehicle_csv).sort_values(TIME_COL)
    vdf["velocity_mps"] = vdf["velocity_kmh"] / 3.6
    
    grouped = vdf.groupby(TIME_COL)["velocity_mps"].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)

    t        = grouped[TIME_COL].to_numpy()
    v_avg    = grouped["mean"].to_numpy()
    v_std    = grouped["std"].to_numpy()
    v_ci_bnd = 1.96 * (v_std / np.sqrt(grouped["count"].to_numpy()))

    color = GROUP_N_COLORS.get(vehicle_count, "#ef3b2c")
    ctrl  = CTRL_DISPLAY[group_name]

    ax.fill_between(t, v_avg - v_ci_bnd, v_avg + v_ci_bnd, color=color, alpha=0.20, edgecolor="none")
    ax.plot(t, v_avg, color=_darken(color), linewidth=2.8, label=f"{ctrl}, N={vehicle_count} — {run_label}")

    ax.axhline(TARGET_SPEED_MPS, color="#222222", linewidth=1.5,
               linestyle=(0, (4, 3)), alpha=0.8,
               label=f"Target speed ({TARGET_SPEED_MPS:.2f} m/s)")

    ax.set_xlabel("Time, t, [s]")
    ax.set_ylabel("Speed, v, [m/s]")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=8)

    handles, labels = ax.get_legend_handles_labels()
    ci_patch = mpatches.Patch(color=color, alpha=0.20, label="95% CI")
    ax.legend(handles=handles + [ci_patch], labels=labels + ["95% CI"],
              loc="best", ncol=2, frameon=False)


def plot_gap_vehicle_band(ax: plt.Axes, dataset_item: tuple) -> None:
    _, run_label, group_name, vehicle_count, vehicle_csv, *_ = dataset_item

    if vehicle_csv is None or not Path(vehicle_csv).is_file():
        return

    vdf = pd.read_csv(vehicle_csv).sort_values(TIME_COL)
    if "gap_meters" not in vdf.columns:
        return

    grouped = vdf.groupby(TIME_COL)["gap_meters"].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)

    t        = grouped[TIME_COL].to_numpy()
    g_avg    = grouped["mean"].to_numpy()
    g_std    = grouped["std"].to_numpy()
    g_ci_bnd = 1.96 * (g_std / np.sqrt(grouped["count"].to_numpy()))

    color = GAP_N_COLORS.get(vehicle_count, "#2E8B2E")
    ctrl  = CTRL_DISPLAY[group_name]
    target_gap = TRACK_LENGTH_M / vehicle_count

    ax.fill_between(t, g_avg - g_ci_bnd, g_avg + g_ci_bnd, color=color, alpha=0.20, edgecolor="none")
    ax.plot(t, g_avg, color=_darken(color), linewidth=2.8, label=f"{ctrl}, N={vehicle_count} — {run_label}")

    ax.axhline(target_gap, color="#222222", linewidth=1.5,
               linestyle=(0, (4, 3)), alpha=0.8,
               label=f"Target gap ({target_gap:.1f} m)")

    ax.set_xlabel("Time, t, [s]")
    ax.set_ylabel("Gap, g, [m]")
    ax.set_ylim(bottom=target_gap - 2.5, top=target_gap + 2.5)

    handles, labels = ax.get_legend_handles_labels()
    ci_patch = mpatches.Patch(color=color, alpha=0.20, label="95% CI")
    ax.legend(handles=handles + [ci_patch], labels=labels + ["95% CI"],
              loc="best", ncol=2, frameon=False)


# ---------- Gap all-runs plot -----------------------------------------------

def plot_gap_all_runs_with_band(ax: plt.Axes, datasets: list) -> None:
    groups: dict[tuple[str, int], list] = {}
    for item in datasets:
        _, _, group_name, vehicle_count, vehicle_csv, *_ = item
        groups.setdefault((group_name, vehicle_count), []).append((vehicle_csv, vehicle_count))

    first_ci = True
    target_hlines: dict[int, float] = {}

    for (group_name, vehicle_count) in _sorted_keys(groups):
        items = groups[(group_name, vehicle_count)]
        target_gap = TRACK_LENGTH_M / vehicle_count
        target_hlines[vehicle_count] = target_gap

        run_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for vehicle_csv, _ in items:
            if vehicle_csv is None or not Path(vehicle_csv).is_file():
                continue
            try:
                vdf = pd.read_csv(vehicle_csv).sort_values(TIME_COL)
            except Exception:
                continue
            if "gap_meters" not in vdf.columns:
                continue
            grouped = vdf.groupby(TIME_COL)["gap_meters"].mean().reset_index()
            run_curves.append((grouped[TIME_COL].to_numpy(), grouped["gap_meters"].to_numpy()))

        if not run_curves:
            continue

        color = GAP_N_COLORS.get(vehicle_count, "#2E8B2E")
        ctrl = CTRL_DISPLAY[group_name]
        n_runs = len(run_curves)
        label = f"{ctrl}, N={vehicle_count} (n={n_runs})"

        time_ref = run_curves[0][0]
        gap_matrix = np.vstack([
            np.interp(time_ref, t, g) for t, g in run_curves
        ])

        mean_vals = gap_matrix.mean(axis=0)
        std_vals = gap_matrix.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mean_vals)
        ci_bound = 1.96 * std_vals / np.sqrt(n_runs)

        ci_label = "95% CI" if first_ci else ""
        ax.fill_between(time_ref, mean_vals - ci_bound, mean_vals + ci_bound,
                        color=color, alpha=0.30, edgecolor="none", label=ci_label)
        ax.plot(time_ref, mean_vals, color=_darken(color), linewidth=2.8, label=label)
        first_ci = False

    linestyles = ["--", "-.", ":", (0, (4, 3))]
    for i, (vc, tg) in enumerate(sorted(target_hlines.items())):
        ax.axhline(tg, color="#222222", linewidth=1.4,
                   linestyle=linestyles[i % len(linestyles)], alpha=0.8,
                   label=f"Target gap N={vc} ({tg:.1f} m)")

    ax.set_title("Average Gap vs. Time — Barrier Control Batches")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gap [m]")
    if target_hlines:
        ax.set_ylim(bottom=min(target_hlines.values()) - 0.5,
                    top=max(target_hlines.values()) + 0.5)
    _append_band_legend_bottom(ax, ncol=3)


# ---------- Convergence time ------------------------------------------------

def compute_convergence_time(t: np.ndarray, vals: np.ndarray, target: float, tol: float = 0.10) -> float:
    """Return the first time vals permanently stays within tol*100 % of target.

    Uses the *last-exit* definition: the timestep after the last sample that
    falls outside the band.  Returns NaN if the signal never converges.
    """
    within = np.abs(vals - target) <= tol * abs(target)
    outside = np.where(~within)[0]
    if len(outside) == 0:
        return float(t[0])
    last_out = outside[-1]
    if last_out + 1 >= len(t):
        return float("nan")
    return float(t[last_out + 1])


def report_convergence_times(datasets: list) -> None:
    speed_times: dict[int, list[float]] = {}
    gap_times:   dict[int, list[float]] = {}
    target_gaps: dict[int, float]       = {}

    for item in datasets:
        _, _, _, vehicle_count, vehicle_csv, *_ = item
        if vehicle_csv is None or not Path(vehicle_csv).is_file():
            continue
        try:
            vdf = pd.read_csv(vehicle_csv).sort_values(TIME_COL)
        except Exception:
            continue

        tg = TRACK_LENGTH_M / vehicle_count
        target_gaps[vehicle_count] = tg

        if "velocity_kmh" in vdf.columns:
            sg = vdf.groupby(TIME_COL)["velocity_kmh"].mean().reset_index()
            ct = compute_convergence_time(
                sg[TIME_COL].to_numpy(), sg["velocity_kmh"].to_numpy() / 3.6, TARGET_SPEED_MPS
            )
            speed_times.setdefault(vehicle_count, []).append(ct)

        if "gap_meters" in vdf.columns:
            gg = vdf.groupby(TIME_COL)["gap_meters"].mean().reset_index()
            ct = compute_convergence_time(gg[TIME_COL].to_numpy(), gg["gap_meters"].to_numpy(), tg)
            gap_times.setdefault(vehicle_count, []).append(ct)

    print("\n=== Convergence Times (within ±10 % of target) ===")
    all_n = sorted(set(list(speed_times) + list(gap_times)))
    for vc in all_n:
        tg = target_gaps.get(vc, float("nan"))
        st = [x for x in speed_times.get(vc, []) if not np.isnan(x)]
        gt = [x for x in gap_times.get(vc, [])   if not np.isnan(x)]
        n_sp = len(speed_times.get(vc, []))
        n_gp = len(gap_times.get(vc, []))

        sp_mean = np.mean(st) if st else float("nan")
        sp_std  = np.std(st, ddof=1) if len(st) > 1 else 0.0
        gp_mean = np.mean(gt) if gt else float("nan")
        gp_std  = np.std(gt, ddof=1) if len(gt) > 1 else 0.0

        print(f"\n  N={vc}  (target speed={TARGET_SPEED_KMH} km/h, target gap={tg:.1f} m)")
        print(f"    Speed convergence : {sp_mean:6.1f} ± {sp_std:.1f} s  ({len(st)}/{n_sp} runs converged)")
        print(f"    Gap   convergence : {gp_mean:6.1f} ± {gp_std:.1f} s  ({len(gt)}/{n_gp} runs converged)")


# ---------- Stability plot — one lap ----------------------------------------

def plot_stability_one_lap(
    ax_speed: plt.Axes,
    ax_gap: plt.Axes,
    datasets: list,
    n_filter: int = 77,
) -> None:
    """Individual vehicle speed & gap traces over one lap [STABILITY_START_S, STABILITY_END_S]."""
    n_items = [item for item in datasets if item[3] == n_filter]
    if not n_items:
        return

    target_gap = TRACK_LENGTH_M / n_filter
    group_name = n_items[0][2]
    ctrl       = CTRL_DISPLAY[group_name]
    color_spd  = GROUP_N_COLORS.get(n_filter, "#ef3b2c")
    color_gap  = GAP_N_COLORS.get(n_filter, "#2E8B2E")

    # Collect per-vehicle traces across all runs
    all_spd_t: list[np.ndarray] = []
    all_spd_v: list[np.ndarray] = []
    all_gap_t: list[np.ndarray] = []
    all_gap_g: list[np.ndarray] = []

    for item in n_items:
        _, _, _, _, vehicle_csv, *_ = item
        if vehicle_csv is None or not Path(vehicle_csv).is_file():
            continue
        try:
            vdf = pd.read_csv(vehicle_csv).sort_values(TIME_COL)
        except Exception:
            continue

        mask    = (vdf[TIME_COL] >= STABILITY_START_S) & (vdf[TIME_COL] <= STABILITY_END_S)
        vdf_lap = vdf[mask]
        if vdf_lap.empty or "vehicle_id" not in vdf_lap.columns:
            continue

        for _, grp in vdf_lap.groupby("vehicle_id", sort=False):
            t = grp[TIME_COL].to_numpy()

            if "velocity_kmh" in grp.columns:
                v = grp["velocity_kmh"].to_numpy() / 3.6
                all_spd_t.append(t)
                all_spd_v.append(v)

            if "gap_meters" in grp.columns:
                g = grp["gap_meters"].to_numpy()
                all_gap_t.append(t)
                all_gap_g.append(g)

    n_runs = len(n_items)

    def _overlay(ax, all_t, all_v, target, color, ylabel, title, target_label):
        if not all_t:
            return

        t_min = max(t[0]  for t in all_t)
        t_max = min(t[-1] for t in all_t)
        t_ref  = np.linspace(t_min, t_max, 600)
        matrix = np.vstack([np.interp(t_ref, t, v) for t, v in zip(all_t, all_v)])
        mean_v = matrix.mean(axis=0)
        std_v  = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean_v)
        ci_bnd = 1.96 * std_v / np.sqrt(matrix.shape[0])

        # Dynamic y-limits: center on target with margin derived from actual data spread
        spread = max((mean_v + ci_bnd).max() - (mean_v - ci_bnd).min(), 0.01 * abs(target))
        margin = spread * 0.3
        ylim_bottom = min((mean_v - ci_bnd).min(), target) - margin
        ylim_top    = max((mean_v + ci_bnd).max(), target) + margin

        ax.fill_between(t_ref, mean_v - ci_bnd, mean_v + ci_bnd,
                        color=color, alpha=0.30, edgecolor="none", label="95% CI")
        ax.plot(t_ref, mean_v, color=_darken(color), linewidth=2.5, zorder=5,
                label=f"Mean — {ctrl}, N={n_filter} ({n_runs} runs)")
        ax.axhline(target, color="#222222", linewidth=1.4,
                   linestyle=(0, (4, 3)), alpha=0.8, zorder=6, label=target_label)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)

        ax.set_xlim(STABILITY_START_S, STABILITY_END_S)
        ax.set_ylim(ylim_bottom, ylim_top)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    _overlay(
        ax_speed, all_spd_t, all_spd_v, TARGET_SPEED_MPS, color_spd,
        "Speed [m/s]",
        f"Speed Stability — One Lap ({STABILITY_START_S:.0f}–{STABILITY_END_S:.0f} s), N={n_filter}",
        f"Target speed ({TARGET_SPEED_MPS:.2f} m/s)",
    )
    _overlay(
        ax_gap, all_gap_t, all_gap_g, target_gap, color_gap,
        "Gap [m]",
        f"Gap Stability — One Lap ({STABILITY_START_S:.0f}–{STABILITY_END_S:.0f} s), N={n_filter}",
        f"Target gap ({target_gap:.1f} m)",
    )


# ---------- Run-level distribution plots ------------------------------------

def _read_csv_safely(path: Path | str | None, **kwargs) -> pd.DataFrame | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def build_run_summary_table(datasets: list) -> pd.DataFrame:
    """One row per completed run, used for boxplots and compact comparisons."""
    rows: list[dict] = []

    for item in datasets:
        df, run_label, group_name, vehicle_count, vehicle_csv, *rest = item
        lap_csv = rest[0] if rest else None
        if df.empty:
            continue

        final = df.iloc[-1]
        total_distance_km = float(final.get(DIST_COL, np.nan))
        energy_kwh = float(final.get(ENERGY_COL, np.nan))
        energy_per_100km = (
            100.0 * energy_kwh / total_distance_km
            if np.isfinite(total_distance_km) and total_distance_km > 0
            else np.nan
        )

        headway_p95_abs = np.nan
        headway_mean_abs = np.nan
        vdf = _read_csv_safely(
            vehicle_csv,
            usecols=lambda c: c in {TIME_COL, "headway_error_meters"},
        )
        if vdf is not None and "headway_error_meters" in vdf.columns:
            eval_mask = vdf[TIME_COL] >= HEADWAY_EVAL_START_S
            err = vdf.loc[eval_mask, "headway_error_meters"].abs()
            if not err.empty:
                headway_p95_abs = float(err.quantile(0.95))
                headway_mean_abs = float(err.mean())

        lap_duration_mean = np.nan
        lap_duration_p95 = np.nan
        lap_delay_mean = np.nan
        lap_df = _read_csv_safely(lap_csv)
        if lap_df is not None and not lap_df.empty:
            if "lap_duration_s" in lap_df.columns:
                lap_duration_mean = float(lap_df["lap_duration_s"].mean())
                lap_duration_p95 = float(lap_df["lap_duration_s"].quantile(0.95))
            if "delay_time_s" in lap_df.columns:
                lap_delay_mean = float(lap_df["delay_time_s"].mean())

        barrier_pair_events = np.nan
        if BARRIER_FIRINGS_COL in df.columns:
            # The throughput log counts affected vehicles. A conflict-pair firing
            # normally appears as one positive and one negative correction.
            barrier_pair_events = float(df[BARRIER_FIRINGS_COL].sum()) / 2.0

        rows.append({
            "group_name": group_name,
            "run_index": _run_index(run_label),
            "run_label": run_label,
            "vehicle_count": vehicle_count,
            "final_throughput_veh_s": float(final.get(THROUGHPUT_COL, np.nan)),
            "energy_per_100km": energy_per_100km,
            "total_delay_s": float(final.get(DELAY_COL, np.nan)),
            "safety_violation_rate": float(final.get(SAFETY_COL, np.nan)),
            "lap_duration_mean_s": lap_duration_mean,
            "lap_duration_p95_s": lap_duration_p95,
            "lap_delay_mean_s": lap_delay_mean,
            "headway_p95_abs_error_m": headway_p95_abs,
            "headway_mean_abs_error_m": headway_mean_abs,
            "barrier_pair_events": barrier_pair_events,
        })

    return pd.DataFrame(rows)


def _boxplot_by_vehicle_count(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    target_line: float | None = None,
    target_label: str | None = None,
) -> None:
    plot_df = summary_df[["vehicle_count", metric_col]].dropna()
    if plot_df.empty:
        ax.set_visible(False)
        return

    vehicle_counts = sorted(plot_df["vehicle_count"].unique())
    data = [
        plot_df.loc[plot_df["vehicle_count"] == vc, metric_col].to_numpy()
        for vc in vehicle_counts
    ]

    bp = ax.boxplot(
        data,
        tick_labels=[f"N={vc}\n(n={len(vals)})" for vc, vals in zip(vehicle_counts, data)],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.7},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
    )

    rng = np.random.default_rng(7)
    for idx, (vc, vals) in enumerate(zip(vehicle_counts, data), start=1):
        color = GROUP_N_COLORS.get(int(vc), "#ef3b2c")
        bp["boxes"][idx - 1].set(facecolor=color, alpha=0.28, edgecolor=_darken(color), linewidth=1.3)
        x = rng.normal(idx, 0.035, size=len(vals))
        ax.scatter(x, vals, s=22, color=_darken(color), alpha=0.72, linewidth=0, zorder=3)

    if target_line is not None:
        ax.axhline(
            target_line,
            color="#222222",
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            alpha=0.8,
            label=target_label or f"{target_line:.2f}",
        )
        ax.legend(loc="best", fontsize=8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")


def plot_run_summary_boxplots(axes: np.ndarray, summary_df: pd.DataFrame) -> None:
    axes = axes.ravel()
    specs = [
        ("final_throughput_veh_s", "Final Throughput", "Throughput [veh/s]", None, None),
        ("energy_per_100km", "Energy Intensity", "Energy [kWh / 100 km]", None, None),
        ("total_delay_s", "Total Delay", "Delay [s]", None, None),
        ("lap_duration_mean_s", "Mean Lap Duration", "Lap duration [s]", LAP_DURATION_S, f"Nominal lap ({LAP_DURATION_S:.2f} s)"),
        ("headway_p95_abs_error_m", "Gap Tracking Error", "P95 |gap error| [m]", HEADWAY_TARGET_BAND_M, f"Target band ({HEADWAY_TARGET_BAND_M:.1f} m)"),
        ("barrier_pair_events", "Barrier Activity", "Pair events [count/run]", None, None),
    ]

    for ax, (metric, title, ylabel, target, target_label) in zip(axes, specs):
        _boxplot_by_vehicle_count(ax, summary_df, metric, title, ylabel, target, target_label)

    for ax in axes[len(specs):]:
        ax.set_visible(False)


# ---------- Barrier conflict-pair response ---------------------------------

def _read_vehicle_event_frame(vehicle_csv: Path | str | None) -> pd.DataFrame | None:
    return _read_csv_safely(
        vehicle_csv,
        usecols=lambda c: c in {
            TIME_COL, "vehicle_id", "velocity_kmh", "gap_meters",
            "target_gap_meters", "headway_error_meters", "u_barrier",
        },
    )


def _collect_conflict_event_candidates(
    datasets: list,
    preferred_n: int | None = None,
    preferred_run: int | None = None,
    min_time_s: float | None = None,
) -> list[dict]:
    candidates: list[dict] = []

    for item in datasets:
        _, run_label, group_name, vehicle_count, vehicle_csv, *_ = item
        run_idx = _run_index(run_label)
        if preferred_n is not None and vehicle_count != preferred_n:
            continue
        if preferred_run is not None and run_idx != preferred_run:
            continue

        vdf = _read_vehicle_event_frame(vehicle_csv)
        if vdf is None or "u_barrier" not in vdf.columns:
            continue

        active = vdf[vdf["u_barrier"].abs() > BARRIER_EPS]
        if min_time_s is not None:
            active = active[active[TIME_COL] >= min_time_s]
        if active.empty:
            continue

        for event_time, grp in active.groupby(TIME_COL):
            neg = grp[grp["u_barrier"] < -BARRIER_EPS]
            pos = grp[grp["u_barrier"] > BARRIER_EPS]
            if neg.empty or pos.empty:
                continue

            neg_row = neg.loc[neg["u_barrier"].idxmin()]
            pos_row = pos.loc[pos["u_barrier"].idxmax()]
            score = min(abs(float(neg_row["u_barrier"])), abs(float(pos_row["u_barrier"])))

            candidates.append({
                "dataset_item": item,
                "group_name": group_name,
                "vehicle_count": vehicle_count,
                "run_index": run_idx,
                "run_label": run_label,
                "event_time": float(event_time),
                "negative_vehicle_id": int(neg_row["vehicle_id"]),
                "positive_vehicle_id": int(pos_row["vehicle_id"]),
                "negative_u_barrier": float(neg_row["u_barrier"]),
                "positive_u_barrier": float(pos_row["u_barrier"]),
                "score": score,
            })

    return candidates


def find_conflict_pair_event(datasets: list) -> dict | None:
    preferred = _collect_conflict_event_candidates(
        datasets,
        preferred_n=PREFERRED_CONFLICT_N,
        preferred_run=PREFERRED_CONFLICT_RUN,
    )
    if preferred:
        return max(preferred, key=lambda row: row["score"])

    post_warmup = _collect_conflict_event_candidates(datasets, min_time_s=RAMP_UP_TIME_S)
    if post_warmup:
        return max(post_warmup, key=lambda row: row["score"])

    all_candidates = _collect_conflict_event_candidates(datasets)
    if all_candidates:
        return max(all_candidates, key=lambda row: row["score"])

    return None


def plot_barrier_conflict_pair_response(axes: np.ndarray, event: dict) -> None:
    item = event["dataset_item"]
    _, run_label, group_name, vehicle_count, vehicle_csv, *_ = item
    vdf = _read_vehicle_event_frame(vehicle_csv)
    if vdf is None or vdf.empty:
        return

    event_time = event["event_time"]
    start_time = max(0.0, event_time - CONFLICT_WINDOW_BEFORE_S)
    end_time = event_time + CONFLICT_WINDOW_AFTER_S
    pair = [
        ("negative", event["negative_vehicle_id"], event["negative_u_barrier"]),
        ("positive", event["positive_vehicle_id"], event["positive_u_barrier"]),
    ]
    pair_ids = [vid for _, vid, _ in pair]

    focus = vdf[
        (vdf[TIME_COL] >= start_time) &
        (vdf[TIME_COL] <= end_time) &
        (vdf["vehicle_id"].isin(pair_ids))
    ].copy()
    if focus.empty:
        return

    if "headway_error_meters" not in focus.columns and {"gap_meters", "target_gap_meters"}.issubset(focus.columns):
        focus["headway_error_meters"] = focus["gap_meters"] - focus["target_gap_meters"]

    ctrl = CTRL_DISPLAY[group_name]
    axes[0].set_title(
        f"Barrier Firing Response — {ctrl}, {run_label}, N={vehicle_count}, t={event_time:.0f} s"
    )

    for ax in axes:
        ax.axvspan(event_time - 0.5, event_time + 0.5, color="#DDE5F3", alpha=0.85, zorder=0)
        ax.axvline(event_time, color="#222222", linewidth=1.0, linestyle=(0, (2, 3)), alpha=0.75)
        ax.set_xlim(start_time, end_time)

    for sign_key, vid, event_u in pair:
        color = BARRIER_SIGN_COLORS[sign_key]
        grp = focus[focus["vehicle_id"] == vid].sort_values(TIME_COL)
        sign_label = "negative" if sign_key == "negative" else "positive"
        label = f"Vehicle {vid} ({sign_label}, {event_u:+.2f} km/h)"

        axes[0].plot(grp[TIME_COL], grp["u_barrier"], color=color, linewidth=1.9, marker="o", markersize=3.5, label=label)
        axes[0].scatter([event_time], [event_u], s=70, color=color, edgecolor="white", linewidth=0.8, zorder=5)

        if "velocity_kmh" in grp.columns:
            axes[1].plot(grp[TIME_COL], grp["velocity_kmh"], color=color, linewidth=2.2)
            event_speed = grp.loc[np.isclose(grp[TIME_COL], event_time), "velocity_kmh"]
            if not event_speed.empty:
                axes[1].scatter([event_time], [event_speed.iloc[0]], s=58, color=color, edgecolor="white", linewidth=0.8, zorder=5)

        if "headway_error_meters" in grp.columns:
            axes[2].plot(grp[TIME_COL], grp["headway_error_meters"], color=color, linewidth=2.2)
            event_gap_error = grp.loc[np.isclose(grp[TIME_COL], event_time), "headway_error_meters"]
            if not event_gap_error.empty:
                axes[2].scatter([event_time], [event_gap_error.iloc[0]], s=58, color=color, edgecolor="white", linewidth=0.8, zorder=5)

    axes[0].axhline(0, color="#222222", linewidth=1.0, alpha=0.7)
    axes[1].axhline(TARGET_SPEED_KMH, color="#222222", linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.8)
    axes[2].axhline(0, color="#222222", linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.8)

    axes[0].set_ylabel("Barrier correction [km/h]")
    axes[1].set_ylabel("Speed [km/h]")
    axes[2].set_ylabel("Gap error [m]")
    axes[2].set_xlabel("Time [s]")

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)


# ---------- Main ------------------------------------------------------------

datasets: list = []
for group_name, batch_dir in BATCHES.items():
    datasets.extend(load_batch_runs(group_name, batch_dir))

print("\n=== Aligning run indices across controllers ===")
datasets = align_run_counts(datasets)

# ------ Figures Setup --------
fig_throughput, ax_throughput = plt.subplots(figsize=(12, 7))
plot_throughput_combined(ax_throughput, datasets)
save_and_show(fig_throughput, "throughput_vs_time_all.png")

fig_energy, ax_energy = plt.subplots(figsize=(12, 7))
plot_energy_metric_with_band(
    ax_energy, datasets,
    "Energy vs. Distance — Barrier Control Batches",
    "Energy consumption [kWh]",
)
save_and_show(fig_energy, "energy_vs_distance_all.png")

fig_speed, ax_speed = plt.subplots(figsize=(12, 7))
plot_time_metric_with_band(
    ax_speed, datasets, SPEED_COL,
    "Average Speed vs. Time — Barrier Control Batches",
    "Average speed [m/s]",
    scale=1 / 3.6,
    ylim_bottom=8.3,
    ylim_top=8.45,
    hline=TARGET_SPEED_MPS,
    hline_label=f"Target speed ({TARGET_SPEED_MPS:.2f} m/s)",
)
save_and_show(fig_speed, "speed_vs_time_all.png")

fig_gap_all, ax_gap_all = plt.subplots(figsize=(12, 7))
plot_gap_all_runs_with_band(ax_gap_all, datasets)
save_and_show(fig_gap_all, "gap_vs_time_all.png")

# ------ Run-level distributions / boxplots ---------------------------------
run_summary_df = build_run_summary_table(datasets)
fig_boxplots, axes_boxplots = plt.subplots(2, 3, figsize=(15, 9))
plot_run_summary_boxplots(axes_boxplots, run_summary_df)
save_and_show(fig_boxplots, "run_summary_boxplots.png")

# ------ Conflict-pair barrier firing response -------------------------------
conflict_event = find_conflict_pair_event(datasets)
if conflict_event is not None:
    fig_conflict, axes_conflict = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plot_barrier_conflict_pair_response(axes_conflict, conflict_event)
    save_and_show(
        fig_conflict,
        f"barrier_conflict_pair_run{conflict_event['run_index']:02d}_N{conflict_event['vehicle_count']}.png",
    )
else:
    print("No conflict-pair barrier event with opposite signed corrections found.")

# ------ Per-vehicle speed & gap — Run 1 only --------------------------------
datasets_run1 = [item for item in datasets if _run_index(item[1]) == 1]
_n_run1 = datasets_run1[0][3] if datasets_run1 else "?"

fig_speed_r1, ax_speed_r1 = plt.subplots(figsize=(12, 7))
for item in datasets_run1:
    plot_speed_vehicle_band(ax_speed_r1, item)
save_and_show(fig_speed_r1, f"speed_vs_time_run1_N={_n_run1}.png")

fig_gap_r1, ax_gap_r1 = plt.subplots(figsize=(12, 7))
for item in datasets_run1:
    plot_gap_vehicle_band(ax_gap_r1, item)
save_and_show(fig_gap_r1, f"gap_vs_time_run1_N={_n_run1}.png")

# ------ Total delay ---------------------------------------------------------
fig_delay, ax_delay = plt.subplots(figsize=(12, 7))
plot_time_metric_with_band(
    ax_delay, datasets, DELAY_COL,
    "Total Delay vs. Time — Barrier Control Batches",
    "Total delay [s]",
)
save_and_show(fig_delay, "delay_vs_time_all.png")

# ------ Safety violation rate -----------------------------------------------
fig_safety, ax_safety = plt.subplots(figsize=(12, 7))
plot_time_metric_with_band(
    ax_safety, datasets, SAFETY_COL,
    "Safety Violation Rate vs. Time — Barrier Control Batches",
    "Safety violation rate",
    ylim_bottom=0.0,
)
save_and_show(fig_safety, "safety_violation_rate_vs_time_all.png")

# ------ Convergence times (printed to console) ------------------------------
report_convergence_times(datasets)

# ------ Stability plot — one lap, one figure per N present in batch ---------
unique_n_values = sorted({item[3] for item in datasets})
for n_val in unique_n_values:
    fig_stab_spd, ax_stab_spd = plt.subplots(figsize=(12, 7))
    fig_stab_gap, ax_stab_gap = plt.subplots(figsize=(12, 7))
    plot_stability_one_lap(ax_stab_spd, ax_stab_gap, datasets, n_filter=n_val)
    save_and_show(fig_stab_spd, f"stability_one_lap_speed_N{n_val}.png")
    save_and_show(fig_stab_gap, f"stability_one_lap_gap_N{n_val}.png")

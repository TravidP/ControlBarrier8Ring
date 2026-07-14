#!/usr/bin/env python3
"""
Publication-quality plots for the 61/71/81 vehicle traffic-control batches.

Figures produced:
  1. Representative vehicle behaviour, five shared-x panels.
  2. KPI mean comparison with 95% confidence intervals.
  3. KPI distributions for energy intensity and delay.

Controller mapping used here:
  Controller A = FAS / traffic-light batch
  Controller B = barrier-control batch

If your final 30-run folders move, only edit BATCH_DIRS below.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


# ---------- Paths and experiment setup --------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR / "simulation_results"
OUTPUT_DIR = BASE_DIR / "PlotsControllers" / "PublicationQuality_51_61_71_81" / "output"

DEMAND_LEVELS = [51, 61, 71, 81]

# Keep this simple and explicit. Each tuple can contain one or more folders;
# all complete rows from all listed summary CSVs are loaded.
BATCH_DIRS: dict[tuple[str, int], tuple[Path, ...]] = {
    ("A", 51): (
        BASE_DIR / "TrafficlightResults" / "Batch_FAS_N=51_final",
    ),
    ("A", 61): (
        BASE_DIR / "TrafficlightResults" / "Batch_FAS_N=61_final",
    ),
    ("A", 71): (
        BASE_DIR / "TrafficlightResults" / "Batch_FAS_N=71_final",
    ),
    ("A", 81): (
        BASE_DIR / "TrafficlightResults" / "Batch_FAS_N=81_final",
         ),
    ("B", 51): (
        BASE_DIR / "BarrierControlResults" / "Batch_barrier_N=51_final",
    ),
    ("B", 61): (
        BASE_DIR / "BarrierControlResults" / "Batch_barrier_N=61_final",
    ),
    ("B", 71): (
        BASE_DIR / "BarrierControlResults" / "Batch_barrier_N=71_final",
    ),
    ("B", 81): (
        BASE_DIR / "BarrierControlResults" / "Batch_barrier_N=81_final",
    ),
}

CONTROLLER_LABELS = {
    "A": "FAS Controller",
    "B": "Barrier Controller",
}
CONTROLLER_COLORS = {"A": "#0072B2", "B": "#E69F00"}  # blue, orange
DEMAND_COLORS = {
    51: "#1F77B4",  # blue
    61: "#D62728",  # red
    71: "#2CA02C",  # green
    81: "#FFD23F",  # yellow
}
CONTROLLER_LINESTYLES = {"A": "-", "B": "--"}

COLORS = {
    "target": "#111111",
    "barrier": "#D62728",
    "gap": "#009E73",
    "acceleration": "#E69F00",
    "propulsion": "#009E73",
    "regen": "#D62728",
}

TIME_COL = "time_seconds"
TARGET_SPEED_KMH = 30.0
TARGET_SPEED_MS = TARGET_SPEED_KMH / 3.6
BARRIER_EPS = 1e-6
MAX_RUNS_PER_CONFIG = 30
POWER_BIN_S = 1.0
REPRESENTATIVE_RUN_INDEX = 1
REPRESENTATIVE_VEHICLE_POSITION = 1
SAFETY_VIOLATION_COLS = [
    "safety_violation", "violation", "is_violation",
    "constraint_violation", "gap_violation", "ttc_violation",
    "n_violations", "num_violations", "violated",
]
DEMAND_LINESTYLES = {51: "-.", 61: "-", 71: "--", 81: ":"}


# ---------- Style -----------------------------------------------------------

def apply_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    else:
        print("Seaborn is not installed; using a Matplotlib white-grid fallback.")
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#BBBBBB",
        "axes.linewidth": 0.8,
        "grid.color": "#BBBBBB",
        "grid.alpha": 0.28,
        "grid.linewidth": 0.8,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------- Loading ---------------------------------------------------------

def _summary_files(batch_dir: Path) -> list[Path]:
    return sorted(batch_dir.glob("batch_summary_*.csv")) if batch_dir.is_dir() else []


def _resolve_csv(row: pd.Series, batch_dir: Path, column: str, pattern: str) -> Path | None:
    raw = row.get(column, "")
    if pd.notna(raw):
        candidate = Path(str(raw))
        if candidate.is_file():
            return candidate

        # Some copied summaries contain absolute paths to the original batch
        # name while the actual files live in the wrapper folder.
        local_same_name = batch_dir / f"run_{int(row['run_index']):02d}" / candidate.name
        if local_same_name.is_file():
            return local_same_name

    run_dir = batch_dir / f"run_{int(row['run_index']):02d}"
    hits = sorted(run_dir.glob(pattern))
    return hits[0] if hits else None


def load_summary_rows(controller: str, demand: int, batch_dirs: tuple[Path, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_priority, batch_dir in enumerate(batch_dirs):
        for csv_path in _summary_files(batch_dir):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"Warning: could not read {csv_path}: {exc}")
                continue
            if df.empty:
                continue
            df = df.copy()
            df["source_priority"] = source_priority
            df["source_batch_dir"] = str(batch_dir)
            df["source_summary_csv"] = str(csv_path)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.columns = out.columns.str.strip()
    out = out[out["final_time_seconds"] >= out["duration_s"] * 0.99].copy()
    out["controller"] = controller
    out["controller_label"] = CONTROLLER_LABELS[controller]
    out["demand"] = demand
    out["run_order"] = np.arange(1, len(out) + 1)

    resolved_vehicle: list[str | None] = []
    resolved_throughput: list[str | None] = []
    for _, row in out.iterrows():
        batch_dir = Path(row["source_batch_dir"])
        vehicle_csv = _resolve_csv(row, batch_dir, "vehicle_csv", "vehicle_all*.csv")
        throughput_csv = _resolve_csv(row, batch_dir, "throughput_csv", "throughput_*.csv")
        resolved_vehicle.append(str(vehicle_csv) if vehicle_csv else None)
        resolved_throughput.append(str(throughput_csv) if throughput_csv else None)

    out["resolved_vehicle_csv"] = resolved_vehicle
    out["resolved_throughput_csv"] = resolved_throughput
    out = out.drop_duplicates(
        subset=["controller", "demand", "run_index", "resolved_throughput_csv"],
        keep="first",
    )
    out = (
        out.sort_values(["run_index", "source_priority", "source_summary_csv"])
        .drop_duplicates(subset=["controller", "demand", "run_index"], keep="first")
        .head(MAX_RUNS_PER_CONFIG)
    )
    return out.reset_index(drop=True)


def load_all_summaries() -> dict[tuple[str, int], pd.DataFrame]:
    data: dict[tuple[str, int], pd.DataFrame] = {}
    for key, dirs in BATCH_DIRS.items():
        controller, demand = key
        df = load_summary_rows(controller, demand, dirs)
        data[key] = df
        label = f"{CONTROLLER_LABELS[controller]} N={demand}"
        if df.empty:
            print(f"{label}: 0 complete runs found")
        else:
            print(f"{label}: {len(df)} complete runs found")
    return data


def build_kpi_table(data: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (controller, demand), df in data.items():
        if df.empty:
            continue

        kpi = pd.DataFrame({
            "controller": controller,
            "controller_label": CONTROLLER_LABELS[controller],
            "demand": demand,
            "run_index": df["run_index"].astype(int),
            "energy_kwh_per_km": df["energy_kWh"] / df["total_distance_km"],
            "throughput_veh_s": df["final_throughput"],
            "safety_violations_count": df.get("total_safety_violations", np.nan),
            "delay_s": df["total_delay_s"],
        })
        rows.append(kpi)

    if not rows:
        raise RuntimeError("No completed runs were loaded. Check BATCH_DIRS.")
    return pd.concat(rows, ignore_index=True)


def ci95(values: pd.Series) -> tuple[float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(clean.size)
    mean = float(clean.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan"), n
    sem = float(clean.std(ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, t_crit * sem, n


# ---------- Figure helpers --------------------------------------------------

def save_figure(fig: plt.Figure, stem: str, subdir: Path | None = None) -> None:
    target_dir = subdir if subdir is not None else OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (
        ("png", {"dpi": 300, "bbox_inches": "tight"}),
        ("pdf", {"bbox_inches": "tight"}),
    ):
        path = target_dir / f"{stem}.{ext}"
        fig.savefig(path, **kwargs)
        print(f"Saved {path}")


def _set_time_ticks(ax: plt.Axes) -> None:
    """Always extend x-axis to 1800 s and add it as an explicit tick."""
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, 1800.0))
    existing = [round(v) for v in ax.get_xticks()]
    if 1800 not in existing:
        ax.set_xticks(sorted(existing + [1800]))


def _representative_rows(data: dict[tuple[str, int], pd.DataFrame]) -> list[pd.Series]:
    rows: list[pd.Series] = []
    for controller in ("A", "B"):
        for demand in DEMAND_LEVELS:
            df = data.get((controller, demand), pd.DataFrame())
            if df.empty:
                print(f"Warning: no Figure 1 data for {CONTROLLER_LABELS[controller]} N={demand}")
                continue

            with_csv = df[
                df["resolved_vehicle_csv"].apply(lambda p: isinstance(p, str) and Path(p).is_file())
            ].copy()
            if with_csv.empty:
                print(f"Warning: no vehicle CSV for Figure 1: {CONTROLLER_LABELS[controller]} N={demand}")
                continue

            exact = with_csv[with_csv["run_index"].astype(int) == REPRESENTATIVE_RUN_INDEX]
            if exact.empty:
                row = with_csv.sort_values("run_index").iloc[0]
                print(
                    f"Warning: run {REPRESENTATIVE_RUN_INDEX} unavailable for "
                    f"{CONTROLLER_LABELS[controller]} N={demand}; using run {int(row['run_index'])}"
                )
            else:
                row = exact.sort_values("run_index").iloc[0]
            rows.append(row)
    return rows


def _selected_vehicle_trace(vehicle_csv: Path, vehicle_position: int) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    df = pd.read_csv(vehicle_csv)
    df.columns = df.columns.str.strip()
    if "vehicle_id" not in df:
        raise ValueError(f"{vehicle_csv} has no vehicle_id column")

    vehicle_ids = sorted(df["vehicle_id"].dropna().unique())
    if not vehicle_ids:
        raise ValueError(f"{vehicle_csv} has no vehicle rows")
    position_index = min(max(vehicle_position, 1), len(vehicle_ids)) - 1
    selected_id = vehicle_ids[position_index]
    trace = df[df["vehicle_id"] == selected_id].sort_values(TIME_COL).copy()
    trace = trace.drop_duplicates(subset=[TIME_COL])
    return trace, int(selected_id), df


def _event_times(vehicle_df: pd.DataFrame) -> np.ndarray:
    if "u_barrier" not in vehicle_df:
        return np.array([])
    active = (
        vehicle_df.loc[vehicle_df["u_barrier"].abs() > BARRIER_EPS, TIME_COL]
        .dropna()
        .sort_values()
        .to_numpy()
    )
    if active.size == 0:
        return active
    # Merge contiguous samples into one event marker.
    jumps = np.r_[True, np.diff(active) > POWER_BIN_S]
    return active[jumps]


def _bin_net_power(t: np.ndarray, power_kw: np.ndarray, bin_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t.size == 0:
        return t, power_kw, power_kw
    bins = np.floor((t - t[0]) / bin_s).astype(int)
    grouped = (
        pd.DataFrame({"bin": bins, "t": t, "power_kw": power_kw})
        .groupby("bin", as_index=False)
        .agg(t=("t", "mean"), power_kw=("power_kw", "mean"))
    )
    p = grouped["power_kw"].to_numpy()
    return grouped["t"].to_numpy(), np.clip(p, 0, None), np.clip(p, None, 0)


def _cumulative_propulsion_energy_kwh(t: np.ndarray, power_kw: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return power_kw
    dt_h = np.diff(t, prepend=t[0]) / 3600.0
    dt_h = np.clip(dt_h, 0, None)
    return np.cumsum(np.clip(power_kw, 0, None) * dt_h)



def _plot_representative_vehicle_row(row: pd.Series) -> None:
    controller = str(row["controller"])
    demand = int(row["demand"])
    run_index = int(row["run_index"])
    vehicle_csv = Path(str(row["resolved_vehicle_csv"]))
    trace, vehicle_id, _ = _selected_vehicle_trace(vehicle_csv, REPRESENTATIVE_VEHICLE_POSITION)

    t = trace[TIME_COL].to_numpy()
    if "velocity_ms" in trace:
        speed = trace["velocity_ms"].to_numpy()
    else:
        speed = trace["velocity_kmh"].to_numpy() / 3.6
    target_speed = np.full_like(speed, TARGET_SPEED_MS, dtype=float)

    gap = trace["gap_meters"].to_numpy()
    if "target_gap_meters" in trace:
        target_gap = trace["target_gap_meters"].to_numpy()
    else:
        target_gap = np.full_like(gap, float(np.nanmean(gap)), dtype=float)

    if "acceleration_ms2" not in trace:
        raise ValueError(f"{vehicle_csv} has no acceleration_ms2 column")
    accel = trace["acceleration_ms2"].to_numpy()

    power = trace["power_kw"].to_numpy() if "power_kw" in trace else np.zeros_like(t)
    power_t, prop, regen = _bin_net_power(t, power, POWER_BIN_S)
    energy_kwh = _cumulative_propulsion_energy_kwh(t, power)
    # Use trace (selected vehicle only) so barrier firings shown are only for this vehicle.
    events = _event_times(trace)

    safe_controller = "traffic_light" if controller == "A" else "barrier_control"
    run_dir = OUTPUT_DIR / (
        f"figure_1_{controller}_{safe_controller}_N{demand}_run{run_index:02d}"
        f"_vehiclepos{REPRESENTATIVE_VEHICLE_POSITION:02d}_id{vehicle_id}"
    )
    def _new_ax(ylabel: str) -> tuple[plt.Figure, plt.Axes]:
        f, a = plt.subplots(1, 1, figsize=(8, 5))
        a.set_ylabel(ylabel)
        a.set_xlabel(r"Time, $t$, [s]")
        a.grid(True, alpha=0.28)
        a.margins(x=0)
        return f, a

    def _finish(fig: plt.Figure, ax: plt.Axes, handles: list, ncol: int) -> None:
        _set_time_ticks(ax)
        fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.10, 1, 1])

    # Speed
    fig, ax = _new_ax(r"Speed, $v$, [m/s]")
    ax.plot(t, speed, color=CONTROLLER_COLORS["A"], label="Actual speed")
    ax.plot(t, target_speed, color=COLORS["target"], linestyle="--", label="Target speed")
    for i, ev in enumerate(events):
        ax.axvline(ev, color=COLORS["barrier"], alpha=0.42, linewidth=1.5,
                   label="Barrier firing" if i == 0 else None)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    h, _ = ax.get_legend_handles_labels()
    _finish(fig, ax, h, ncol=len(h))
    save_figure(fig, "speed", run_dir)
    plt.close(fig)

    # Gap
    fig, ax = _new_ax(r"Headway Gap, $d_\mathrm{actual}$, [m]")
    ax.plot(t, gap, color=COLORS["gap"], label="Actual gap")
    ax.plot(t, target_gap, color=COLORS["target"], linestyle="--", label="Target gap")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    h, _ = ax.get_legend_handles_labels()
    _finish(fig, ax, h, ncol=len(h))
    save_figure(fig, "gap", run_dir)
    plt.close(fig)

    # Acceleration
    fig, ax = _new_ax(r"Acceleration, $a$, [m/s²]")
    ax.plot(t, accel, color=COLORS["acceleration"], label="Acceleration")
    ax.axhline(0.0, color=COLORS["target"], linewidth=1.5, alpha=0.75)
    ax.set_xlim(left=0)
    h, _ = ax.get_legend_handles_labels()
    _finish(fig, ax, h, ncol=len(h))
    save_figure(fig, "acceleration", run_dir)
    plt.close(fig)

    # Power
    power_handles = [
        mpatches.Patch(color=COLORS["propulsion"], alpha=0.45, label=f"Propulsion ({POWER_BIN_S:g} s mean)"),
        mpatches.Patch(color=COLORS["regen"], alpha=0.45, label=f"Regenerative Braking ({POWER_BIN_S:g} s mean)"),
    ]
    fig, ax = _new_ax(r"Battery Power, $P_\mathrm{bat}$, [kW]")
    ax.fill_between(power_t, 0, prop, where=prop >= 0, color=COLORS["propulsion"], alpha=0.45)
    ax.fill_between(power_t, 0, regen, where=regen <= 0, color=COLORS["regen"], alpha=0.45)
    ax.axhline(0.0, color=COLORS["target"], linewidth=1.5, alpha=0.75)
    ax.set_xlim(left=0)
    _finish(fig, ax, power_handles, ncol=2)
    save_figure(fig, "power", run_dir)
    plt.close(fig)

    # Cumulative energy
    fig, ax = _new_ax(r"Energy, $E$, [kWh]")
    ax.plot(t, energy_kwh, color=COLORS["propulsion"], label="Cumulative propulsion energy")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    h, _ = ax.get_legend_handles_labels()
    _finish(fig, ax, h, ncol=1)
    save_figure(fig, "energy", run_dir)
    plt.close(fig)


def plot_representative_vehicles(data: dict[tuple[str, int], pd.DataFrame]) -> None:
    rows = _representative_rows(data)
    if not rows:
        raise RuntimeError("No representative vehicle CSVs found.")
    for row in rows:
        _plot_representative_vehicle_row(row)


def _viol_cumulative_from_csv(csv_path: Path, t_grid: np.ndarray, first: bool = False) -> np.ndarray | None:
    """
    Read csv_path and return a cumulative violation count interpolated onto t_grid.

    Preferred path: throughput CSV has `safety_violation_rate` = total_violations / elapsed_time,
    so cumulative = round(rate * time).  Fallback: look for a binary violation column.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    if TIME_COL not in df.columns:
        return None
    df = df.sort_values(TIME_COL) 
    t_src = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy()

    # --- Preferred: throughput CSV — safety_violation_rate = total_violations / max(t, 1) ---
    if "safety_violation_rate" in df.columns:
        rate = pd.to_numeric(df["safety_violation_rate"], errors="coerce").fillna(0.0).to_numpy()
        c_src = np.maximum.accumulate(np.round(rate * t_src))
        last = float(c_src[-1]) if c_src.size > 0 else 0.0
        return np.interp(t_grid, t_src, c_src, left=0.0, right=last)

    # --- Fallback: binary violation column in vehicle CSV ---
    candidates = list(SAFETY_VIOLATION_COLS)
    for col in df.columns:
        low = col.lower()
        if any(kw in low for kw in ("violat", "unsafe", "collision")) and col not in candidates:
            candidates.append(col)
    viol_col = next(
        (c for c in candidates if c in df.columns and c != "safety_violation_rate"), None
    )

    if viol_col is None:
        if first:
            print(f"  [safety violations] no violation column found in {csv_path.name}.")
            print(f"  Available columns: {list(df.columns)}")
            print(f"  Add the correct name to SAFETY_VIOLATION_COLS at the top of the file.")
        return None

    agg = (
        df.groupby(TIME_COL)[viol_col]
        .apply(lambda x: (pd.to_numeric(x, errors="coerce").fillna(0).abs() > 0).sum())
        .reset_index()
    )
    agg.columns = [TIME_COL, "count"]
    agg = agg.sort_values(TIME_COL)
    t_src2 = agg[TIME_COL].to_numpy()
    c_src2 = agg["count"].to_numpy().cumsum()
    last = float(c_src2[-1]) if c_src2.size > 0 else 0.0
    return np.interp(t_grid, t_src2, c_src2, left=0.0, right=last)


def plot_safety_violations_over_time(data: dict[tuple[str, int], pd.DataFrame]) -> None:
    """
    Two figures:
    1. Boxplot of total safety violations per run (by controller × demand, Barrier before FAS).
    2. Cumulative safety violations vs time, mean ± 95% CI, step style.
    """
    T_GRID = np.arange(0.0, 1801.0, 1.0)
    fig2_dir = OUTPUT_DIR / "figure_2_kpi_comparison"

    # Collect cumulative traces (loop order: demand first, B before A for legend)
    traces_dict: dict[tuple[str, int], list[np.ndarray]] = {}
    first_attempt = True
    for demand in DEMAND_LEVELS:
        for controller in ("B", "A"):
            df = data.get((controller, demand), pd.DataFrame())
            if df.empty:
                continue
            run_traces: list[np.ndarray] = []
            for _, row in df.iterrows():
                arr = None
                for col in ("resolved_throughput_csv", "resolved_vehicle_csv"):
                    csv_path = row.get(col)
                    if isinstance(csv_path, str) and Path(csv_path).is_file():
                        arr = _viol_cumulative_from_csv(Path(csv_path), T_GRID, first=first_attempt)
                        first_attempt = False
                        if arr is not None:
                            break
                if arr is not None:
                    run_traces.append(arr)
            if run_traces:
                traces_dict[(controller, demand)] = run_traces

    # --- 1. Boxplot: total violations per run ---
    plot_order: list[tuple[str, int]] = []
    grouped_counts: list[np.ndarray] = []
    for demand in DEMAND_LEVELS:
        for controller in ("B", "A"):
            key = (controller, demand)
            if key not in traces_dict:
                continue
            counts = np.array([float(arr[-1]) for arr in traces_dict[key]])
            if counts.size > 0:
                plot_order.append(key)
                grouped_counts.append(counts)

    if grouped_counts:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        pos = np.arange(1, len(grouped_counts) + 1)
        xlabels = [f"{'Bar' if c == 'B' else 'FAS'}\nN={n}" for c, n in plot_order]
        bp = ax.boxplot(
            grouped_counts, positions=pos, widths=0.58, patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.6},
            whiskerprops={"color": "#666666", "linewidth": 1.5},
            capprops={"color": "#666666", "linewidth": 1.5},
        )
        for patch, (ctrl, _) in zip(bp["boxes"], plot_order):
            patch.set_facecolor(CONTROLLER_COLORS[ctrl])
            patch.set_alpha(0.34)
            patch.set_edgecolor(CONTROLLER_COLORS[ctrl])
            patch.set_linewidth(1.5)
        for p, (ctrl, _), values in zip(pos, plot_order, grouped_counts):
            mean, err, _ = ci95(pd.Series(values))
            yerr = None if np.isnan(err) else [[err], [err]]
            ax.errorbar([p], [mean], yerr=yerr, fmt="D",
                        color=CONTROLLER_COLORS[ctrl],
                        markeredgecolor="white", markeredgewidth=0.8,
                        capsize=4, markersize=6.2, zorder=5)
            if not np.isnan(mean):
                ax.annotate(
                    f"{mean:.4g}",
                    xy=(p, mean), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=CONTROLLER_COLORS[ctrl], zorder=6,
                )
        ax.set_xlabel(r"Traffic Density, $N$, [veh]")
        ax.set_ylabel(r"Safety Violations, $N_v$, [violations/run]")
        ax.set_xticks(pos)
        ax.set_xticklabels(xlabels)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.28)
        bp_legend = [
            mpatches.Patch(color=CONTROLLER_COLORS["B"], alpha=0.34, label=CONTROLLER_LABELS["B"]),
            mpatches.Patch(color=CONTROLLER_COLORS["A"], alpha=0.34, label=CONTROLLER_LABELS["A"]),
            mlines.Line2D([], [], color="#333333", marker="D", linestyle="None", label="Mean ± 95% CI"),
        ]
        fig.legend(handles=bp_legend, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        save_figure(fig, "safety-violations-per-demand", fig2_dir)
        plt.close(fig)

    # --- 2. Cumulative time series (step style, B before A per demand) ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    legend_handles: list = []
    has_data = False

    for demand in DEMAND_LEVELS:
        for controller in ("B", "A"):
            run_traces = traces_dict.get((controller, demand))
            if not run_traces:
                continue
            mat = np.vstack(run_traces)
            n = mat.shape[0]
            mean = mat.mean(axis=0)
            ci = 1.96 * mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            ci_low = np.maximum(mean - ci, 0.0)
            ci_high = mean + ci
            color = DEMAND_COLORS[demand]
            ls = CONTROLLER_LINESTYLES[controller]
            label = f"{CONTROLLER_LABELS[controller]} N={demand} (n={n})"
            (line,) = ax.step(T_GRID, mean, where="post", color=color, linestyle=ls,
                              linewidth=1.8, label=label)
            ax.fill_between(T_GRID, ci_low, ci_high, step="post", color=color, alpha=0.2)
            legend_handles.append(line)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5,
                "No safety violation column found — see console for column names",
                transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=10,
                wrap=True)

    ax.set_xlabel(r"Time, $t$, [s]")
    ax.set_ylabel(r"Cumulative Safety Violations, $N_v$, [-]")
    ax.set_xlim(0, 1800)
    ax.set_xticks(range(0, 1801, 200))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.28)
    ncol = max(min(len(legend_handles), 3), 1)
    fig.legend(handles=legend_handles, loc="lower center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    save_figure(fig, "safety-violations-over-time", fig2_dir)
    plt.close(fig)


def plot_kpi_comparison(kpi_df: pd.DataFrame) -> None:
    specs = [
        ("energy_kwh_per_km", "Energy Intensity", r"Energy Consumption, $E$, [kWh/km]"),
        ("throughput_veh_s", "Throughput", r"Throughput, $I$, [veh/s]"),
        ("safety_violations_count", "Safety Violations", r"Safety Violations, $N_v$, [violations/run]"),
        ("delay_s", "Delay Time", r"Cumulative Delay Time, $T_\mathrm{delay}$, [s]"),
    ]

    def _kpi_series(controller: str, demand: int, metric: str) -> tuple[float, float]:
        vals = kpi_df.loc[
            (kpi_df["controller"] == controller) & (kpi_df["demand"] == demand), metric
        ]
        mean, err, _ = ci95(vals)
        return mean, 0.0 if np.isnan(err) else err

    handles = [
        mlines.Line2D([], [], color=CONTROLLER_COLORS[c], marker="o", label=CONTROLLER_LABELS[c])
        for c in ("B", "A")
    ]

    # Individual figures in subfolder
    fig2_dir = OUTPUT_DIR / "figure_2_kpi_comparison"
    for metric, _, ylabel in specs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for controller in ("B", "A"):
            means = [_kpi_series(controller, d, metric)[0] for d in DEMAND_LEVELS]
            errs  = [_kpi_series(controller, d, metric)[1] for d in DEMAND_LEVELS]
            ax.errorbar(DEMAND_LEVELS, means, yerr=errs,
                        color=CONTROLLER_COLORS[controller], marker="o",
                        capsize=4, elinewidth=1.5, label=CONTROLLER_LABELS[controller])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Traffic Density, $N$, [veh]")
        ax.set_xticks(DEMAND_LEVELS)
        if metric in {"safety_violations_count", "delay_s"}:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.28)
        h, _ = ax.get_legend_handles_labels()
        fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        save_figure(fig, metric.replace("_", "-"), fig2_dir)
        plt.close(fig)

    # Combined 2×2 overview
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, (metric, _, ylabel) in zip(axes.ravel(), specs):
        for controller in ("B", "A"):
            means = [_kpi_series(controller, d, metric)[0] for d in DEMAND_LEVELS]
            errs  = [_kpi_series(controller, d, metric)[1] for d in DEMAND_LEVELS]
            ax.errorbar(DEMAND_LEVELS, means, yerr=errs,
                        color=CONTROLLER_COLORS[controller], marker="o",
                        capsize=4, elinewidth=1.5, label=CONTROLLER_LABELS[controller])
        ax.set_ylabel(ylabel)
        ax.set_xticks(DEMAND_LEVELS)
        if metric in {"safety_violations_count", "delay_s"}:
            ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.28)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Traffic Density, $N$, [veh]")
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    save_figure(fig, "figure_2_kpi_comparison")
    plt.close(fig)


def plot_throughput_per_demand(kpi_df: pd.DataFrame) -> None:
    """
    Throughput boxplot grouped by demand level, both controllers side-by-side.
    A black dotted line spanning each demand group marks the theoretical maximum
    throughput N × 2 / 281.78 (veh/s).
    """
    fig2_dir = OUTPUT_DIR / "figure_2_kpi_comparison"

    plot_order: list[tuple[str, int]] = []
    grouped: list[np.ndarray] = []
    for demand in DEMAND_LEVELS:
        for controller in ("B", "A"):
            vals = pd.to_numeric(
                kpi_df.loc[
                    (kpi_df["controller"] == controller) & (kpi_df["demand"] == demand),
                    "throughput_veh_s",
                ],
                errors="coerce",
            ).dropna().to_numpy()
            if vals.size > 0:
                plot_order.append((controller, demand))
                grouped.append(vals)

    if not grouped:
        print("Warning: no throughput data for per-demand boxplot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    pos = np.arange(1, len(grouped) + 1)
    xlabels = [f"{'Bar' if c == 'B' else 'FAS'}\nN={n}" for c, n in plot_order]

    bp = ax.boxplot(
        grouped, positions=pos, widths=0.58, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.6},
        whiskerprops={"color": "#666666", "linewidth": 1.5},
        capprops={"color": "#666666", "linewidth": 1.5},
    )
    for patch, (ctrl, _) in zip(bp["boxes"], plot_order):
        patch.set_facecolor(CONTROLLER_COLORS[ctrl])
        patch.set_alpha(0.34)
        patch.set_edgecolor(CONTROLLER_COLORS[ctrl])
        patch.set_linewidth(1.5)

    for p, (ctrl, _), vals in zip(pos, plot_order, grouped):
        mean, err, _ = ci95(pd.Series(vals))
        yerr = None if np.isnan(err) else [[err], [err]]
        ax.errorbar([p], [mean], yerr=yerr, fmt="D",
                    color=CONTROLLER_COLORS[ctrl],
                    markeredgecolor="white", markeredgewidth=0.8,
                    capsize=4, markersize=6.2, zorder=5)
        if not np.isnan(mean):
            q3 = np.percentile(vals, 75)
            iqr = q3 - np.percentile(vals, 25)
            whisker_top = min(float(np.max(vals)), q3 + 1.5 * iqr)
            ax.annotate(
                f"{mean:.4g}",
                xy=(p, whisker_top), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=CONTROLLER_COLORS[ctrl], zorder=6,
            )

    # Theoretical max N × 2 / 281.78, one spanning line per demand group
    seen_demands: set[int] = set()
    for j, (_, demand) in enumerate(plot_order):
        if demand in seen_demands:
            continue
        seen_demands.add(demand)
        theo_max = demand * 2 / 281.78
        demand_pos = [pos[k] for k in range(len(plot_order)) if plot_order[k][1] == demand]
        x0 = min(demand_pos) - 0.36
        x1 = max(demand_pos) + 0.36
        ax.hlines(theo_max, x0, x1, colors="black", linestyles=":", linewidth=1.8, zorder=4)
        ax.annotate(
            f"{theo_max:.4g}",
            xy=((x0 + x1) / 2, theo_max), xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color="black", zorder=5,
        )

    ax.set_xlabel(r"Controller and Traffic Density, $N$, [veh]")
    ax.set_ylabel(r"Throughput, $I$, [veh/s]")
    ax.set_xticks(pos)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.28)

    legend_handles = [
        mpatches.Patch(color=CONTROLLER_COLORS["B"], alpha=0.34, label=CONTROLLER_LABELS["B"]),
        mpatches.Patch(color=CONTROLLER_COLORS["A"], alpha=0.34, label=CONTROLLER_LABELS["A"]),
        mlines.Line2D([], [], color="#333333", marker="D", linestyle="None", label="Mean ± 95% CI"),
        mlines.Line2D([], [], color="black", linestyle=":", linewidth=1.8, label="Theoretical max (N×2/281.78)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_figure(fig, "throughput-per-demand", fig2_dir)
    plt.close(fig)


def plot_kpi_distributions(kpi_df: pd.DataFrame) -> None:
    specs = [
        ("energy_kwh_per_km", "Energy Intensity", r"Energy Consumption, $E$, [kWh/km]"),
        ("safety_violations_count", "Safety Violations", r"Safety Violations, $N_v$, [violations/run]"),
        ("delay_s", "Delay Time", r"Cumulative Delay Time, $T_\mathrm{delay}$, [s]"),
    ]
    order = [(c, n) for n in DEMAND_LEVELS for c in ("B", "A")]

    legend_handles = [
        mpatches.Patch(color=CONTROLLER_COLORS["A"], alpha=0.34, label=CONTROLLER_LABELS["A"]),
        mpatches.Patch(color=CONTROLLER_COLORS["B"], alpha=0.34, label=CONTROLLER_LABELS["B"]),
        mlines.Line2D([], [], color="#333333", marker="D", linestyle="None", label="Mean ± 95% CI"),
    ]

    def _draw_dist_ax(ax: plt.Axes, metric: str, ylabel: str) -> None:
        # Build only non-empty groups so missing demand/controller combos don't break boxplot
        plot_order: list[tuple[str, int]] = []
        grouped: list[np.ndarray] = []
        for c, n in order:
            vals = pd.to_numeric(
                kpi_df.loc[(kpi_df["controller"] == c) & (kpi_df["demand"] == n), metric],
                errors="coerce",
            ).dropna().to_numpy()
            if vals.size > 0:
                plot_order.append((c, n))
                grouped.append(vals)

        if not grouped:
            return

        pos = np.arange(1, len(grouped) + 1)
        xlabels = [f"{c}-{n}" for c, n in plot_order]

        bp = ax.boxplot(
            grouped, positions=pos, widths=0.58, patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.6},
            whiskerprops={"color": "#666666", "linewidth": 1.5},
            capprops={"color": "#666666", "linewidth": 1.5},
        )
        for patch, (controller, _) in zip(bp["boxes"], plot_order):
            patch.set_facecolor(CONTROLLER_COLORS[controller])
            patch.set_alpha(0.34)
            patch.set_edgecolor(CONTROLLER_COLORS[controller])
            patch.set_linewidth(1.5)
        for p, (controller, _), values in zip(pos, plot_order, grouped):
            mean, err, _ = ci95(pd.Series(values))
            yerr = None if np.isnan(err) else [[err], [err]]
            ax.errorbar([p], [mean], yerr=yerr, fmt="D",
                        color=CONTROLLER_COLORS[controller],
                        markeredgecolor="white", markeredgewidth=0.8,
                        capsize=4, markersize=6.2, zorder=5)
            if not np.isnan(mean):
                ax.annotate(
                    f"{mean:.4g}",
                    xy=(p, mean), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color=CONTROLLER_COLORS[controller], zorder=6,
                )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Traffic Density, $N$, [veh]")
        ax.set_xticks(pos)
        ax.set_xticklabels(xlabels)
        ax.grid(True, axis="y", alpha=0.28)

    # Individual figures in subfolder
    fig3_dir = OUTPUT_DIR / "figure_3_kpi_distributions"
    for metric, _, ylabel in specs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        _draw_dist_ax(ax, metric, ylabel)
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        save_figure(fig, metric.replace("_", "-"), fig3_dir)
        plt.close(fig)

    # Combined side-by-side overview
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    for ax, (metric, _, ylabel) in zip(axes, specs):
        _draw_dist_ax(ax, metric, ylabel)
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    save_figure(fig, "figure_3_kpi_distributions")
    plt.close(fig)


# ---------- Mean vehicle behaviour across all runs --------------------------

T_MEAN_GRID = np.arange(0.0, 1801.0, 1.0)


def _load_all_vehicle_traces(
    data: dict[tuple[str, int], pd.DataFrame],
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """
    For every (controller, demand) pair, extract the representative vehicle
    position from each run, interpolate each metric to T_MEAN_GRID, and
    return stacked matrices [n_runs × n_time].  Points past the run's end
    are NaN (except cumulative energy, which is forward-filled).
    """
    result: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for (controller, demand), df in data.items():
        if df.empty:
            continue
        with_csv = df[
            df["resolved_vehicle_csv"].apply(
                lambda p: isinstance(p, str) and Path(p).is_file()
            )
        ].copy()
        if with_csv.empty:
            continue

        stacks: dict[str, list[np.ndarray]] = {k: [] for k in ("speed", "gap", "target_gap", "accel", "power", "energy")}

        for _, row in with_csv.iterrows():
            try:
                trace, _, _ = _selected_vehicle_trace(
                    Path(str(row["resolved_vehicle_csv"])), REPRESENTATIVE_VEHICLE_POSITION
                )
            except Exception as exc:
                print(f"Warning: skipping trace: {exc}")
                continue
            t = trace[TIME_COL].to_numpy()
            if t.size < 2:
                continue

            speed = trace["velocity_ms"].to_numpy() if "velocity_ms" in trace else trace["velocity_kmh"].to_numpy() / 3.6
            gap = trace["gap_meters"].to_numpy()
            target_gap_col = trace["target_gap_meters"].to_numpy() if "target_gap_meters" in trace else np.full_like(gap, float(np.nanmean(gap)))
            accel = trace["acceleration_ms2"].to_numpy() if "acceleration_ms2" in trace else np.zeros_like(t)
            power_kw = trace["power_kw"].to_numpy() if "power_kw" in trace else np.zeros_like(t)
            energy = _cumulative_propulsion_energy_kwh(t, power_kw)

            def _interp(y: np.ndarray, forward: bool = False) -> np.ndarray:
                out = np.interp(T_MEAN_GRID, t, y, right=float(y[-1]) if forward else np.nan)
                if not forward:
                    out[T_MEAN_GRID > t[-1]] = np.nan
                return out

            stacks["speed"].append(_interp(speed))
            stacks["gap"].append(_interp(gap))
            stacks["target_gap"].append(_interp(target_gap_col))
            stacks["accel"].append(_interp(accel))
            stacks["power"].append(_interp(power_kw))
            stacks["energy"].append(_interp(energy, forward=True))

        if not any(stacks.values()):
            continue
        result[(controller, demand)] = {
            "t": T_MEAN_GRID,
            **{k: np.vstack(v) for k, v in stacks.items() if v},
        }
    return result


def plot_mean_vehicle_behaviour(data: dict[tuple[str, int], pd.DataFrame]) -> None:
    traces = _load_all_vehicle_traces(data)
    if not traces:
        print("Warning: no vehicle traces loaded for mean behaviour plots.")
        return

    mean_dir = OUTPUT_DIR / "figure_1_mean_across_runs"

    metrics = [
        ("speed",  r"Speed, $v$, [m/s]",                "speed"),
        ("gap",    r"Mean Headway Gap, $d_\mathrm{actual}$, [m]", "gap"),
        ("accel",  r"Acceleration, $a$, [m/s²]",         "acceleration"),
        ("power",  r"Battery Power, $P_\mathrm{bat}$, [kW]",                  "power"),
        ("energy", r"Energy, $E$, [kWh]",                "energy"),
    ]

    for controller in ("A", "B"):
        ctrl_dir = mean_dir / f"controller_{controller}"
        for metric_key, ylabel, stem in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.set_xlabel(r"Time, $t$, [s]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.28)
            ax.margins(x=0)

            if metric_key == "speed":
                ax.axhline(TARGET_SPEED_MS, color=COLORS["target"], linestyle="--",
                           linewidth=1.5, alpha=0.75)

            legend_handles: list = []
            for demand in DEMAND_LEVELS:
                key = (controller, demand)
                if key not in traces or metric_key not in traces[key]:
                    continue
                mat = traces[key][metric_key]
                t = traces[key]["t"]
                n = mat.shape[0]
                mean = np.nanmean(mat, axis=0)
                ci = 1.96 * np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
                ls = DEMAND_LINESTYLES[demand]
                color = DEMAND_COLORS[demand]
                label = f"N={demand} (n={n})"
                (line,) = ax.plot(t, mean, color=color, linestyle=ls, linewidth=1.8, label=label)
                ax.fill_between(t, mean - ci, mean + ci, color=color, alpha=0.2)
                legend_handles.append(line)
                if metric_key == "gap" and "target_gap" in traces[key]:
                    tg_mean = np.nanmean(traces[key]["target_gap"], axis=0)
                    ax.plot(t, tg_mean, color="black", linestyle=":", linewidth=1.5, alpha=0.75)

            if metric_key == "speed":
                legend_handles.insert(
                    0,
                    mlines.Line2D([], [], color=COLORS["target"], linestyle="--", label="Target speed"),
                )
            if metric_key == "gap":
                legend_handles.insert(
                    0,
                    mlines.Line2D([], [], color="black", linestyle=":", linewidth=1.5, label="Target gap"),
                )

            ax.set_xlim(left=0)
            if metric_key in ("speed", "gap", "energy"):
                ax.set_ylim(bottom=0)
            _set_time_ticks(ax)
            ncol = max(min(len(legend_handles), 4), 1)
            fig.legend(handles=legend_handles, loc="lower center", ncol=ncol,
                       frameon=False, bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=[0, 0.10, 1, 1])
            save_figure(fig, stem, ctrl_dir)
            plt.close(fig)


# ---------- Barrier firing stats --------------------------------------------

def _barrier_firings_cumulative(vehicle_csv: Path, t_grid: np.ndarray) -> np.ndarray | None:
    """
    Across all vehicles in vehicle_csv, find barrier-onset events (rising edge of
    u_barrier) and return the cumulative firing count interpolated onto t_grid.
    """
    try:
        df = pd.read_csv(vehicle_csv)
    except Exception:
        return None
    df.columns = df.columns.str.strip()
    if "u_barrier" not in df.columns or TIME_COL not in df.columns or "vehicle_id" not in df.columns:
        return None

    event_times: list[float] = []
    for _, grp in df.groupby("vehicle_id"):
        t_active = (
            grp.loc[grp["u_barrier"].abs() > BARRIER_EPS, TIME_COL]
            .dropna()
            .sort_values()
            .to_numpy()
        )
        if t_active.size == 0:
            continue
        onsets = t_active[np.r_[True, np.diff(t_active) > POWER_BIN_S]]
        event_times.extend(onsets.tolist())

    if not event_times:
        return np.zeros(len(t_grid), dtype=float)

    t_events = np.sort(np.array(event_times))
    return np.searchsorted(t_events, t_grid, side="right").astype(float)


def _barrier_firing_matrix(df: pd.DataFrame, t_grid: np.ndarray) -> np.ndarray | None:
    with_csv = df[
        df["resolved_vehicle_csv"].apply(lambda p: isinstance(p, str) and Path(p).is_file())
    ]
    if with_csv.empty:
        return None

    run_traces: list[np.ndarray] = []
    for _, row in with_csv.iterrows():
        arr = _barrier_firings_cumulative(Path(str(row["resolved_vehicle_csv"])), t_grid)
        if arr is not None:
            run_traces.append(arr)
    if not run_traces:
        return None
    return np.vstack(run_traces)


def plot_barrier_stats(data: dict[tuple[str, int], pd.DataFrame]) -> None:
    """
    Figure 4 — barrier firing statistics for controller B.

    Saves to figure_4_barrier_stats/:
      firings-per-demand.png   — mean ± 95% CI total firings per run, by demand level
      firings-over-time-N{d}.png — mean ± 95% CI cumulative firings vs time
      firings-over-time-all-demands.png — all demand batches on one mean ± 95% CI plot
    """
    fig4_dir = OUTPUT_DIR / "figure_4_barrier_stats"
    T_GRID = np.arange(0.0, 1801.0, 1.0)
    BARRIER_DEMANDS = [d for d in DEMAND_LEVELS if ("B", d) in {k: None for k in data}]

    # ---- Boxplot + mean diamond: total firings per demand ----------------------
    all_counts: list[list[float]] = []
    demands_with_data: list[int] = []

    for demand in BARRIER_DEMANDS:
        df = data.get(("B", demand), pd.DataFrame())
        if df.empty:
            continue
        counts: list[float] = []
        for _, row in df.iterrows():
            csv_path = row.get("resolved_vehicle_csv")
            if not isinstance(csv_path, str) or not Path(csv_path).is_file():
                continue
            arr = _barrier_firings_cumulative(Path(csv_path), T_GRID)
            if arr is not None:
                counts.append(float(arr[-1]))
        if not counts:
            continue
        all_counts.append(counts)
        demands_with_data.append(demand)

    if demands_with_data:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        pos = np.arange(1, len(demands_with_data) + 1)
        xlabels = [f"N={d}" for d in demands_with_data]

        bp = ax.boxplot(
            all_counts, positions=pos, widths=0.58, patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.6},
            whiskerprops={"color": "#666666", "linewidth": 1.5},
            capprops={"color": "#666666", "linewidth": 1.5},
        )
        for patch, demand in zip(bp["boxes"], demands_with_data):
            color = DEMAND_COLORS[demand]
            patch.set_facecolor(color)
            patch.set_alpha(0.34)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)
        for p, demand, vals in zip(pos, demands_with_data, all_counts):
            color = DEMAND_COLORS[demand]
            mean, err, _ = ci95(pd.Series(vals))
            yerr = None if np.isnan(err) else [[err], [err]]
            ax.errorbar([p], [mean], yerr=yerr, fmt="D",
                        color=color, markeredgecolor="white", markeredgewidth=0.8,
                        capsize=4, markersize=6.2, zorder=5)
            if not np.isnan(mean):
                q3 = np.percentile(vals, 75)
                iqr = q3 - np.percentile(vals, 25)
                whisker_top = min(float(np.max(vals)), q3 + 1.5 * iqr)
                ax.annotate(
                    f"{mean:.0f}",
                    xy=(p, whisker_top), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                    color="black", zorder=6,
                )

        ax.set_xlabel(r"Traffic Density, $N$, [veh]")
        ax.set_ylabel(r"Barrier Activations, $N_b$, [-]")
        ax.set_xticks(pos)
        ax.set_xticklabels(xlabels)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", alpha=0.28)

        legend_handles = [
            mpatches.Patch(facecolor=DEMAND_COLORS[d], alpha=0.34, edgecolor=DEMAND_COLORS[d],
                           linewidth=1.5, label=f"N={d}")
            for d in demands_with_data
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        save_figure(fig, "firings-per-demand", fig4_dir)
        plt.close(fig)
    else:
        print("Warning: no barrier firing data found for controller B.")

    demand_mats: dict[int, np.ndarray] = {}
    for demand in BARRIER_DEMANDS:
        df = data.get(("B", demand), pd.DataFrame())
        if df.empty:
            continue
        mat = _barrier_firing_matrix(df, T_GRID)
        if mat is not None:
            demand_mats[demand] = mat

    if demand_mats:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        legend_handles: list = []
        for demand in DEMAND_LEVELS:
            mat = demand_mats.get(demand)
            if mat is None:
                continue
            n = mat.shape[0]
            mean = mat.mean(axis=0)
            ci = 1.96 * mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            ci_low = np.maximum(mean - ci, 0.0)
            ci_high = mean + ci
            color = DEMAND_COLORS[demand]
            (line,) = ax.step(T_GRID, mean, where="post", color=color, linewidth=1.8,
                              linestyle=DEMAND_LINESTYLES[demand], label=f"N={demand} (n={n})")
            ax.fill_between(T_GRID, ci_low, ci_high, step="post", color=color, alpha=0.2)
            legend_handles.append(line)

        ax.set_xlabel(r"Time, $t$, [s]")
        ax.set_ylabel(r"Cumulative Barrier Activations, $N_b$, [-]")
        ax.set_xlim(0, 1800)
        ax.set_xticks(range(0, 1801, 200))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.28)
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=max(min(len(legend_handles), 4), 1), frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        save_figure(fig, "firings-over-time-all-demands", fig4_dir)
        plt.close(fig)

    # ---- Cumulative firings over time: mean ± 95% CI across runs ---------------
    for demand in BARRIER_DEMANDS:
        mat = demand_mats.get(demand)
        if mat is None:
            continue

        n = mat.shape[0]
        mean = mat.mean(axis=0)
        ci = 1.96 * mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        ci_low = np.maximum(mean - ci, 0.0)
        ci_high = mean + ci

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        color = DEMAND_COLORS[demand]
        (line,) = ax.step(T_GRID, mean, where="post", color=color, linewidth=1.8,
                          label=f"Mean (n={n})")
        ax.fill_between(T_GRID, ci_low, ci_high, step="post", color=color, alpha=0.2,
                        label="95% CI")

        ax.set_xlabel(r"Time, $t$, [s]")
        ax.set_ylabel(r"Cumulative Barrier Activations, $N_b$, [-]")
        ax.set_xlim(0, 1800)
        ax.set_xticks(range(0, 1801, 200))
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.28)
        ci_patch = mpatches.Patch(color=color, alpha=0.2, label="95% CI")
        fig.legend(handles=[line, ci_patch], loc="lower center",
                   ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        save_figure(fig, f"firings-over-time-N{demand}", fig4_dir)
        plt.close(fig)


# ---------- KPI summary table with percentage differences -------------------

KPI_SPECS = [
    ("energy_kwh_per_km",       "Energy (kWh/km)"),
    ("throughput_veh_s",        "Throughput (veh/s)"),
    ("safety_violations_count", "Safety Violations (violations/run)"),
    ("delay_s",                 "Delay (s/run)"),
]


def export_kpi_summary_table(kpi_df: pd.DataFrame) -> None:
    """
    Write a wide-format CSV (and Excel if openpyxl is available) with:
      - mean and 95% CI for every KPI × controller × demand
      - absolute difference  (B − A)
      - percentage difference (B − A) / |A| × 100  (positive = B is higher)

    Output: OUTPUT_DIR / kpi_summary_with_pct_diff.csv  (+ .xlsx)
    """
    records: list[dict] = []
    for demand in DEMAND_LEVELS:
        for metric, metric_label in KPI_SPECS:
            row: dict = {"Demand (N)": demand, "KPI": metric_label}
            means: dict[str, float] = {}
            for ctrl in ("A", "B"):
                vals = kpi_df.loc[
                    (kpi_df["controller"] == ctrl) & (kpi_df["demand"] == demand), metric
                ]
                mean, half_ci, n = ci95(vals)
                ctrl_label = "FAS Controller" if ctrl == "A" else "Barrier Controller"
                row[f"{ctrl_label} — n"] = n
                row[f"{ctrl_label} — mean"] = round(mean, 6) if not np.isnan(mean) else np.nan
                row[f"{ctrl_label} — 95% CI (±)"] = round(half_ci, 6) if not np.isnan(half_ci) else np.nan
                means[ctrl] = mean

            a, b = means.get("A", np.nan), means.get("B", np.nan)
            if not np.isnan(a) and not np.isnan(b):
                row["B − A (absolute)"] = round(b - a, 6)
                denom = abs(a) if abs(a) > 1e-12 else np.nan
                row["(B − A) / |A| × 100  (%)"] = round((b - a) / denom * 100, 2) if not np.isnan(denom) else np.nan
            else:
                row["B − A (absolute)"] = np.nan
                row["(B − A) / |A| × 100  (%)"] = np.nan

            records.append(row)

    out_df = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "kpi_summary_with_pct_diff.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    try:
        import importlib.util
        if importlib.util.find_spec("openpyxl") is None:
            raise ImportError
        xlsx_path = OUTPUT_DIR / "kpi_summary_with_pct_diff.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="KPI Summary")
            ws = writer.sheets["KPI Summary"]
            # Auto-fit column widths
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 42)
        print(f"Saved {xlsx_path}")
    except ImportError:
        print("openpyxl not available — skipping Excel export (CSV was saved).")

    # Pretty-print to console
    print("\n── KPI Summary Table ──────────────────────────────────────────────")
    print(out_df.to_string(index=False))
    print("────────────────────────────────────────────────────────────────────\n")


# ---------- Export mean vehicle behaviour time-series -----------------------

def export_mean_vehicle_behaviour_csv(data: dict[tuple[str, int], pd.DataFrame]) -> None:
    """
    Save mean ± 95% CI of per-vehicle time-series metrics to a long-format CSV.
    Columns: time_s, controller, demand, n_runs, speed_mean, speed_ci95,
             gap_mean, gap_ci95, accel_mean, accel_ci95,
             power_mean, power_ci95, energy_mean, energy_ci95
    """
    traces = _load_all_vehicle_traces(data)
    if not traces:
        print("Warning: no vehicle traces to export.")
        return

    metric_keys = ["speed", "gap", "accel", "power", "energy"]
    rows: list[dict] = []
    for (controller, demand), entry in traces.items():
        t = entry["t"]
        n_runs_list: list[int] = []
        metric_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for mk in metric_keys:
            if mk not in entry:
                continue
            mat = entry[mk]
            n = mat.shape[0]
            n_runs_list.append(n)
            mean = np.nanmean(mat, axis=0)
            ci = (1.96 * np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
            metric_arrays[mk] = (mean, ci)

        n_runs = n_runs_list[0] if n_runs_list else 0
        for i, t_val in enumerate(t):
            row: dict = {"time_s": t_val, "controller": controller, "demand": demand, "n_runs": n_runs}
            for mk, (mean_arr, ci_arr) in metric_arrays.items():
                row[f"{mk}_mean"] = round(float(mean_arr[i]), 6)
                row[f"{mk}_ci95"] = round(float(ci_arr[i]), 6)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "mean_vehicle_behaviour_timeseries.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


# ---------- Per-vehicle-type energy extraction ------------------------------

VEHICLE_TYPE_LABELS = {
    "audi": "Audi (e-tron)",
    "tesla": "Tesla (Model 3)",
}


def _brand_from_blueprint(bp: str) -> str | None:
    bp_lower = str(bp).lower()
    for key in VEHICLE_TYPE_LABELS:
        if key in bp_lower:
            return key
    return None


def export_vehicle_type_energy(data: dict[tuple[str, int], pd.DataFrame]) -> None:
    """
    For every (controller, demand) batch, load each vehicle CSV, compute
    energy intensity (kWh/km) per individual vehicle by integrating power
    for energy and velocity for distance, then aggregate by brand (Tesla / Audi).

    Outputs:
      - vehicle_type_energy_per_run.csv  — one row per (controller, demand, run, brand, vehicle_id)
      - vehicle_type_energy_summary.csv  — mean ± 95% CI per (controller, demand, brand)
                                           plus an "All N" aggregate row per (controller, brand)
    Prints a compact summary table to stdout.
    """
    per_vehicle_rows: list[dict] = []

    for (controller, demand), df in data.items():
        if df.empty:
            continue
        with_csv = df[
            df["resolved_vehicle_csv"].apply(lambda p: isinstance(p, str) and Path(p).is_file())
        ]
        for _, row in with_csv.iterrows():
            run_index = int(row["run_index"])
            vdf = pd.read_csv(str(row["resolved_vehicle_csv"]))
            vdf.columns = vdf.columns.str.strip()
            if "blueprint_id" not in vdf.columns or "power_kw" not in vdf.columns:
                continue

            speed_col = "velocity_ms" if "velocity_ms" in vdf.columns else None
            if speed_col is None and "velocity_kmh" in vdf.columns:
                vdf["velocity_ms"] = vdf["velocity_kmh"] / 3.6
                speed_col = "velocity_ms"

            for vid, grp in vdf.groupby("vehicle_id"):
                brand = _brand_from_blueprint(grp["blueprint_id"].iloc[0])
                if brand is None:
                    continue
                grp = grp.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL])
                t = grp[TIME_COL].to_numpy()
                if t.size < 2:
                    continue
                power = grp["power_kw"].to_numpy()
                energy_kwh = float(_cumulative_propulsion_energy_kwh(t, power)[-1])

                # distance by trapezoidal integration of speed (m → km)
                speed_ms = grp[speed_col].to_numpy() if speed_col else np.zeros_like(t)
                distance_km = float(np.trapz(np.clip(speed_ms, 0, None), t)) / 1000.0

                energy_kwh_per_km = energy_kwh / distance_km if distance_km > 0 else np.nan

                per_vehicle_rows.append({
                    "controller": controller,
                    "controller_label": CONTROLLER_LABELS[controller],
                    "demand": demand,
                    "run_index": run_index,
                    "vehicle_id": vid,
                    "brand": brand,
                    "brand_label": VEHICLE_TYPE_LABELS[brand],
                    "energy_kwh": round(energy_kwh, 6),
                    "distance_km": round(distance_km, 6),
                    "energy_kwh_per_km": round(energy_kwh_per_km, 6) if not np.isnan(energy_kwh_per_km) else np.nan,
                })

    if not per_vehicle_rows:
        print("Warning: no Tesla/Audi vehicles found in vehicle CSVs.")
        return

    pv_df = pd.DataFrame(per_vehicle_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pv_csv = OUTPUT_DIR / "vehicle_type_energy_per_run.csv"
    pv_df.to_csv(pv_csv, index=False)
    print(f"Saved {pv_csv}")

    # Summary: mean ± 95% CI per (controller, demand, brand)
    summary_rows: list[dict] = []
    for (ctrl, demand, brand), grp in pv_df.groupby(["controller", "demand", "brand"]):
        mean, half_ci, n = ci95(grp["energy_kwh_per_km"].dropna())
        summary_rows.append({
            "controller": ctrl,
            "controller_label": CONTROLLER_LABELS[ctrl],
            "demand": demand,
            "brand": brand,
            "brand_label": VEHICLE_TYPE_LABELS[brand],
            "n_vehicles": n,
            "energy_kwh_per_km_mean": round(mean, 6) if not np.isnan(mean) else np.nan,
            "energy_kwh_per_km_ci95": round(half_ci, 6) if not np.isnan(half_ci) else np.nan,
        })

    # "All N" aggregate: pool across all demand levels per (controller, brand)
    for (ctrl, brand), grp in pv_df.groupby(["controller", "brand"]):
        mean, half_ci, n = ci95(grp["energy_kwh_per_km"].dropna())
        summary_rows.append({
            "controller": ctrl,
            "controller_label": CONTROLLER_LABELS[ctrl],
            "demand": "All N",
            "brand": brand,
            "brand_label": VEHICLE_TYPE_LABELS[brand],
            "n_vehicles": n,
            "energy_kwh_per_km_mean": round(mean, 6) if not np.isnan(mean) else np.nan,
            "energy_kwh_per_km_ci95": round(half_ci, 6) if not np.isnan(half_ci) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["controller", "brand", "demand"])
    sum_csv = OUTPUT_DIR / "vehicle_type_energy_summary.csv"
    summary_df.to_csv(sum_csv, index=False)
    print(f"Saved {sum_csv}")

    print("\n── Vehicle-Type Energy Summary (kWh/km per vehicle) ────────────────")
    print(summary_df[["controller_label", "demand", "brand_label", "n_vehicles",
                       "energy_kwh_per_km_mean", "energy_kwh_per_km_ci95"]].to_string(index=False))
    print("─────────────────────────────────────────────────────────────────────\n")


# ---------- Main ------------------------------------------------------------

PLOTS = {
    "representative": lambda data, kpi_df: plot_representative_vehicles(data),
    "mean_behaviour": lambda data, kpi_df: plot_mean_vehicle_behaviour(data),
    "safety": lambda data, kpi_df: plot_safety_violations_over_time(data),
    "kpi_comparison": lambda data, kpi_df: plot_kpi_comparison(kpi_df),
    "throughput_per_demand": lambda data, kpi_df: plot_throughput_per_demand(kpi_df),
    "kpi_distributions": lambda data, kpi_df: plot_kpi_distributions(kpi_df),
    "barrier": lambda data, kpi_df: plot_barrier_stats(data),
    "vehicle_type_energy": lambda data, kpi_df: export_vehicle_type_energy(data),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        choices=list(PLOTS),
        default=None,
        help="Run only the specified plot (default: run all).",
    )
    args = parser.parse_args()

    apply_style()
    data = load_all_summaries()
    kpi_df = build_kpi_table(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kpi_path = OUTPUT_DIR / "publication_kpi_run_values.csv"
    kpi_df.to_csv(kpi_path, index=False)
    print(f"Saved {kpi_path}")

    print("\nRun counts used for KPI plots:")
    counts = (
        kpi_df.groupby(["controller_label", "demand"])
        .size()
        .rename("runs")
        .reset_index()
        .sort_values(["demand", "controller_label"])
    )
    print(counts.to_string(index=False))

    if (counts["runs"] < 30).any():
        print("\nWarning: at least one controller-demand group has fewer than 30 loaded runs.")
        print("Edit BATCH_DIRS if additional final batch folders are available.")

    if args.plot:
        PLOTS[args.plot](data, kpi_df)
    else:
        export_kpi_summary_table(kpi_df)
        export_mean_vehicle_behaviour_csv(data)
        export_vehicle_type_energy(data)
        plot_representative_vehicles(data)
        plot_mean_vehicle_behaviour(data)
        plot_safety_violations_over_time(data)
        plot_kpi_comparison(kpi_df)
        plot_throughput_per_demand(kpi_df)
        plot_kpi_distributions(kpi_df)
        plot_barrier_stats(data)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract and plot CARLA batch simulation results.

The script looks under:
  simulation_results/BarrierControlResults/Batch_30runs_N=*
  simulation_results/TrafficlightResults/Batch_30runs_N=*

For each batch it loads the 30 run folders, computes the mean and 95%
confidence interval for the main KPIs, and writes:
  - final_metric_summary.csv
  - one curve-summary CSV per plotted metric
  - PNG plots with all six batch/controller series
  - separate final-value boxplots with mean +/- 95% CI

Energy is plotted against total distance. Safety violations, delay, and
throughput are plotted against simulation time.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


TIME_COL = "time_seconds"
DISTANCE_COL = "total_distance_km"
ENERGY_COL = "Energy_kWh"
THROUGHPUT_COL = "throughput"
DELAY_COL = "total_delay_s"
SAFETY_RATE_COL = "safety_violation_rate"
SAFETY_COUNT_COL = "safety_violations"
VEHICLE_ID_COL = "vehicle_id"
VELOCITY_MS_COL = "velocity_ms"
VELOCITY_KMH_COL = "velocity_kmh"
POWER_KW_COL = "power_kw"
ENERGY_PER_KM_COL = "energy_kWh_per_km_curve"
SELECTED_DISTANCE_COL = "selected_vehicle_distance_km"
SELECTED_ENERGY_COL = "selected_vehicle_energy_kWh"
BLUEPRINT_ID_COL = "blueprint_id"
VEHICLE_TYPE_ENERGY_PER_KM_COL = "vehicle_type_energy_kWh_per_km"
VEHICLE_TYPE_SPEED_COL = "vehicle_type_speed_kmh"
FLEET_SPEED_COL = "avg_speed_kmh"

TESLA_BLUEPRINT = "vehicle.tesla.model3"
AUDI_BLUEPRINT = "vehicle.audi.etron"
VEHICLE_TYPES = [
    (TESLA_BLUEPRINT, "Tesla_Model3", "Tesla Model 3"),
    (AUDI_BLUEPRINT, "Audi_Etron", "Audi e-tron"),
    (None, "All_Vehicles", "All Vehicles (Fleet Mean)"),
]

BATCH_NAME_RE = re.compile(r"Batch_(?:.*?)[_]?[Nn]=(?P<n>\d+)(?P<suffix>.*)", re.IGNORECASE)

ALLOWED_BATCH_DIRS: dict[str, set[str]] = {
    "BarrierControlResults": {"Batch_barrier_N=61", "Batch_barrier_N=71","Batch_barrier_N=81"},
    "TrafficlightResults": {"Batch_FAS_N=61", "Batch_FAS_N=71", "Batch_FAS_N=81"},
}
RUN_DIR_RE = re.compile(r"run_(?P<run>\d+)$", re.IGNORECASE)

CONTROLLER_DIRS = {
    "Barrier Control": "BarrierControlResults",
    "Traffic Light": "TrafficlightResults",
}

CONTROLLER_LINESTYLES = {
    "Barrier Control": "-",
    "Traffic Light": "--",
}

SERIES_COLORS = {
    ("Barrier Control", 57): "#005AB5",
    ("Barrier Control", 67): "#009E73",
    ("Barrier Control", 69): "#009E73",
    ("Barrier Control", 77): "#7A3E9D",
    ("Barrier Control", 81): "#56B4E9",
    ("Traffic Light", 57): "#F0A202",
    ("Traffic Light", 67): "#D41159",
    ("Traffic Light", 69): "#D41159",
    ("Traffic Light", 77): "#111111",
    ("Traffic Light", 81): "#CC79A7",
}

FALLBACK_COLORS = [
    "#005AB5",
    "#F0A202",
    "#009E73",
    "#D41159",
    "#7A3E9D",
    "#111111",
    "#56B4E9",
]


@dataclass(frozen=True)
class BatchInfo:
    controller: str
    vehicle_count: int
    suffix: str
    directory: Path

    @property
    def label(self) -> str:
        suffix_text = f" {self.suffix}" if self.suffix else ""
        return f"{self.controller}{suffix_text} (N={self.vehicle_count})"

    @property
    def sort_key(self) -> tuple[int, int, str]:
        controller_order = 0 if self.controller == "Barrier Control" else 1
        return (self.vehicle_count, controller_order, self.suffix)


@dataclass
class BatchRuns:
    info: BatchInfo
    runs: list[pd.DataFrame]
    run_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 30-run means/95% CIs and plot all simulation batches."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("simulation_results"),
        help="Root folder containing BarrierControlResults and TrafficlightResults.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for summary CSVs and plots. Defaults inside simulation_results.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=700,
        help="Number of interpolation points for each mean/CI curve.",
    )
    parser.add_argument(
        "--vehicle-index",
        type=int,
        default=1,
        help=(
            "1-based ordinal vehicle to compare across all batches. "
            "Because CARLA IDs change per run, vehicle 1 means the first "
            "vehicle ID in each run, vehicle 2 the second, and so on."
        ),
    )
    parser.add_argument(
        "--skip-selected-vehicle",
        action="store_true",
        help="Skip the selected-vehicle energy/speed plots.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#CFCFCF",
            "legend.facecolor": "white",
            "legend.frameon": True,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "lines.solid_capstyle": "round",
            "savefig.dpi": 220,
        }
    )


def discover_batches(results_root: Path) -> list[BatchInfo]:
    batches: list[BatchInfo] = []

    for controller, subdir in CONTROLLER_DIRS.items():
        controller_dir = results_root / subdir
        if not controller_dir.is_dir():
            print(f"Warning: missing controller directory: {controller_dir}")
            continue

        for batch_dir in sorted(controller_dir.iterdir()):
            if not batch_dir.is_dir():
                continue

            if batch_dir.name not in ALLOWED_BATCH_DIRS.get(subdir, set()):
                continue

            match = BATCH_NAME_RE.search(batch_dir.name)
            if not match:
                continue

            vehicle_count = int(match.group("n"))
            # Build a unique suffix from the full directory name: strip "Batch_" and
            # the "_N=\d+" token, then concatenate what's before and after it.
            stripped = re.sub(r"(?i)^Batch_", "", batch_dir.name)
            n_token = re.search(r"(?i)[_]?[Nn]=\d+", stripped)
            if n_token:
                pre = stripped[: n_token.start()].strip(" _-")
                post = stripped[n_token.end() :].strip(" _-")
                suffix = (pre + (" " + post if post else "")).strip()
            else:
                suffix = match.group("suffix").strip(" _-")
            batches.append(
                BatchInfo(
                    controller=controller,
                    vehicle_count=vehicle_count,
                    suffix=suffix,
                    directory=batch_dir,
                )
            )

    return sorted(batches, key=lambda batch: batch.sort_key)


def run_number(run_dir: Path) -> int | None:
    match = RUN_DIR_RE.search(run_dir.name)
    return int(match.group("run")) if match else None


def latest_csv(run_dir: Path, prefix: str) -> Path | None:
    files = sorted(
        run_dir.glob(f"{prefix}_*.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not files:
        files = sorted(
            run_dir.glob(f"{prefix}*.csv"),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
    return files[-1] if files else None


def run_dir_for(batch: BatchInfo, run_id: int) -> Path:
    return batch.directory / f"run_{run_id:02d}"


def load_run_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        TIME_COL,
        DISTANCE_COL,
        ENERGY_COL,
        THROUGHPUT_COL,
        DELAY_COL,
        SAFETY_RATE_COL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    optional = [FLEET_SPEED_COL]
    keep = required + [c for c in optional if c in df.columns]
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    df = df.drop_duplicates(subset=[TIME_COL], keep="last")
    df[SAFETY_COUNT_COL] = df[SAFETY_RATE_COL] * df[TIME_COL]
    dist = df[DISTANCE_COL].replace(0.0, np.nan)
    df[ENERGY_PER_KM_COL] = df[ENERGY_COL] / dist
    return df.reset_index(drop=True)


def load_batch_runs(batch: BatchInfo) -> BatchRuns:
    runs: list[pd.DataFrame] = []
    run_ids: list[int] = []

    run_dirs = [
        run_dir
        for run_dir in batch.directory.iterdir()
        if run_dir.is_dir() and run_number(run_dir) is not None
    ]

    for run_dir in sorted(run_dirs, key=lambda path: run_number(path) or 0):
        run_id = run_number(run_dir)
        csv_path = latest_csv(run_dir, "throughput")
        if run_id is None or csv_path is None:
            print(f"Warning: skipped {run_dir}; no throughput_*.csv found.")
            continue

        try:
            runs.append(load_run_csv(csv_path))
            run_ids.append(run_id)
        except Exception as exc:
            print(f"Warning: skipped {csv_path}: {exc}")

    if len(runs) == 0:
        print(f"Warning: {batch.label} loaded 0 runs.")
    else:
        print(f"Info: {batch.label} loaded {len(runs)} run(s).")

    return BatchRuns(info=batch, runs=runs, run_ids=run_ids)


def load_selected_vehicle_csv(csv_path: Path, vehicle_index: int) -> pd.DataFrame | None:
    if vehicle_index < 1:
        raise ValueError("--vehicle-index must be 1 or higher.")

    wanted = {
        TIME_COL,
        VEHICLE_ID_COL,
        VELOCITY_MS_COL,
        VELOCITY_KMH_COL,
        POWER_KW_COL,
    }
    df = pd.read_csv(csv_path, usecols=lambda col: col in wanted)
    missing = sorted(wanted - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    for col in [TIME_COL, VEHICLE_ID_COL, VELOCITY_MS_COL, VELOCITY_KMH_COL, POWER_KW_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TIME_COL, VEHICLE_ID_COL])

    vehicle_ids = sorted(df[VEHICLE_ID_COL].dropna().unique())
    if len(vehicle_ids) < vehicle_index:
        return None

    selected_id = vehicle_ids[vehicle_index - 1]
    vehicle = df[df[VEHICLE_ID_COL] == selected_id].copy()
    vehicle = vehicle.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
    vehicle = vehicle.dropna(subset=[VELOCITY_MS_COL, POWER_KW_COL])
    if len(vehicle) < 2:
        return None

    time_s = vehicle[TIME_COL].to_numpy(dtype=float)
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.clip(dt, 0.0, None)
    speed_ms = vehicle[VELOCITY_MS_COL].to_numpy(dtype=float)
    power_kw = vehicle[POWER_KW_COL].to_numpy(dtype=float)

    vehicle[SELECTED_DISTANCE_COL] = np.cumsum(speed_ms * dt / 1000.0)
    vehicle[SELECTED_ENERGY_COL] = np.cumsum(power_kw * dt / 3600.0)
    return vehicle[
        [
            TIME_COL,
            VEHICLE_ID_COL,
            VELOCITY_KMH_COL,
            SELECTED_DISTANCE_COL,
            SELECTED_ENERGY_COL,
        ]
    ].reset_index(drop=True)


def load_selected_vehicle_runs(batch: BatchRuns, vehicle_index: int) -> BatchRuns:
    runs: list[pd.DataFrame] = []
    run_ids: list[int] = []

    for run_id in batch.run_ids:
        run_dir = run_dir_for(batch.info, run_id)
        csv_path = latest_csv(run_dir, "vehicle_all")
        if csv_path is None:
            print(f"Warning: skipped selected vehicle for {run_dir}; no vehicle_all_*.csv found.")
            continue

        try:
            run = load_selected_vehicle_csv(csv_path, vehicle_index)
        except Exception as exc:
            print(f"Warning: skipped selected vehicle in {csv_path}: {exc}")
            continue

        if run is None:
            print(
                f"Warning: skipped {batch.info.label} run_{run_id:02d}; "
                f"vehicle index {vehicle_index} is unavailable."
            )
            continue

        runs.append(run)
        run_ids.append(run_id)

    if len(runs) != len(batch.run_ids):
        print(
            f"Warning: {batch.info.label} selected-vehicle analysis loaded "
            f"{len(runs)} of {len(batch.run_ids)} runs."
        )

    return BatchRuns(info=batch.info, runs=runs, run_ids=run_ids)


def load_vehicle_type_csv(csv_path: Path, blueprint_filter: str | None) -> pd.DataFrame | None:
    """Load vehicle_all CSV and return a per-time DataFrame of means across all vehicles
    matching blueprint_filter (or all vehicles if None).  Each vehicle's cumulative
    net energy/km is computed independently, then averaged."""
    wanted = {TIME_COL, VEHICLE_ID_COL, BLUEPRINT_ID_COL, VELOCITY_MS_COL, VELOCITY_KMH_COL, POWER_KW_COL}
    df = pd.read_csv(csv_path, usecols=lambda col: col in wanted)
    required = {TIME_COL, VEHICLE_ID_COL, VELOCITY_MS_COL, POWER_KW_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    has_kmh = VELOCITY_KMH_COL in df.columns
    for col in required | ({VELOCITY_KMH_COL} if has_kmh else set()):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TIME_COL, VEHICLE_ID_COL])

    if blueprint_filter is not None and BLUEPRINT_ID_COL in df.columns:
        df = df[df[BLUEPRINT_ID_COL] == blueprint_filter]
    if df.empty:
        return None

    vehicle_series: list[pd.DataFrame] = []
    for _, vdf in df.groupby(VEHICLE_ID_COL):
        vdf = vdf.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
        vdf = vdf.dropna(subset=[VELOCITY_MS_COL, POWER_KW_COL])
        if len(vdf) < 2:
            continue
        time_s = vdf[TIME_COL].to_numpy(dtype=float)
        dt = np.clip(np.diff(time_s, prepend=time_s[0]), 0.0, None)
        speed_ms = vdf[VELOCITY_MS_COL].to_numpy(dtype=float)
        power_kw = vdf[POWER_KW_COL].to_numpy(dtype=float)
        dist_km = np.cumsum(speed_ms * dt / 1000.0)
        energy_kwh = np.cumsum(power_kw * dt / 3600.0)
        energy_per_km = np.where(dist_km > 1e-6, energy_kwh / dist_km, np.nan)
        row = {
            TIME_COL: time_s,
            SELECTED_DISTANCE_COL: dist_km,
            SELECTED_ENERGY_COL: energy_kwh,
            VEHICLE_TYPE_ENERGY_PER_KM_COL: energy_per_km,
        }
        if has_kmh:
            row[VEHICLE_TYPE_SPEED_COL] = vdf[VELOCITY_KMH_COL].to_numpy(dtype=float)
        vehicle_series.append(pd.DataFrame(row))

    if not vehicle_series:
        return None

    t_start = max(float(vs[TIME_COL].iloc[0]) for vs in vehicle_series)
    t_end = min(float(vs[TIME_COL].iloc[-1]) for vs in vehicle_series)
    if t_end <= t_start:
        return None

    grid = np.linspace(t_start, t_end, 500)

    def _interp_mean(col: str) -> np.ndarray:
        rows = []
        for vs in vehicle_series:
            vals = pd.Series(vs[col].to_numpy(dtype=float)).ffill().bfill().to_numpy()
            rows.append(np.interp(grid, vs[TIME_COL].to_numpy(dtype=float), vals))
        return np.nanmean(np.vstack(rows), axis=0)

    out = {
        TIME_COL: grid,
        SELECTED_DISTANCE_COL: _interp_mean(SELECTED_DISTANCE_COL),
        SELECTED_ENERGY_COL: _interp_mean(SELECTED_ENERGY_COL),
        VEHICLE_TYPE_ENERGY_PER_KM_COL: _interp_mean(VEHICLE_TYPE_ENERGY_PER_KM_COL),
    }
    if has_kmh:
        out[VEHICLE_TYPE_SPEED_COL] = _interp_mean(VEHICLE_TYPE_SPEED_COL)
    return pd.DataFrame(out)


def load_vehicle_type_batch_runs(batch: BatchRuns, blueprint_filter: str | None) -> BatchRuns:
    type_label = blueprint_filter.split(".")[-1] if blueprint_filter else "all_vehicles"
    runs: list[pd.DataFrame] = []
    run_ids: list[int] = []
    for run_id in batch.run_ids:
        run_dir = run_dir_for(batch.info, run_id)
        csv_path = latest_csv(run_dir, "vehicle_all")
        if csv_path is None:
            print(f"Warning: skipped {type_label} for {run_dir}; no vehicle_all_*.csv found.")
            continue
        try:
            run = load_vehicle_type_csv(csv_path, blueprint_filter)
        except Exception as exc:
            print(f"Warning: skipped {type_label} in {csv_path}: {exc}")
            continue
        if run is None:
            print(f"Warning: no {type_label} vehicles in {run_dir}.")
            continue
        runs.append(run)
        run_ids.append(run_id)
    if len(runs) != len(batch.run_ids):
        print(
            f"Warning: {batch.info.label} {type_label} loaded "
            f"{len(runs)} of {len(batch.run_ids)} runs."
        )
    return BatchRuns(info=batch.info, runs=runs, run_ids=run_ids)


def load_single_vehicle_type_csv(
    csv_path: Path, blueprint_filter: str | None, vehicle_index: int
) -> pd.DataFrame | None:
    """Pick the vehicle_index-th vehicle (1-based) of the given blueprint from vehicle_all CSV
    and return its time-series with cumulative distance, net energy, and energy/km."""
    wanted = {TIME_COL, VEHICLE_ID_COL, BLUEPRINT_ID_COL, VELOCITY_MS_COL, POWER_KW_COL}
    df = pd.read_csv(csv_path, usecols=lambda col: col in wanted)
    required = {TIME_COL, VEHICLE_ID_COL, VELOCITY_MS_COL, POWER_KW_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TIME_COL, VEHICLE_ID_COL])

    if blueprint_filter is not None and BLUEPRINT_ID_COL in df.columns:
        df = df[df[BLUEPRINT_ID_COL] == blueprint_filter]
    if df.empty:
        return None

    vehicle_ids = sorted(df[VEHICLE_ID_COL].unique())
    if len(vehicle_ids) < vehicle_index:
        return None

    selected_id = vehicle_ids[vehicle_index - 1]
    vdf = df[df[VEHICLE_ID_COL] == selected_id].copy()
    vdf = vdf.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
    vdf = vdf.dropna(subset=[VELOCITY_MS_COL, POWER_KW_COL])
    if len(vdf) < 2:
        return None

    time_s = vdf[TIME_COL].to_numpy(dtype=float)
    dt = np.clip(np.diff(time_s, prepend=time_s[0]), 0.0, None)
    speed_ms = vdf[VELOCITY_MS_COL].to_numpy(dtype=float)
    power_kw = vdf[POWER_KW_COL].to_numpy(dtype=float)
    dist_km = np.cumsum(speed_ms * dt / 1000.0)
    energy_kwh = np.cumsum(power_kw * dt / 3600.0)
    energy_per_km = np.where(dist_km > 1e-6, energy_kwh / dist_km, np.nan)

    return pd.DataFrame({
        TIME_COL: time_s,
        SELECTED_DISTANCE_COL: dist_km,
        SELECTED_ENERGY_COL: energy_kwh,
        VEHICLE_TYPE_ENERGY_PER_KM_COL: energy_per_km,
    })


def load_single_vehicle_type_batch_runs(
    batch: BatchRuns, blueprint_filter: str | None, vehicle_index: int
) -> BatchRuns:
    type_label = blueprint_filter.split(".")[-1] if blueprint_filter else "all_vehicles"
    runs: list[pd.DataFrame] = []
    run_ids: list[int] = []
    for run_id in batch.run_ids:
        run_dir = run_dir_for(batch.info, run_id)
        csv_path = latest_csv(run_dir, "vehicle_all")
        if csv_path is None:
            print(f"Warning: skipped single {type_label} for {run_dir}; no vehicle_all_*.csv found.")
            continue
        try:
            run = load_single_vehicle_type_csv(csv_path, blueprint_filter, vehicle_index)
        except Exception as exc:
            print(f"Warning: skipped single {type_label} in {csv_path}: {exc}")
            continue
        if run is None:
            print(
                f"Warning: {batch.info.label} run_{run_id:02d}: "
                f"vehicle index {vehicle_index} of type {type_label} unavailable."
            )
            continue
        runs.append(run)
        run_ids.append(run_id)
    if len(runs) != len(batch.run_ids):
        print(
            f"Warning: {batch.info.label} single {type_label} loaded "
            f"{len(runs)} of {len(batch.run_ids)} runs."
        )
    return BatchRuns(info=batch.info, runs=runs, run_ids=run_ids)


def student_t_critical_95(df: np.ndarray) -> np.ndarray:
    """Two-sided 95% t critical values, with SciPy if available."""
    try:
        from scipy.stats import t as student_t

        return student_t.ppf(0.975, df)
    except Exception:
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
            21: 2.080,
            22: 2.074,
            23: 2.069,
            24: 2.064,
            25: 2.060,
            26: 2.056,
            27: 2.052,
            28: 2.048,
            29: 2.045,
            30: 2.042,
        }
        out = np.full_like(df, 1.96, dtype=float)
        for key, value in table.items():
            out[df == key] = value
        out[df < 1] = np.nan
        return out


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return n, mean, lower CI, upper CI, and CI half-width by column."""
    values = np.asarray(values, dtype=float)
    counts = np.sum(~np.isnan(values), axis=0)
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0, ddof=1)
    sems = stds / np.sqrt(counts)
    tcrit = student_t_critical_95(np.maximum(counts - 1, 0))
    half_widths = tcrit * sems

    half_widths[counts <= 1] = np.nan
    lower = means - half_widths
    upper = means + half_widths
    return counts, means, lower, upper, half_widths


def final_metric_values(batch_runs: list[BatchRuns]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for batch in batch_runs:
        if not batch.runs:
            continue

        for run_id, run in zip(batch.run_ids, batch.runs):
            final = run.iloc[-1]
            rows.append(
                {
                    "controller": batch.info.controller,
                    "vehicles": batch.info.vehicle_count,
                    "batch_label": batch.info.label,
                    "run_id": run_id,
                    "energy_kWh": float(final[ENERGY_COL]),
                    "energy_kWh_per_km": float(final[ENERGY_COL] / final[DISTANCE_COL]),
                    "energy_kWh_per_vehicle": float(final[ENERGY_COL] / batch.info.vehicle_count),
                    "safety_violations": float(final[SAFETY_COUNT_COL]),
                    "safety_violations_per_vehicle": float(
                        final[SAFETY_COUNT_COL] / batch.info.vehicle_count
                    ),
                    "delay_s": float(final[DELAY_COL]),
                    "delay_s_per_vehicle": float(final[DELAY_COL] / batch.info.vehicle_count),
                    "throughput_veh_s": float(final[THROUGHPUT_COL]),
                    "throughput_veh_s_per_vehicle": float(
                        final[THROUGHPUT_COL] / batch.info.vehicle_count
                    ),
                    "total_distance_km": float(final[DISTANCE_COL]),
                    "total_distance_km_per_vehicle": float(
                        final[DISTANCE_COL] / batch.info.vehicle_count
                    ),
                }
            )

    return pd.DataFrame(rows)


def final_metric_summary(final_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics = [
        "energy_kWh",
        "energy_kWh_per_km",
        "energy_kWh_per_vehicle",
        "safety_violations",
        "safety_violations_per_vehicle",
        "delay_s",
        "delay_s_per_vehicle",
        "throughput_veh_s",
        "throughput_veh_s_per_vehicle",
        "total_distance_km",
        "total_distance_km_per_vehicle",
    ]

    group_cols = ["controller", "vehicles", "batch_label"]
    for (controller, vehicles, batch_label), group in final_values.groupby(group_cols, sort=False):
        for metric in metrics:
            values_1d = group[metric].to_numpy(dtype=float)
            values = values_1d[:, np.newaxis]
            counts, means, lower, upper, half_widths = mean_ci(values)
            rows.append(
                {
                    "controller": controller,
                    "vehicles": int(vehicles),
                    "batch_label": batch_label,
                    "metric": metric,
                    "n_runs": int(counts[0]),
                    "mean": float(means[0]),
                    "std": float(np.nanstd(values_1d, ddof=1)),
                    "ci95_half_width": float(half_widths[0]),
                    "ci95_low": float(lower[0]),
                    "ci95_high": float(upper[0]),
                    "min": float(np.nanmin(values_1d)),
                    "max": float(np.nanmax(values_1d)),
                }
            )

    return pd.DataFrame(rows)


def prepare_xy(run: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    xy = run[[x_col, y_col]].dropna().sort_values(x_col)
    xy = xy.groupby(x_col, as_index=False)[y_col].last()
    xy = xy[xy[x_col].diff().fillna(1.0) >= 0.0]
    return xy[x_col].to_numpy(dtype=float), xy[y_col].to_numpy(dtype=float)


def build_curve_summary(
    batch: BatchRuns,
    x_col: str,
    y_col: str,
    x_name: str,
    y_name: str,
    grid_points: int,
) -> pd.DataFrame:
    try:
        prepared = [prepare_xy(run, x_col, y_col) for run in batch.runs]
    except KeyError:
        return pd.DataFrame()
    prepared = [(x, y) for x, y in prepared if len(x) >= 2 and np.nanmax(x) > np.nanmin(x)]
    if not prepared:
        return pd.DataFrame()

    x_start = max(float(np.nanmin(x)) for x, _ in prepared)
    x_end = min(float(np.nanmax(x)) for x, _ in prepared)
    if x_end <= x_start:
        return pd.DataFrame()

    grid = np.linspace(x_start, x_end, grid_points)
    interpolated = np.vstack([np.interp(grid, x, y) for x, y in prepared])
    counts, means, lower, upper, half_widths = mean_ci(interpolated)

    return pd.DataFrame(
        {
            "controller": batch.info.controller,
            "vehicles": batch.info.vehicle_count,
            "batch_label": batch.info.label,
            x_name: grid,
            y_name: means,
            "n_runs": counts.astype(int),
            "ci95_low": lower,
            "ci95_high": upper,
            "ci95_half_width": half_widths,
        }
    )


def filter_curve_window(
    curve: pd.DataFrame,
    x_col: str,
    x_limits: tuple[float, float] | None,
) -> pd.DataFrame:
    if curve.empty or x_limits is None:
        return curve

    x_min, x_max = x_limits
    return curve[(curve[x_col] >= x_min) & (curve[x_col] <= x_max)].reset_index(drop=True)


def color_for(batch: BatchInfo, index: int) -> str:
    return SERIES_COLORS.get(
        (batch.controller, batch.vehicle_count),
        FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
    )


def plot_curve(
    summaries: list[tuple[BatchInfo, pd.DataFrame]],
    x_col: str,
    y_col: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: Path,
    clamp_low_to_zero: bool = True,
    x_limits: tuple[float, float] | None = None,
    tight_y_axis: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    y_bounds: list[np.ndarray] = []

    for idx, (batch, summary) in enumerate(summaries):
        if summary.empty:
            continue
        summary = filter_curve_window(summary, x_col, x_limits)
        if summary.empty:
            continue

        color = color_for(batch, idx)
        linestyle = CONTROLLER_LINESTYLES.get(batch.controller, "-")
        x = summary[x_col].to_numpy(dtype=float)
        mean = summary[y_col].to_numpy(dtype=float)
        low = summary["ci95_low"].to_numpy(dtype=float)
        high = summary["ci95_high"].to_numpy(dtype=float)
        if clamp_low_to_zero:
            low = np.maximum(low, 0.0)

        y_bounds.extend([low, high])
        ax.fill_between(x, low, high, color=color, alpha=0.22, linewidth=0)
        ax.plot(x, low, color=color, linestyle=":", linewidth=1.0, alpha=0.85)
        ax.plot(x, high, color=color, linestyle=":", linewidth=1.0, alpha=0.85)
        ax.plot(
            x,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.6,
            label=batch.label,
        )

    ax.set_title(f"{title} (mean with 95% CI)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if tight_y_axis and y_bounds:
        y_values = np.concatenate([values[np.isfinite(values)] for values in y_bounds])
        if len(y_values):
            y_min = float(np.nanmin(y_values))
            y_max = float(np.nanmax(y_values))
            pad = max((y_max - y_min) * 0.15, abs(y_max) * 0.002, 1e-6)
            ax.set_ylim(y_min - pad, y_max + pad)
    else:
        ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            title="Controller and density",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def plot_final_metric_boxplots(
    final_values: pd.DataFrame,
    batches: list[BatchRuns],
    output_dir: Path,
) -> None:
    metrics = [
        (
            "energy_kWh",
            "Energy Consumption",
            "Energy [kWh]",
            "final_energy_kwh_boxplot.png",
            True,
        ),
        (
            "safety_violations",
            "Safety Violations",
            "Violations [count]",
            "final_safety_violations_boxplot_with_95ci.png",
            False,
        ),
        (
            "delay_s",
            "Delay Time",
            "Delay [s]",
            "final_delay_s_boxplot_with_95ci.png",
            False,
        ),
        (
            "throughput_veh_s",
            "Throughput",
            "Throughput [vehicles/s]",
            "final_throughput_boxplot_with_95ci.png",
            True,
        ),
    ]

    labels = [batch.info.label for batch in batches]
    positions = np.arange(1, len(labels) + 1)

    for metric, title, ylabel, filename, tight_y_axis in metrics:
        fig, ax = plt.subplots(figsize=(13.5, 6.8))
        rng = np.random.default_rng(20260601)
        grouped_values = [
            final_values.loc[final_values["batch_label"] == label, metric].to_numpy(dtype=float)
            for label in labels
        ]

        box = ax.boxplot(
            grouped_values,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.6},
            whiskerprops={"color": "#666666", "linewidth": 1.0},
            capprops={"color": "#666666", "linewidth": 1.0},
        )

        for idx, patch in enumerate(box["boxes"]):
            color = color_for(batches[idx].info, idx)
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.8)

        is_energy = metric.startswith("energy")
        for idx, values in enumerate(grouped_values):
            color = color_for(batches[idx].info, idx)
            jitter = rng.normal(0.0, 0.055, size=len(values))
            ax.scatter(
                np.full(len(values), positions[idx]) + jitter,
                values,
                s=20,
                color=color,
                alpha=0.70,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )

            if not is_energy:
                counts, means, _, _, half_widths = mean_ci(values[:, np.newaxis])
                if counts[0] > 1:
                    ax.errorbar(
                        positions[idx],
                        means[0],
                        yerr=half_widths[0],
                        fmt="D",
                        color="#111111",
                        markerfacecolor="white",
                        markeredgecolor="#111111",
                        markersize=5,
                        elinewidth=1.8,
                        capsize=5,
                        capthick=1.8,
                        zorder=4,
                    )

        ax.set_title(f"{title}: final run values")
        ax.set_ylabel(ylabel)
        if tight_y_axis:
            all_values = np.concatenate([values for values in grouped_values if len(values)])
            if len(all_values):
                y_min = float(np.nanmin(all_values))
                y_max = float(np.nanmax(all_values))
                pad = max((y_max - y_min) * 0.20, abs(y_max) * 0.002, 1e-6)
                ax.set_ylim(y_min - pad, y_max + pad)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        if is_energy:
            footnote = "Boxes show run distributions (median, IQR, whiskers); points are individual runs."
        else:
            footnote = "Boxes show run distributions; points are individual runs; diamond/error bar is mean +/- 95% CI."
        fig.text(
            0.5,
            0.012,
            footnote,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#333333",
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)


def plot_energy_per_km_boxplot(
    final_values: pd.DataFrame,
    batches: list[BatchRuns],
    output_dir: Path,
) -> None:
    labels = [batch.info.label for batch in batches]
    positions = np.arange(1, len(labels) + 1)
    values_by_label = [
        final_values.loc[final_values["batch_label"] == label, "energy_kWh_per_km"].to_numpy(dtype=float)
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    box = ax.boxplot(
        values_by_label,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )

    rng = np.random.default_rng(20260602)
    for idx, values in enumerate(values_by_label):
        color = color_for(batches[idx].info, idx)
        box["boxes"][idx].set_facecolor(color)
        box["boxes"][idx].set_alpha(0.20)
        box["boxes"][idx].set_edgecolor(color)
        box["boxes"][idx].set_linewidth(1.8)
        ax.scatter(
            np.full(len(values), positions[idx]) + rng.normal(0.0, 0.055, size=len(values)),
            values,
            s=24,
            color=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    ax.set_title("Energy per Distance: final run values")
    ax.set_ylabel("Energy per distance [kWh/km]")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    all_values = np.concatenate([values for values in values_by_label if len(values)])
    if len(all_values):
        y_min = float(np.nanmin(all_values))
        y_max = float(np.nanmax(all_values))
        pad = max((y_max - y_min) * 0.20, 0.001)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    path = output_dir / "final_energy_per_km_boxplot.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def save_curve_csv(curves: list[tuple[BatchInfo, pd.DataFrame]], output_dir: Path, filename: str) -> Path:
    frames = [curve for _, curve in curves if not curve.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = output_dir / filename
    combined.to_csv(path, index=False)
    print(f"Saved {path}")
    return path


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_root / "PlotsControllers" / "barrier_FAS_N57_81"
    )

    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    batches = discover_batches(results_root)
    if not batches:
        raise FileNotFoundError(f"No Batch_30runs_N=* folders found under {results_root}")

    print("Discovered batches:")
    for batch in batches:
        print(f"  - {batch.label}: {batch.directory}")

    loaded_batches = [load_batch_runs(batch) for batch in batches]
    loaded_batches = [batch for batch in loaded_batches if batch.runs]
    if not loaded_batches:
        raise RuntimeError("No usable throughput CSVs were loaded.")

    final_values = final_metric_values(loaded_batches)
    final_values_path = csv_dir / "final_run_values.csv"
    final_values.to_csv(final_values_path, index=False)
    print(f"Saved {final_values_path}")

    summary = final_metric_summary(final_values)
    summary_path = csv_dir / "final_metric_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    plot_final_metric_boxplots(final_values, loaded_batches, plots_dir)
    plot_energy_per_km_boxplot(final_values, loaded_batches, plots_dir)

    curve_specs = [
        {
            "name": "energy_kwh_vs_distance",
            "x_source": DISTANCE_COL,
            "y_source": ENERGY_COL,
            "x_col": "total_distance_km",
            "y_col": "energy_kWh_mean",
            "csv": "curve_energy_kwh_vs_distance.csv",
            "plot": "energy_kwh_vs_total_distance_all_batches.png",
            "title": "Energy Consumption vs Total Distance",
            "xlabel": "Total distance travelled [km]",
            "ylabel": "Energy consumption [kWh]",
        },
        {
            "name": "energy_per_km_vs_distance",
            "x_source": DISTANCE_COL,
            "y_source": ENERGY_PER_KM_COL,
            "x_col": "total_distance_km",
            "y_col": "energy_kWh_per_km_mean",
            "csv": "curve_energy_per_km_vs_distance.csv",
            "plot": "energy_per_km_vs_total_distance_all_batches.png",
            "title": "Energy per Distance vs Total Distance",
            "xlabel": "Total distance travelled [km]",
            "ylabel": "Energy per distance [kWh/km]",
            "tight_y_axis": True,
        },
        {
            "name": "safety_violations_vs_time",
            "x_source": TIME_COL,
            "y_source": SAFETY_COUNT_COL,
            "x_col": "time_seconds",
            "y_col": "safety_violations_mean",
            "csv": "curve_safety_violations_vs_time.csv",
            "plot": "safety_violations_vs_time_all_batches.png",
            "title": "Safety Violations vs Time",
            "xlabel": "Simulation time [s]",
            "ylabel": "Cumulative safety violations [count]",
        },
        {
            "name": "delay_vs_time",
            "x_source": TIME_COL,
            "y_source": DELAY_COL,
            "x_col": "time_seconds",
            "y_col": "delay_s_mean",
            "csv": "curve_delay_vs_time.csv",
            "plot": "delay_vs_time_all_batches.png",
            "title": "Delay Time vs Time",
            "xlabel": "Simulation time [s]",
            "ylabel": "Cumulative delay [s]",
        },
        {
            "name": "throughput_vs_time",
            "x_source": TIME_COL,
            "y_source": THROUGHPUT_COL,
            "x_col": "time_seconds",
            "y_col": "throughput_veh_s_mean",
            "csv": "curve_throughput_vs_time.csv",
            "plot": "throughput_vs_time_all_batches.png",
            "title": "Throughput vs Time",
            "xlabel": "Simulation time [s]",
            "ylabel": "Throughput [vehicles/s]",
            "tight_y_axis": True,
        },
        {
            "name": "avg_speed_vs_time",
            "x_source": TIME_COL,
            "y_source": FLEET_SPEED_COL,
            "x_col": "time_seconds",
            "y_col": "avg_speed_kmh_mean",
            "csv": "curve_avg_speed_vs_time.csv",
            "plot": "avg_speed_vs_time_all_batches.png",
            "title": "Fleet Average Speed vs Time",
            "xlabel": "Simulation time [s]",
            "ylabel": "Average speed [km/h]",
            "tight_y_axis": True,
        },
    ]

    for spec in curve_specs:
        curves: list[tuple[BatchInfo, pd.DataFrame]] = []
        for batch in loaded_batches:
            curve = build_curve_summary(
                batch=batch,
                x_col=spec["x_source"],
                y_col=spec["y_source"],
                x_name=spec["x_col"],
                y_name=spec["y_col"],
                grid_points=args.grid_points,
            )
            curves.append((batch.info, curve))

        curves_for_csv = [
            (
                batch,
                filter_curve_window(curve, spec["x_col"], spec.get("x_limits")),
            )
            for batch, curve in curves
        ]
        save_curve_csv(curves_for_csv, csv_dir, spec["csv"])
        plot_curve(
            summaries=curves,
            x_col=spec["x_col"],
            y_col=spec["y_col"],
            filename=spec["plot"],
            title=spec["title"],
            xlabel=spec["xlabel"],
            ylabel=spec["ylabel"],
            output_dir=plots_dir,
            x_limits=spec.get("x_limits"),
            tight_y_axis=spec.get("tight_y_axis", False),
        )

    if not args.skip_selected_vehicle:
        print(f"\nLoading selected vehicle index {args.vehicle_index} from vehicle_all CSVs...")
        selected_batches = [
            load_selected_vehicle_runs(batch, args.vehicle_index) for batch in loaded_batches
        ]
        selected_batches = [batch for batch in selected_batches if batch.runs]

        selected_specs = [
            {
                "x_source": SELECTED_DISTANCE_COL,
                "y_source": SELECTED_ENERGY_COL,
                "x_col": "selected_vehicle_distance_km",
                "y_col": "selected_vehicle_energy_kWh_mean",
                "csv": f"curve_selected_vehicle_{args.vehicle_index}_energy_vs_distance.csv",
                "plot": f"selected_vehicle_{args.vehicle_index}_energy_vs_distance_all_batches.png",
                "title": f"Selected Vehicle {args.vehicle_index} Energy vs Distance",
                "xlabel": "Selected vehicle distance travelled [km]",
                "ylabel": "Selected vehicle net energy [kWh]",
            },
            {
                "x_source": TIME_COL,
                "y_source": VELOCITY_KMH_COL,
                "x_col": "time_seconds",
                "y_col": "selected_vehicle_speed_kmh_mean",
                "csv": f"curve_selected_vehicle_{args.vehicle_index}_speed_vs_time.csv",
                "plot": f"selected_vehicle_{args.vehicle_index}_speed_vs_time_all_batches.png",
                "title": f"Selected Vehicle {args.vehicle_index} Speed vs Time",
                "xlabel": "Simulation time [s]",
                "ylabel": "Selected vehicle speed [km/h]",
            },
        ]

        for spec in selected_specs:
            curves = []
            for batch in selected_batches:
                curve = build_curve_summary(
                    batch=batch,
                    x_col=spec["x_source"],
                    y_col=spec["y_source"],
                    x_name=spec["x_col"],
                    y_name=spec["y_col"],
                    grid_points=args.grid_points,
                )
                curves.append((batch.info, curve))

            save_curve_csv(curves, csv_dir, spec["csv"])
            plot_curve(
                summaries=curves,
                x_col=spec["x_col"],
                y_col=spec["y_col"],
                filename=spec["plot"],
                title=spec["title"],
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
                output_dir=plots_dir,
                clamp_low_to_zero=False,
            )

    print("\nGenerating per-vehicle-type energy/km plots...")
    for blueprint_filter, filename_label, display_label in VEHICLE_TYPES:
        type_batches = [
            load_vehicle_type_batch_runs(batch, blueprint_filter)
            for batch in loaded_batches
        ]
        type_batches = [b for b in type_batches if b.runs]
        if not type_batches:
            print(f"Warning: no data for {display_label}, skipping.")
            continue

        fl = filename_label.lower()
        type_curve_specs = [
            {
                "x_source": TIME_COL,
                "y_source": VEHICLE_TYPE_ENERGY_PER_KM_COL,
                "x_col": "time_seconds",
                "y_col": f"{fl}_energy_per_km_mean",
                "csv": f"curve_{fl}_energy_per_km_vs_time.csv",
                "plot": f"{fl}_energy_per_km_vs_time_all_batches.png",
                "title": f"{display_label}: Energy per km vs Time",
                "xlabel": "Simulation time [s]",
                "ylabel": "Energy per km [kWh/km]",
                "tight_y_axis": True,
            },
            {
                "x_source": SELECTED_DISTANCE_COL,
                "y_source": VEHICLE_TYPE_ENERGY_PER_KM_COL,
                "x_col": "selected_vehicle_distance_km",
                "y_col": f"{fl}_energy_per_km_dist_mean",
                "csv": f"curve_{fl}_energy_per_km_vs_distance.csv",
                "plot": f"{fl}_energy_per_km_vs_distance_all_batches.png",
                "title": f"{display_label}: Energy per km vs Distance",
                "xlabel": "Mean distance travelled [km]",
                "ylabel": "Energy per km [kWh/km]",
                "tight_y_axis": True,
            },
            {
                "x_source": TIME_COL,
                "y_source": VEHICLE_TYPE_SPEED_COL,
                "x_col": "time_seconds",
                "y_col": f"{fl}_speed_kmh_mean",
                "csv": f"curve_{fl}_speed_vs_time.csv",
                "plot": f"{fl}_speed_vs_time_all_batches.png",
                "title": f"{display_label}: Average Speed vs Time",
                "xlabel": "Simulation time [s]",
                "ylabel": "Average speed [km/h]",
                "tight_y_axis": True,
            },
        ]

        for spec in type_curve_specs:
            curves: list[tuple[BatchInfo, pd.DataFrame]] = []
            for batch in type_batches:
                curve = build_curve_summary(
                    batch=batch,
                    x_col=spec["x_source"],
                    y_col=spec["y_source"],
                    x_name=spec["x_col"],
                    y_name=spec["y_col"],
                    grid_points=args.grid_points,
                )
                curves.append((batch.info, curve))
            save_curve_csv(curves, csv_dir, spec["csv"])
            plot_curve(
                summaries=curves,
                x_col=spec["x_col"],
                y_col=spec["y_col"],
                filename=spec["plot"],
                title=spec["title"],
                xlabel=spec["xlabel"],
                ylabel=spec["ylabel"],
                output_dir=plots_dir,
                clamp_low_to_zero=False,
                tight_y_axis=True,
            )

    print(f"\nGenerating single-vehicle energy/km plots (vehicle index {args.vehicle_index})...")
    for blueprint_filter, filename_label, display_label in VEHICLE_TYPES:
        single_batches = [
            load_single_vehicle_type_batch_runs(batch, blueprint_filter, args.vehicle_index)
            for batch in loaded_batches
        ]
        single_batches = [b for b in single_batches if b.runs]
        if not single_batches:
            print(f"Warning: no single-vehicle data for {display_label}, skipping.")
            continue

        fl = filename_label.lower()
        y_col = f"{fl}_single_energy_per_km_mean"
        csv_name = f"curve_{fl}_single_v{args.vehicle_index}_energy_per_km_vs_time.csv"
        plot_name = f"{fl}_single_v{args.vehicle_index}_energy_per_km_vs_time_all_batches.png"
        curves_single: list[tuple[BatchInfo, pd.DataFrame]] = []
        for batch in single_batches:
            curve = build_curve_summary(
                batch=batch,
                x_col=TIME_COL,
                y_col=VEHICLE_TYPE_ENERGY_PER_KM_COL,
                x_name="time_seconds",
                y_name=y_col,
                grid_points=args.grid_points,
            )
            curves_single.append((batch.info, curve))
        save_curve_csv(curves_single, csv_dir, csv_name)
        plot_curve(
            summaries=curves_single,
            x_col="time_seconds",
            y_col=y_col,
            filename=plot_name,
            title=f"{display_label} (vehicle {args.vehicle_index}): Energy per km vs Time",
            xlabel="Simulation time [s]",
            ylabel="Energy per km [kWh/km]",
            output_dir=plots_dir,
            clamp_low_to_zero=False,
            tight_y_axis=True,
        )

    print("\nDone.")
    print(f"Plots in:  {plots_dir}")
    print(f"CSVs in:   {csv_dir}")


if __name__ == "__main__":
    main()

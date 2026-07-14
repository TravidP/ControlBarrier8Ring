#!/usr/bin/env python3
"""
Plot per-vehicle metrics from CARLA simulation CSVs.

For each vehicle spawn index, produces comparison plots overlaying the same
vehicle across all controller types and vehicle counts (N=37/47/57).
All completed, aligned runs in each (controller, N) group are aggregated
(mean ± 1 std band).  Run indices are intersected across controllers per N
value so both sides always use the same set of runs.

Also produces a barrier-control firing scatter plot for all vehicles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Style -----------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "legend.framealpha": 0.88,
    "legend.edgecolor": "#CCCCCC",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#F7F7F7",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.3,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
})

# ---------- Constants -------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = (
    PROJECT_DIR
    / "simulation_results"
    / "Final Results"
    / "Noise 0.1 22-05-2026"
)

# (controller, N) → batch directory that holds batch_summary_*.csv files.
# The summaries reference absolute paths for the actual run data.
BATCHES: dict[tuple[str, int], Path] = {
    ("Trafficlight",   37): BASE_DIR / "Trafficlight"   / "batch_20260521_164039_N37_T5400_R250p0",
    ("Trafficlight",   47): BASE_DIR / "Trafficlight"   / "batch_20260519_215956_N47_T5400_R250p0",
    ("Trafficlight",   57): BASE_DIR / "Trafficlight"   / "batch_20260520_203223_N57_T5400_R250p0",
    ("Barriercontrol", 37): BASE_DIR / "Barriercontrol" / "batch_20260521_094829_N37_T5400_R250p0",
    ("Barriercontrol", 47): BASE_DIR / "Barriercontrol" / "batch_20260519_155403_N47_T5400_R250p0",
    ("Barriercontrol", 57): BASE_DIR / "Barriercontrol" / "batch_20260520_151150_N57_T5400_R250p0",
}

DEFAULT_OUTPUT_DIR = BASE_DIR / "simulation_results" / "PlotsPerVehicle"

# Colours: TL = blue family, BC = red family; darker = fewer vehicles
GROUP_N_COLORS: dict[tuple[str, int], str] = {
    ("Trafficlight",   37): "#08306B",
    ("Trafficlight",   47): "#2171B5",
    ("Trafficlight",   57): "#6BAED6",
    ("Barriercontrol", 37): "#67000D",
    ("Barriercontrol", 47): "#CB181D",
    ("Barriercontrol", 57): "#FC9272",
}

CTRL_DISPLAY: dict[str, str] = {
    "Trafficlight":   "Traffic Light",
    "Barriercontrol": "Barrier Control",
}

X_COL = "time_seconds"

# (column, y-label, plot title, y_clip_pct or None)
# y_clip_pct: after plotting, clamp y-axis to [P(100-p), P(p)] of all data.
# power_kw has transient spikes to 90-177 kW while cruising mean is ~2 kW;
# clipping to P2-P98 keeps the band meaningful without hiding any series.
PLOT_COLS = [
    ("velocity_kmh",     "Velocity [km/h]",        "Velocity vs. Time",        None),
    ("acceleration_ms2", "Acceleration [m/s²]",    "Acceleration vs. Time",    None),
    ("u_headway",        "Headway Control [km/h]",  "Headway Control vs. Time", None),
    ("u_barrier",        "Barrier Control [km/h]",  "Barrier Control vs. Time", None),
    ("power_kw",         "Power [kW]",              "Power vs. Time",           98.0),
]


# ---------- Argument parsing ------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-vehicle metrics for all controllers and vehicle counts "
            "(N=37/47/57).  Run indices are intersected per N value so both "
            "controllers always use the same runs."
        )
    )
    parser.add_argument(
        "--num-vehicles",
        type=int,
        default=8,
        help="Maximum number of vehicles (by spawn order) to plot (default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where PNG plots are saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the figures interactively after saving.",
    )
    return parser.parse_args()


# ---------- Sorting helpers -------------------------------------------------

def _sorted_keys(groups: dict) -> list:
    """Sort (controller, N) keys by N asc then controller name asc.

    'Barriercontrol' < 'Trafficlight' alphabetically, so for each N the order
    is BC then TL — which gives left-column / right-column in a ncol=2 legend.
    """
    return sorted(groups.keys(), key=lambda k: (k[1], k[0]))


# ---------- Loading ---------------------------------------------------------

def find_vehicle_csv(run_dir: Path) -> Path | None:
    """Prefer vehicle_all*.csv, fall back to vehicle_first4*.csv."""
    for pattern in ("vehicle_all*.csv", "vehicle_first4*.csv"):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def load_vehicle_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if X_COL not in df.columns:
        raise ValueError(f"CSV '{path}' missing required column '{X_COL}'")
    for col, *_ in PLOT_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def load_all_runs(
    group_name: str, n_vehicles: int, batch_dir: Path
) -> list[tuple[pd.DataFrame, int]]:
    """Return (vehicle_df, run_index) for every *complete* run in batch_dir."""
    batch_dir = batch_dir.resolve()
    summary_files = sorted(batch_dir.glob("batch_summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No batch_summary_*.csv in {batch_dir}")

    frames: list[pd.DataFrame] = []
    for sf in summary_files:
        try:
            df = pd.read_csv(sf)
            if not df.empty:
                frames.append(df)
        except Exception:
            pass

    if not frames:
        raise FileNotFoundError(f"No valid batch summary found in {batch_dir}")

    summary_df = pd.concat(frames, ignore_index=True).sort_values("run_index")

    # Only complete runs (>= 99 % of target duration)
    summary_df = summary_df[
        summary_df["final_time_seconds"] >= summary_df["duration_s"] * 0.99
    ].reset_index(drop=True)

    result: list[tuple[pd.DataFrame, int]] = []
    for _, row in summary_df.iterrows():
        run_idx = int(row["run_index"])

        # Absolute path from summary first, then local fallback
        vehicle_csv: Path | None = None
        summary_path = Path(row.get("vehicle_csv", ""))
        if summary_path.is_file():
            vehicle_csv = summary_path
        else:
            run_dir = batch_dir / f"run_{run_idx:02d}"
            vehicle_csv = find_vehicle_csv(run_dir)

        if vehicle_csv is None:
            print(
                f"  Warning: no vehicle CSV for "
                f"{CTRL_DISPLAY[group_name]} N={n_vehicles} run {run_idx:02d}, skipping"
            )
            continue

        try:
            df = load_vehicle_csv(vehicle_csv)
        except Exception as exc:
            print(f"  Warning: could not load {vehicle_csv}: {exc}")
            continue

        n_veh = df["vehicle_id"].nunique()
        t_range = f"t={df[X_COL].min():.0f}..{df[X_COL].max():.0f} s"
        print(
            f"  [{CTRL_DISPLAY[group_name]} N={n_vehicles} | run {run_idx:02d}] "
            f"{vehicle_csv.name} — {n_veh} vehicles, {t_range}"
        )
        result.append((df, run_idx))

    return result


# ---------- Run alignment ---------------------------------------------------

def align_run_counts(
    group_runs: dict[tuple[str, int], list[tuple[pd.DataFrame, int]]]
) -> dict[tuple[str, int], list[tuple[pd.DataFrame, int]]]:
    """For each N value keep only run indices present in *all* controllers."""
    n_values = {n for _, n in group_runs}
    for n in n_values:
        keys = [(g, n) for (g, nv) in group_runs if nv == n]
        if len(keys) < 2:
            continue
        run_sets = [set(ri for _, ri in group_runs[k]) for k in keys]
        common = set.intersection(*run_sets)
        for k in keys:
            excluded = [ri for _, ri in group_runs[k] if ri not in common]
            for ri in excluded:
                print(
                    f"  Align: excluding {k[0]} N={k[1]} run {ri:02d} "
                    f"(not present in all controllers)"
                )
            group_runs[k] = [(df, ri) for df, ri in group_runs[k] if ri in common]
    return group_runs


# ---------- Vehicle extraction ---------------------------------------------

def ordered_vehicle_ids(df: pd.DataFrame) -> list:
    return list(dict.fromkeys(df["vehicle_id"].tolist()))


def extract_vehicle(df: pd.DataFrame, vehicle_id) -> pd.DataFrame:
    return df[df["vehicle_id"] == vehicle_id].sort_values(X_COL).reset_index(drop=True)


def build_vehicle_runs(
    runs: list[tuple[pd.DataFrame, int]], veh_idx: int
) -> list[pd.DataFrame]:
    """Return one df per run for the vehicle at spawn-position veh_idx."""
    result = []
    for df, _ in runs:
        ids = ordered_vehicle_ids(df)
        if veh_idx < len(ids):
            result.append(extract_vehicle(df, ids[veh_idx]))
    return result


# ---------- Plot utilities --------------------------------------------------

def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved {output_path}")


# ---------- Per-vehicle comparison plot ------------------------------------

def plot_vehicle_metric(
    group_data: dict[tuple[str, int], list[pd.DataFrame]],
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    show: bool,
    y_clip_pct: float | None = None,
) -> bool:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    plotted_any = False

    for (group_name, vehicle_count) in _sorted_keys(group_data):
        vdfs = group_data[(group_name, vehicle_count)]
        valid = [df for df in vdfs if df[metric_col].notna().any()]
        if not valid:
            continue

        color  = GROUP_N_COLORS[(group_name, vehicle_count)]
        ctrl   = CTRL_DISPLAY[group_name]
        n_runs = len(valid)
        label  = f"{ctrl}, N={vehicle_count} (n={n_runs})"
        plotted_any = True

        min_t   = min(df[X_COL].max() for df in valid)
        clipped = [df[df[X_COL] <= min_t].reset_index(drop=True) for df in valid]
        time_ref = clipped[0][X_COL].to_numpy()

        metric_matrix = np.vstack([
            np.interp(time_ref,
                      df[X_COL].to_numpy(),
                      df[metric_col].fillna(0).to_numpy())
            for df in clipped
        ])

        for df in clipped:
            ax.plot(df[X_COL], df[metric_col],
                    color=color, alpha=0.10, linewidth=0.8, zorder=2)

        mean_vals = metric_matrix.mean(axis=0)
        std_vals  = metric_matrix.std(axis=0)

        if n_runs > 1:
            ax.fill_between(
                time_ref, mean_vals - std_vals, mean_vals + std_vals,
                color=color, alpha=0.20, linewidth=0, zorder=3,
            )
        ax.plot(time_ref, mean_vals, color=color, linewidth=2.2, zorder=4,
                label=f"{label}  ±1σ band" if n_runs > 1 else label)

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_title(title, pad=10)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)

    if y_clip_pct is not None:
        all_vals = np.concatenate([
            df[metric_col].dropna().to_numpy()
            for vdfs in group_data.values()
            for df in vdfs
            if metric_col in df.columns
        ])
        if len(all_vals):
            lo = np.percentile(all_vals, 100.0 - y_clip_pct)
            hi = np.percentile(all_vals, y_clip_pct)
            margin = (hi - lo) * 0.05
            ax.set_ylim(lo - margin, hi + margin)

    # Legend: left column = Barrier Control, right column = Traffic Light
    all_h, all_l = ax.get_legend_handles_labels()
    bc_items = [(h, l) for h, l in zip(all_h, all_l) if "Barrier" in l]
    tl_items = [(h, l) for h, l in zip(all_h, all_l) if "Traffic" in l]
    other    = [(h, l) for h, l in zip(all_h, all_l)
                if "Barrier" not in l and "Traffic" not in l]
    paired   = [item for pair in zip(bc_items, tl_items) for item in pair]
    # Append any unpaired BC/TL items (if one controller has more N values)
    min_len  = min(len(bc_items), len(tl_items))
    paired  += bc_items[min_len:] + tl_items[min_len:] + other
    ax.legend(
        [h for h, _ in paired],
        [l for _, l in paired],
        loc="best",
        fontsize=9,
        ncol=2,
    )

    save_figure(fig, output_path)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)
    return True


# ---------- Barrier control firings — all vehicles, one plot per run -------

def plot_barrier_firings_per_run(
    bc_runs: list[tuple[pd.DataFrame, int]],
    group_label: str,
    output_dir: Path,
    show: bool,
) -> list[Path]:
    """One scatter plot per run: time vs vehicle index, coloured by u_barrier."""
    saved: list[Path] = []
    for df, run_idx in bc_runs:
        firings    = df[df["u_barrier"].notna() & (df["u_barrier"] != 0)].copy()
        n_vehicles = df["vehicle_id"].nunique()
        id_to_pos  = {v: i for i, v in enumerate(ordered_vehicle_ids(df))}

        fig, ax = plt.subplots(figsize=(14, max(6, n_vehicles * 0.22 + 2)))

        if not firings.empty:
            firings["vehicle_pos"] = firings["vehicle_id"].map(id_to_pos)
            sc = ax.scatter(
                firings["time_seconds"],
                firings["vehicle_pos"],
                c=firings["u_barrier"],
                cmap="RdBu_r",
                s=72,
                alpha=0.87,
                edgecolors="white",
                linewidths=0.4,
                vmin=-10,
                vmax=10,
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.80, aspect=22, pad=0.02)
            cbar.set_label("Barrier control signal [km/h]", fontsize=11)
            cbar.ax.tick_params(labelsize=9)
        else:
            ax.text(
                0.5, 0.5, "No barrier firings in this run",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#888888",
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Vehicle index")
        ax.set_title(
            f"Barrier Control Firings — {group_label} — "
            f"All {n_vehicles} Vehicles — Run {run_idx:02d}",
            pad=12,
        )
        ax.set_yticks(range(n_vehicles))
        ax.set_yticklabels([str(i) for i in range(n_vehicles)], fontsize=8)
        ax.set_xlim(0, df["time_seconds"].max())
        ax.set_ylim(-0.5, n_vehicles - 0.5)

        safe_label = sanitize_name(group_label)
        output_path = output_dir / f"barrier_firings_{safe_label}_run_{run_idx:02d}.png"
        save_figure(fig, output_path)
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)
        saved.append(output_path)

    return saved


# ---------- Main ------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all 6 (controller, N) batches
    group_runs: dict[tuple[str, int], list[tuple[pd.DataFrame, int]]] = {}
    for (group_name, n_vehicles), batch_dir in BATCHES.items():
        print(f"\n[{CTRL_DISPLAY[group_name]} N={n_vehicles}] Loading from {batch_dir.name}")
        try:
            runs = load_all_runs(group_name, n_vehicles, batch_dir)
        except FileNotFoundError as exc:
            print(f"  Skipping: {exc}")
            continue
        group_runs[(group_name, n_vehicles)] = runs
        if runs:
            n_veh = runs[0][0]["vehicle_id"].nunique()
            print(f"  → {len(runs)} complete run(s), {n_veh} vehicles in CSV")

    # Align run indices: for each N, keep only runs present in all controllers
    print("\n=== Aligning run indices across controllers ===")
    group_runs = align_run_counts(group_runs)

    print("\n=== Run counts per group (after alignment) ===")
    for key in _sorted_keys(group_runs):
        g, n = key
        run_indices = sorted(ri for _, ri in group_runs[key])
        print(f"  {CTRL_DISPLAY[g]} N={n}: {len(run_indices)} runs  {run_indices}")

    # Determine how many vehicles to plot across all groups
    max_available = min(
        (runs[0][0]["vehicle_id"].nunique() for runs in group_runs.values() if runs),
        default=0,
    )
    num_vehicles = min(args.num_vehicles, max_available)
    print(
        f"\nPlotting up to {num_vehicles} vehicles "
        f"(of {max_available} available in the smallest group) "
        f"across {len(group_runs)} (controller, N) group(s).\n"
    )

    saved: list[Path] = []

    for veh_idx in range(num_vehicles):
        veh_num = veh_idx + 1
        print(f"=== Vehicle {veh_num} ===")

        group_data: dict[tuple[str, int], list[pd.DataFrame]] = {}
        for key, runs in group_runs.items():
            vdfs = build_vehicle_runs(runs, veh_idx)
            g, n = key
            print(f"  [{CTRL_DISPLAY[g]} N={n}] {len(vdfs)} run(s)")
            if vdfs:
                group_data[key] = vdfs

        prefix   = f"vehicle_{veh_num:02d}"
        n_values = sorted({n for _, n in group_data})
        for col, ylabel, plot_title, y_clip in PLOT_COLS:
            for n in n_values:
                n_subset   = {k: v for k, v in group_data.items() if k[1] == n}
                full_title = f"{plot_title} — Vehicle {veh_num} — N={n}"
                out_path   = output_dir / f"{prefix}_{sanitize_name(col)}_N{n}.png"
                ok = plot_vehicle_metric(
                    n_subset, col, ylabel, full_title, out_path, args.show,
                    y_clip_pct=y_clip,
                )
                if ok:
                    saved.append(out_path)
                else:
                    print(f"  Skipping {col} N={n}: no valid data")

        print()

    # Barrier firing scatter plots — one per run for every BC group
    for (group_name, n_vehicles), runs in group_runs.items():
        if group_name != "Barriercontrol" or not runs:
            continue
        label = f"BC_N{n_vehicles}"
        print(f"=== Barrier Control Firings — N={n_vehicles}, {len(runs)} run(s) ===")
        for out in plot_barrier_firings_per_run(runs, label, output_dir, args.show):
            saved.append(out)

    if args.show:
        plt.show()

    print(f"\nDone. {len(saved)} plot(s) written to {output_dir}")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

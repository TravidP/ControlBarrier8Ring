"""
FiguresBoxplot.py — Grouped box plots, Traffic Light vs Barrier Control.

Produces:
  1. Run-level comparison (reference-image style): 2-panel figure,
     TL (hatched) vs BC (solid) across N=57/67/77, with % change
     and mean annotations, scatter jitter overlay.
  2. Per-vehicle box plots: for each N, one figure showing per-vehicle
     steady-state speed and gap-error distributions across all runs.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.patches as mpatches
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
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "legend.facecolor": "white",
    "legend.frameon": True,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "xtick.color": "#333333",
    "ytick.color": "#444444",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
})

# ---------- Constants -------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR / "simulation_results"

TL_DIRS: dict[int, Path] = {
    57: BASE_DIR / "TrafficlightResults" / "Batch_30runs_N=57 FAS",
    67: BASE_DIR / "TrafficlightResults" / "Batch_30runs_N=67 FAS",
    77: BASE_DIR / "TrafficlightResults" / "Batch_30runs_N=77 FAS",
}
BC_DIRS: dict[int, Path] = {
    57: BASE_DIR / "BarrierControlResults" / "Batch_30runs_N=57",
    67: BASE_DIR / "BarrierControlResults" / "Batch_30runs_N=67",
    77: BASE_DIR / "BarrierControlResults" / "Batch_30runs_N=77",
}

OUTPUT_DIR = BASE_DIR / "PlotsControllers" / "BoxplotsComparison"

N_VALUES = [57, 67, 77]

N_COLORS: dict[int, str] = {
    57: "#1E88E5",  # blue
    67: "#43A047",  # green
    77: "#E53935",  # red
}

TIME_COL = "time_seconds"
STEADY_STATE_START_S = 500.0
RNG = np.random.default_rng(42)

# Two-panel run-level KPIs: (column, y-label, lower_is_better)
PANEL_KPIS = [
    ("avg_speed_kmh",    "Average Speed (km/h)",  False),
    ("final_throughput", "Throughput (veh/s)",     False),
]

# Per-vehicle metrics to plot
VEH_METRICS = [
    ("velocity_kmh",        "Speed (km/h)"),
    ("headway_error_meters", "Gap error (m)"),
]

# Max vehicles to show in per-vehicle plot (keep readable)
MAX_VEH_PLOT = 30


# ---------- Colour helpers --------------------------------------------------

def _lighten(hex_color: str, amount: float = 0.50) -> str:
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return matplotlib.colors.to_hex(rgb + (1 - rgb) * amount)


def _darken(hex_color: str, factor: float = 0.72) -> str:
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return matplotlib.colors.to_hex(np.clip(rgb * factor, 0, 1))


# ---------- Data loading ----------------------------------------------------

def _load_summaries(batch_dir: Path) -> pd.DataFrame:
    files = sorted(batch_dir.glob("batch_summary_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["final_time_seconds"] >= df["duration_s"] * 0.99].reset_index(drop=True)


def load_run_data() -> dict[tuple[str, int], pd.DataFrame]:
    out: dict[tuple[str, int], pd.DataFrame] = {}
    for n in N_VALUES:
        for key, dirs in [("TL", TL_DIRS), ("BC", BC_DIRS)]:
            d = dirs.get(n)
            if d and d.is_dir():
                df = _load_summaries(d)
                if not df.empty:
                    out[(key, n)] = df
    return out


def _find_vehicle_csv(run_dir: Path) -> Path | None:
    for pat in ("vehicle_all*.csv", "vehicle_*.csv"):
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _load_vehicle_frames(batch_dir: Path, summary_df: pd.DataFrame) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for _, row in summary_df.iterrows():
        run_idx = int(row["run_index"])
        vcsv_path: Path | None = None
        raw = Path(str(row.get("vehicle_csv", "")))
        if raw.is_file():
            vcsv_path = raw
        else:
            run_dir = batch_dir / f"run_{run_idx:02d}"
            vcsv_path = _find_vehicle_csv(run_dir)
        if vcsv_path is None:
            continue
        try:
            vdf = pd.read_csv(vcsv_path)
            vdf = vdf[vdf[TIME_COL] >= STEADY_STATE_START_S]
            if not vdf.empty:
                frames.append(vdf)
        except Exception:
            continue
    return frames


# ---------- Single box drawing ----------------------------------------------

def _styled_boxplot(
    ax: plt.Axes,
    values: np.ndarray,
    pos: float,
    width: float,
    color: str,
    hatched: bool,
) -> float:
    """Draw one styled box. Returns the lower whisker y for label placement."""
    fill = _lighten(color, 0.52) if hatched else _lighten(color, 0.10)
    edge = _darken(color)

    bp = ax.boxplot(
        [values],
        positions=[pos],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#111111", linewidth=2.0, zorder=5),
        whiskerprops=dict(color=edge, linewidth=1.2),
        capprops=dict(color="none", linewidth=0),
        boxprops=dict(linewidth=1.3),
        manage_ticks=False,
        zorder=3,
    )
    box = bp["boxes"][0]
    box.set_facecolor(fill)
    box.set_edgecolor(edge)
    if hatched:
        box.set_hatch("///")
        box.set_linewidth(0.8)

    # Jitter scatter — larger dots so tight clusters are visible
    jx = RNG.normal(pos, width * 0.10, len(values))
    ax.scatter(jx, values, s=22, color=_darken(color, 0.85),
               alpha=0.60, linewidth=0, zorder=4)

    q1 = float(np.percentile(values, 25))
    iqr = float(np.percentile(values, 75)) - q1
    lo_whisk = max(float(values.min()), q1 - 1.5 * iqr)
    return lo_whisk


# ---------- Run-level comparison plot  --------------------------------------

def plot_run_comparison(
    axes: list[plt.Axes],
    run_data: dict[tuple[str, int], pd.DataFrame],
    kpis: list[tuple[str, str, bool]],
    n_values: list[int],
) -> None:
    box_w    = 0.30
    gap      = 0.10
    grp_step = 1.8
    centers  = [i * grp_step for i in range(len(n_values))]

    for ax, (col, ylabel, lower_is_better) in zip(axes, kpis):
        all_vals: list[float] = []

        for cx, n in zip(centers, n_values):
            color = N_COLORS[n]
            pos_tl = cx - box_w / 2 - gap / 2
            pos_bc = cx + box_w / 2 + gap / 2

            tl_df = run_data.get(("TL", n))
            bc_df = run_data.get(("BC", n))
            tl_v = tl_df[col].dropna().to_numpy() if (tl_df is not None and col in tl_df) else np.array([])
            bc_v = bc_df[col].dropna().to_numpy() if (bc_df is not None and col in bc_df) else np.array([])

            lo_tl = lo_bc = None
            if len(tl_v) >= 2:
                lo_tl = _styled_boxplot(ax, tl_v, pos_tl, box_w, color, hatched=True)
                all_vals.extend(tl_v.tolist())
            if len(bc_v) >= 2:
                lo_bc = _styled_boxplot(ax, bc_v, pos_bc, box_w, color, hatched=False)
                all_vals.extend(bc_v.tolist())

            # Mean labels below each box's lower whisker
            if lo_tl is not None:
                ax.text(pos_tl, lo_tl, f"{float(np.mean(tl_v)):.2f}",
                        ha="center", va="top", fontsize=7.5, fontweight="bold",
                        color="#333333", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.80))
            if lo_bc is not None:
                ax.text(pos_bc, lo_bc, f"{float(np.mean(bc_v)):.2f}",
                        ha="center", va="top", fontsize=7.5, fontweight="bold",
                        color="#333333", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.80))

            # % change annotation above the pair
            if len(tl_v) and len(bc_v):
                tl_mean = float(np.mean(tl_v))
                bc_mean = float(np.mean(bc_v))
                pct = (bc_mean - tl_mean) / abs(tl_mean) * 100.0
                improved = (pct > 0) != lower_is_better
                color_pct = "#2E7D32" if improved else "#C62828"
                sign = "+" if pct > 0 else ""

                hi = max(
                    np.percentile(tl_v, 100), np.percentile(bc_v, 100),
                )
                ax.text(
                    cx, hi,
                    f"{sign}{pct:.1f}%",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                    color=color_pct, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="#CCCCCC", linewidth=0.7, alpha=0.93),
                )

        # Dashed group separators
        for i in range(len(n_values) - 1):
            sep = (centers[i] + centers[i + 1]) / 2
            ax.axvline(sep, color="#BBBBBB", linewidth=0.8,
                       linestyle="--", alpha=0.7, zorder=1)

        ax.set_xticks(centers)
        ax.set_xticklabels([f"N={n}" for n in n_values])
        ax.set_xlim(centers[0] - grp_step * 0.65, centers[-1] + grp_step * 0.65)
        ax.set_ylabel(ylabel)

        # Zoom y-axis to data + headroom for pct labels
        if all_vals:
            vmin, vmax = min(all_vals), max(all_vals)
            span = vmax - vmin if vmax != vmin else abs(vmax) * 0.05 + 0.01
            ax.set_ylim(vmin - span * 0.10, vmax + span * 0.28)


def make_run_comparison_figure(
    run_data: dict[tuple[str, int], pd.DataFrame],
    n_values: list[int] = N_VALUES,
) -> plt.Figure:
    n_panels = len(PANEL_KPIS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4.8 * n_panels))
    if n_panels == 1:
        axes = [axes]

    plot_run_comparison(list(axes), run_data, PANEL_KPIS, n_values)

    # Shared legend at top
    tl_patch = mpatches.Patch(
        facecolor="#D6E4F7", edgecolor="#6EA8DE",
        hatch="///", linewidth=0.8, label="Traffic Light (baseline)",
    )
    bc_patch = mpatches.Patch(
        facecolor="#888888", edgecolor="#555555",
        linewidth=0.8, label="Barrier Control",
    )
    fig.legend(
        handles=[tl_patch, bc_patch],
        loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=2, frameon=True, fontsize=10,
    )
    fig.suptitle(
        "Traffic Light vs Barrier Control — Run-Level KPI Distributions",
        fontsize=12, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    return fig


# ---------- Per-vehicle box plots -------------------------------------------

def _ordered_veh_ids(vdf: pd.DataFrame) -> list:
    return list(dict.fromkeys(vdf["vehicle_id"].tolist()))


def _per_vehicle_timeseries(
    run_frames: list[pd.DataFrame],
    col: str,
    max_veh: int,
    subsample: int = 5,
) -> dict[int, np.ndarray]:
    """Collect full steady-state time-series values per vehicle spawn position.

    Returns {position: array_of_values} built from every ``subsample``-th
    sample across all runs — enough for accurate box statistics and diverse
    scatter points while keeping memory low.
    """
    pos_lists: dict[int, list[float]] = {}
    for vdf in run_frames:
        if col not in vdf.columns:
            continue
        ids = _ordered_veh_ids(vdf)
        for pos, vid in enumerate(ids[:max_veh]):
            grp = vdf[vdf["vehicle_id"] == vid][col].dropna()
            if not grp.empty:
                pos_lists.setdefault(pos, []).extend(
                    grp.iloc[::subsample].tolist()
                )
    return {pos: np.array(v) for pos, v in pos_lists.items()}


def plot_per_vehicle_panel(
    ax: plt.Axes,
    tl_pv: dict[int, np.ndarray],
    bc_pv: dict[int, np.ndarray],
    color: str,
    ylabel: str,
) -> bool:
    positions = sorted(set(tl_pv) | set(bc_pv))
    if not positions:
        return False

    box_w = 0.28
    gap   = 0.08
    xs    = list(range(len(positions)))
    pos_map = {p: x for x, p in zip(xs, positions)}

    all_vals: list[float] = []

    max_scatter = 60
    for veh_pos, x in pos_map.items():
        for v, pos_x, hatched in [
            (tl_pv.get(veh_pos, np.array([])), x - box_w / 2 - gap / 2, True),
            (bc_pv.get(veh_pos, np.array([])), x + box_w / 2 + gap / 2, False),
        ]:
            if len(v) < 4:
                continue
            _styled_boxplot_novlabel(ax, v, pos_x, box_w, color, hatched)
            n_sc = min(max_scatter, len(v))
            sample = RNG.choice(v, size=n_sc, replace=False)
            jx = RNG.normal(pos_x, box_w * 0.08, n_sc)
            ax.scatter(jx, sample, s=5, color=_darken(color, 0.80),
                       alpha=0.30, linewidth=0, zorder=4)
            all_vals.extend(v.tolist())

    # Every 5th tick label to avoid crowding
    tick_xs     = [x for x, p in zip(xs, positions) if p % 5 == 0]
    tick_labels = [str(p) for p in positions if p % 5 == 0]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlim(-0.6, len(positions) - 0.4)
    ax.set_xlabel("Vehicle spawn index")
    ax.set_ylabel(ylabel)

    if all_vals:
        plo, phi = np.percentile(all_vals, [1, 99])
        span = phi - plo if phi != plo else abs(phi) * 0.05 + 0.1
        ax.set_ylim(plo - span * 0.08, phi + span * 0.15)
    return True


def _styled_boxplot_novlabel(
    ax: plt.Axes,
    values: np.ndarray,
    pos: float,
    width: float,
    color: str,
    hatched: bool,
) -> None:
    """Boxplot without mean label (used in per-vehicle plots)."""
    fill = _lighten(color, 0.52) if hatched else _lighten(color, 0.10)
    edge = _darken(color)

    bp = ax.boxplot(
        [values],
        positions=[pos],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#111111", linewidth=1.5, zorder=5),
        whiskerprops=dict(color=edge, linewidth=0.9),
        capprops=dict(color="none", linewidth=0),
        boxprops=dict(linewidth=1.0),
        manage_ticks=False,
        zorder=3,
    )
    box = bp["boxes"][0]
    box.set_facecolor(fill)
    box.set_edgecolor(edge)
    if hatched:
        box.set_hatch("//")
        box.set_linewidth(0.6)


def make_per_vehicle_figure(
    n: int,
    tl_frames: list[pd.DataFrame],
    bc_frames: list[pd.DataFrame],
) -> plt.Figure | None:
    color = N_COLORS[n]
    fig_w = max(14, min(n, MAX_VEH_PLOT) // 2 + 8)
    fig, axes = plt.subplots(len(VEH_METRICS), 1, figsize=(fig_w, 4.5 * len(VEH_METRICS)))
    if len(VEH_METRICS) == 1:
        axes = [axes]

    any_data = False
    for ax, (col, ylabel) in zip(axes, VEH_METRICS):
        tl_pv = _per_vehicle_timeseries(tl_frames, col, MAX_VEH_PLOT)
        bc_pv = _per_vehicle_timeseries(bc_frames, col, MAX_VEH_PLOT)
        if plot_per_vehicle_panel(ax, tl_pv, bc_pv, color, ylabel):
            any_data = True

    if not any_data:
        plt.close(fig)
        return None

    tl_patch = mpatches.Patch(
        facecolor=_lighten(color, 0.52), edgecolor=_darken(color),
        hatch="//", linewidth=0.6, label="Traffic Light (baseline)",
    )
    bc_patch = mpatches.Patch(
        facecolor=_lighten(color, 0.10), edgecolor=_darken(color),
        linewidth=0.8, label="Barrier Control",
    )
    fig.legend(
        handles=[tl_patch, bc_patch],
        loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=2, frameon=True, fontsize=10,
    )
    fig.suptitle(
        f"Per-Vehicle KPI Distributions — N={n}  "
        f"(TL: {len(tl_frames)} runs, BC: {len(bc_frames)} runs, "
        f"first {min(n, MAX_VEH_PLOT)} vehicles)",
        fontsize=12, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    return fig


# ---------- Save helper -----------------------------------------------------

def save_fig(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------- Main ------------------------------------------------------------

def main() -> None:
    print("Loading run-level summary data …")
    run_data = load_run_data()
    for key, df in sorted(run_data.items()):
        print(f"  {key[0]} N={key[1]}: {len(df)} runs")

    print("\nGenerating run-level comparison box plots …")
    fig_run = make_run_comparison_figure(run_data)
    save_fig(fig_run, "comparison_boxplots_run_level.png")

    print("\nLoading vehicle-level data for per-vehicle box plots …")
    for n in N_VALUES:
        tl_summary = run_data.get(("TL", n), pd.DataFrame())
        bc_summary = run_data.get(("BC", n), pd.DataFrame())

        if tl_summary.empty and bc_summary.empty:
            print(f"  N={n}: no data, skipping")
            continue

        tl_frames = _load_vehicle_frames(TL_DIRS[n], tl_summary) if not tl_summary.empty else []
        bc_frames = _load_vehicle_frames(BC_DIRS[n], bc_summary) if not bc_summary.empty else []
        print(f"  N={n}: {len(tl_frames)} TL run frames, {len(bc_frames)} BC run frames")

        fig_veh = make_per_vehicle_figure(n, tl_frames, bc_frames)
        if fig_veh is not None:
            save_fig(fig_veh, f"comparison_boxplots_per_vehicle_N{n}.png")
        else:
            print(f"  N={n}: no vehicle data found")

    print("\nDone.")


if __name__ == "__main__":
    main()

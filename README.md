# Figure-8 Traffic-Control Experiments for CARLA

This directory contains the local research code and generated data for
figure-eight traffic-control experiments in CARLA. The main comparison is
between distributed barrier/headway control and FAS/adaptive traffic-light
control. It also contains earlier controller variants, map visualizers, and
analysis scripts used while the experiments evolved.

## CARLA provenance

The project was separated from CARLA's official `PythonAPI/examples` directory
by comparing the checkout with the official `ue4-dev` branch:

- Local CARLA commit: [`56c4bca6dd4d49c07e4968ca04698b619a3942a3`](https://github.com/carla-simulator/carla/tree/56c4bca6dd4d49c07e4968ca04698b619a3942a3/PythonAPI/examples), dated 2026-04-28.
- Online `ue4-dev` commit at the time of the audit: `7e951b372daddfd274408ed19025e6ab4b13133d`, dated 2026-07-06.
- [Comparison between those commits](https://github.com/carla-simulator/carla/compare/56c4bca6dd4d49c07e4968ca04698b619a3942a3...7e951b372daddfd274408ed19025e6ab4b13133d).

The local checkout is six commits behind that online revision. Only three
official files under `PythonAPI/examples` changed, all in `ros2/`; no official
example file was added or removed. The original 19 research Python files
documented below and `simulation_results/` are absent from the official tree.
`launch_carla_and_script.py` was added afterward as a local project utility.

`git describe` reports `0.9.15-560-g56c4bca6`, while the checkout's Python
package metadata is on the 0.9.16 line. This is therefore best described as an
official UE4 development-branch commit, not an exact CARLA release tag.

## Directory layout

```text
figure8_project/
├── launch_carla_and_script.py          # Start CARLA and a selected script
├── 8ring_*.py                         # Figure-eight simulation controllers
├── randommap_*.py                     # Generated-centerline experiment
├── Figures*.py, Plotting.py           # Analysis and plotting tools
├── ExportPerformanceExcel.py          # Spreadsheet export
├── extract_and_plot_simulation_results.py
├── publication_quality_traffic_plots.py
├── Map.py, MapPlotter.py, MappingXODR.py
├── simulation_results/                # Local data; ignored by Git
│   ├── BarrierControlResults/
│   ├── TrafficlightResults/
│   └── PlotsControllers/
└── results/                            # Output of the earliest experiment, if run
```

The scripts intentionally remain in one flat directory. Several output paths
are based on the script location, so the flat layout keeps those relationships
simple and preserves existing command names.

## Simulation and control scripts

| Script | Purpose |
| --- | --- |
| `8ring_real_distance.py` | Early/manual figure-eight experiment with an arc-length mapper, speed PID controller, and radar callback. Writes `results/vehicle_speeds_*.csv`. |
| `8ring_real_distance_Barrier.py` | Batched headway and intersection-barrier controller using the `MyFigure8` CARLA map. Writes vehicle, throughput, lap, and batch-summary data. |
| `8ring_real_distance_FAS.py` | Traffic-light/FAS counterpart using CROW-derived signal timing and headway PID control on `8ring`. |
| `8ring_real_distance_geodesic.py` | Barrier/headway variant on `8ring`, with steady-state summary and plotting bookkeeping. |
| `8ring_real_distance_geodesic_Binbin.py` | Compact barrier/headway variant that logs a wide per-agent state CSV. |
| `8ring_real_distance_geodesic_lissajous.py` | Experimental 3:1 Lissajous route with two self-intersections, barrier diagnostics, plots, and CARLA recovery logic. |
| `8ring_real_distance_trafficlight_adaptive_simulation2.py` | Expanded four-state adaptive CROW traffic-light controller with energy, safety, EV-debug metrics, 30-run batching, and server recovery. |
| `randommap_real_distance_geodesic_Binbin.py` | Compact barrier experiment on a generated piecewise centerline of roughly 650 m; it still loads the `8ring` CARLA map. |

The current default run count in several experimental variants is one. The
expanded adaptive traffic-light script defaults to 30 runs. Always inspect
`--help` and the constants near the top of a script before starting a batch.

## Analysis and export scripts

| Script | Purpose and current status |
| --- | --- |
| `publication_quality_traffic_plots.py` | Current analysis for the final N=51/61/71/81 traffic-light and barrier datasets. Produces representative traces, KPI comparisons and distributions, barrier statistics, and CSV/XLSX exports. |
| `extract_and_plot_simulation_results.py` | Generic multi-batch curve, boxplot, and 95% confidence-interval exporter. Its default discovery names predate the `_final` folders, so pass/configure the desired batches for current data. |
| `Plotting.py` | Legacy discovery-based controller comparison for N=37/47/57, including Welch tests, Cohen's d, CSV tables, and LaTeX output. |
| `FiguresPlot.py` | Legacy fixed-path barrier plots for N=57/67/77; the corresponding data now live under `BarrierControlResults/OLD RUNS/`. |
| `FiguresPlotAppealing.py` | Extended legacy plots with run distributions and a conflict-pair response view; it uses the same older N=57/67/77 layout. |
| `FiguresBoxplot.py` | Legacy traffic-light versus barrier grouped and per-vehicle boxplots for N=57/67/77; those batches are under `OLD RUNS/`. |
| `FiguresPlotVehicles.py` | Legacy N=37/47/57 per-vehicle time-series and barrier-firing plots. Its configured dated `Final Results` tree is not present in the current dataset. |
| `ExportPerformanceExcel.py` | Legacy styled three-sheet KPI workbook for N=37/47/57. It targets the same absent dated `Final Results` tree. |

Relocation-sensitive results roots in these tools are now derived from this
directory instead of `/home/rug/carla/PythonAPI/examples`. Historical batch
summaries were deliberately not rewritten; see [Historical paths](#historical-paths).

## Map tools

| Script | Purpose |
| --- | --- |
| `Map.py` | Basic analytical figure-eight preview using NumPy and Matplotlib. |
| `MapPlotter.py` | Preview of a deformed/dented figure-eight centerline. |
| `MappingXODR.py` | Live CARLA waypoint visualization with spectator and vehicle tracking. It includes a Windows fallback path for the CARLA Python API. |

## Environment and dependencies

The code was developed against this local layout:

- CARLA source tree: `/home/rug/carla`
- Unreal Engine 4.26 editor: `/home/rug/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor`
- CARLA Unreal project: `/home/rug/carla/Unreal/CarlaUE4/CarlaUE4.uproject`
- CARLA navigation agents: `/home/rug/carla/PythonAPI/carla`

Several simulation scripts contain those absolute paths. Update their
`CARLA_SERVER_CMD` or `AGENTS_PATH` constants when running on another machine.
`MappingXODR.py` also has a `C:/CARLA_0.9.15/PythonAPI/carla` import fallback;
it is only relevant when `import carla` is otherwise unavailable.

Python packages used across the project are:

- `carla`
- `numpy`
- `networkx`
- `scipy`
- `matplotlib`
- `pandas`
- `openpyxl`
- `seaborn` (optional; publication plots fall back when it is unavailable)

The CARLA server and Python client must use compatible API versions. The
standard-library modules used by the scripts do not require separate
installation.

## Maps and external assets

The simulation scripts expect imported CARLA maps to remain installed in the
CARLA content tree:

- `8ring` is used by most controller variants.
- `MyFigure8` is used by `8ring_real_distance_Barrier.py`.

Supporting source assets remain outside this directory at
`/home/rug/carla/MyFigure8.fbx` and `/home/rug/carla/MyFigure8.xodr`. Imported
map content also remains under CARLA's `Unreal/CarlaUE4/Content` tree, including
the `Trafficlight/Maps/8ring` and `Figure8` content. Do not move installed
`.umap` or `.uasset` files into this project: CARLA loads them from the content
tree.

The separate custom SUMO/JSSP files elsewhere in the CARLA checkout are not
referenced by these 19 scripts and are not a dependency of this directory.

## Running the code safely

Run commands from this directory so relative command-line paths behave
predictably. Use a dedicated CARLA instance with no other connected clients:
the simulation scripts enable synchronous mode and create and destroy actors.

### Recommended launcher

[`launch_carla_and_script.py`](launch_carla_and_script.py) is the recommended
entry point for running the simulations in this directory. It combines the two
commands that would otherwise have to be managed in separate terminals:

1. Start and prepare the CARLA server.
2. Start one selected Python experiment after CARLA is ready.

The launcher is configured for this computer's source-built CARLA installation
and uses these defaults:

| Setting | Default |
| --- | --- |
| Unreal executable | `/home/rug/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor` |
| CARLA project | `/home/rug/carla/Unreal/CarlaUE4/CarlaUE4.uproject` |
| RPC address | `127.0.0.1:2000` |
| Map | `MyFigure8` |
| Graphics | Vulkan, low quality, visible window |
| Server log | `figure8_project/carla_server.log` |
| Spectator | Centered top-down view with automatic height |

#### What happens when it runs

The launcher performs the following sequence:

1. It verifies that UE4Editor, `CarlaUE4.uproject`, the Python interpreter, and
   the selected experiment script exist.
2. It checks whether a CARLA server already answers on the configured RPC
   address.
3. If CARLA is not running, it starts UE4Editor with `-game`, the selected
   rendering backend, RPC port, and quality settings. UE4 output is redirected
   to `carla_server.log`.
4. It repeatedly calls the CARLA Python API until the server is ready or
   `--startup-timeout` expires. If UE4 exits early, the end of the server log is
   included in the error message.
5. It loads `MyFigure8` by default and confirms that CARLA returned the
   requested map.
6. It generates road waypoints, calculates the road network's minimum and
   maximum X/Y coordinates, and places the spectator above their geometric
   center. The requested pitch is `-90` degrees; UE4 normally reports about
   `-89` degrees after normalization.
7. It verifies the applied spectator location, height, and pitch before it
   starts the experiment.
8. It runs the selected script from `figure8_project`, forwarding all script
   arguments unchanged.
9. When the experiment exits, times out, or is interrupted, it stops the
   process group for the experiment and the CARLA process that it started.

If CARLA was already running, the launcher reuses it and never stops it. If
`--keep-carla` is supplied, a newly started CARLA server is also left running.

The launcher only accepts Python files directly inside `figure8_project`; it
will not execute a script from another directory. It does not change the
experiment's output location. Output continues to be controlled by each
experiment's defaults or its `--output-dir` option.

#### Command structure

Use this form:

```text
./launch_carla_and_script.py [LAUNCHER OPTIONS] SCRIPT [--] [SCRIPT OPTIONS]
```

Launcher options must appear before `SCRIPT`. Options after the script name
are forwarded to that experiment. The `--` separator is recommended because
it makes the boundary between launcher options and experiment options clear.

For example, in the following command `--offscreen` belongs to the launcher,
while `--number-of-vehicles` and `--duration-seconds` belong to the Barrier
experiment:

```bash
./launch_carla_and_script.py --offscreen \
  8ring_real_distance_Barrier.py -- \
  --number-of-vehicles 21 \
  --duration-seconds 60
```

#### Common commands

```bash
cd /home/rug/carla/PythonAPI/examples/figure8_project

# Show compatible CARLA client scripts and launcher options.
./launch_carla_and_script.py --list-scripts
./launch_carla_and_script.py --help

# Check the complete commands without launching anything.
./launch_carla_and_script.py --dry-run \
  8ring_real_distance_Barrier.py -- --number-of-vehicles 4

# Launch CARLA in a window, load MyFigure8, and run one short Barrier batch.
./launch_carla_and_script.py \
  8ring_real_distance_Barrier.py -- \
  --number-of-vehicles 21 \
  --duration-seconds 60 \
  --start-run 1 \
  --end-run 1

# Use --map auto for scripts whose CARLA_MAP_NAME should override MyFigure8.
# This adaptive traffic-light script selects 8ring.
./launch_carla_and_script.py --offscreen --no-sound --map auto \
  8ring_real_distance_trafficlight_adaptive_simulation2.py -- \
  --number-of-vehicles 21 \
  --duration-seconds 60 \
  --start-run 1 \
  --end-run 1
```

#### Launcher option reference

| Option | Meaning |
| --- | --- |
| `--map NAME` | Load a specific map. The default is `MyFigure8`. |
| `--map auto` | Read a literal `CARLA_MAP_NAME` from the experiment without importing or executing it. |
| `--map current` | Keep the server's currently loaded map. |
| `--spectator-height METERS` | Use a fixed height above the highest road point instead of automatic height. |
| `--no-top-down-view` | Do not reposition the spectator. |
| `--offscreen` | Run without a visible spectator window. The spectator transform is still configured internally. |
| `--quality-level Low\|Epic` | Select CARLA rendering quality. Low is the default. |
| `--opengl` | Use OpenGL instead of the default Vulkan renderer. |
| `--no-sound` | Disable audio. |
| `--startup-timeout SECONDS` | Change how long the launcher waits for the CARLA RPC server. |
| `--script-timeout SECONDS` | Stop the selected experiment after a time limit. Zero means no limit. |
| `--keep-carla` | Leave a CARLA process started by the launcher running after the experiment. |
| `--log PATH` | Change the CARLA server-log location. The file is replaced on each new launch. |
| `--python PATH` | Select the Python interpreter used to run the experiment. |
| `--editor PATH` | Override the UE4Editor executable. |
| `--carla-project PATH` | Override the `CarlaUE4.uproject` location. |
| `--carla-arg=-FLAG` | Forward an extra argument to UE4/CARLA. Repeat the option for multiple flags. |
| `--dry-run` | Validate paths and print both commands without starting a process. |
| `--list-scripts` | List project scripts that create a `carla.Client`. |

The project simulations currently connect to `127.0.0.1:2000` internally.
Although the launcher exposes `--host` and `--port` for its readiness check,
changing them also requires making the same change in the selected experiment.

#### Map selection and top-down view

The default map is `MyFigure8`. Use `--map auto` for an experiment such as the
adaptive traffic-light controller whose `CARLA_MAP_NAME` is `8ring`. Use
`--map current` to keep the loaded map or `--map NAME` to choose another map
explicitly.

The spectator center is calculated from the minimum and maximum road-waypoint
coordinates. The launcher requests a `-90` degree pitch (UE4 reports about
`-89` degrees after normalization), and the automatic height is scaled to the
map dimensions. This gives `MyFigure8` a centered overview of the complete road
network. Override it with `--spectator-height METERS`, or use
`--no-top-down-view` to leave the current spectator transform unchanged.

To see the top-down view in a CARLA window, do not use `--offscreen`. Offscreen
mode is intended for unattended batches and testing.

#### Exit behavior

The launcher returns the selected experiment's exit code. Launcher-specific
codes are:

| Code | Meaning |
| ---: | --- |
| `0` | Launcher action and experiment completed successfully. |
| `2` | Launcher validation, server startup, map loading, or camera setup failed. |
| `124` | The experiment exceeded `--script-timeout`. |
| `130` | The launcher was interrupted, normally with `Ctrl+C`. |

Only processes started by the launcher are stopped automatically. Note that
some experiments have their own recovery logic; the Lissajous and adaptive
traffic-light scripts can independently run `pkill -f CarlaUE4`.

#### Validation and troubleshooting

The launcher was validated end to end on this computer in offscreen mode: it
started the source-built server, matched client and server commit
`56c4bca6d-dirty`, loaded `MyFigure8`, spawned four vehicles, completed a
two-second Barrier run, wrote the summary/vehicle/throughput/lap CSVs to a
temporary directory, and stopped the CARLA process that it launched.

The centered camera was also verified against the live `MyFigure8` map. CARLA
reported a road span of 503.9 m and applied the spectator at approximately
`(-2.0, 0.0, 554.3)` with a pitch of `-89` degrees. The two-meter X offset is
UE4's normalized result and is within one percent of the map width.

If startup fails, inspect `carla_server.log`. This local UE4 build emits a
handled Vulkan vendor-ID ensure and unrelated `BP_BuildingConverterToCode`
compiler messages during startup; they did not prevent the tested server from
becoming ready. A real failure is reported by the launcher when UE4 exits or
the RPC timeout expires. Try `--opengl` for a Vulkan-specific startup failure,
or increase `--startup-timeout` on a slower machine.

### Manual invocation

```bash
cd /home/rug/carla/PythonAPI/examples/figure8_project

# Inspect options without starting a simulation.
python3 8ring_real_distance_Barrier.py --help
python3 8ring_real_distance_trafficlight_adaptive_simulation2.py --help
python3 publication_quality_traffic_plots.py --help

# Example short run; load MyFigure8 in CARLA before running this script.
python3 8ring_real_distance_Barrier.py \
  --number-of-vehicles 21 \
  --duration-seconds 60 \
  --start-run 1 \
  --end-run 1
```

Important safety behavior:

- `8ring_real_distance_Barrier.py` uses the world that is already loaded; it
  does not load `MyFigure8` for you.
- `8ring_real_distance_geodesic_lissajous.py` and
  `8ring_real_distance_trafficlight_adaptive_simulation2.py` can run
  `pkill -f CarlaUE4` and restart the hard-coded UE4 editor during recovery.
  That internal behavior also applies when they are started through the
  launcher.
- The Barrier, FAS, geodesic, Lissajous, and adaptive traffic-light batch
  drivers delete a partial `run_NN` output directory after a failed run.
- `MappingXODR.py` destroys every existing `vehicle.*` actor in the connected
  world before it spawns its tracker vehicle.
- `FiguresPlot.py`, `FiguresPlotAppealing.py`, `Map.py`, and `MapPlotter.py`
  execute plotting code at import time. Run them as standalone scripts rather
  than importing them as utility modules.
- Do not point `--output-dir` at a directory containing unrelated files.
- Full simulations were not run as part of the reorganization; validation was
  intentionally limited to syntax, help output, and read-only data discovery.

## Results and data formats

At the time of organization, `simulation_results/` contains 3,218 files with
86,086,665,439 logical bytes (86.1 GB or 80.2 GiB; approximately 81 GiB is
allocated on disk):

| Area | Approximate size | Files | Contents |
| --- | ---: | ---: | --- |
| `BarrierControlResults/` | 39 GiB | 1,744 | Barrier-controller batches and archived runs. |
| `TrafficlightResults/` | 42 GiB | 1,159 | FAS/adaptive traffic-light batches and archived runs. |
| `PlotsControllers/` | 71 MiB | 315 | Generated plots, tables, and publication outputs. |

The eight final folders cover both controllers at N=51, 61, 71, and 81. Each
final folder has 30 `run_NN` directories, for 240 final run directories in
total. The whole results tree contains 2,373 CSV, 775 PNG, 68 PDF, and 2 XLSX
files. `OLD RUNS/` retains approximately 22 GiB of earlier experiments, and
multiple summary files in some batches record resumed runs.

A typical batch is arranged as follows:

```text
Batch_<controller>_N=<demand>_final/
├── batch_summary_<timestamp>.csv
├── run_01/
│   ├── vehicle_all*.csv
│   ├── throughput_*.csv
│   ├── lap_times_*.csv
│   ├── ev_debug_*.csv        # present in relevant runs
│   ├── gap_plot.png          # present in relevant runs
│   └── speed_plot.png        # present in relevant runs
└── run_30/
```

Core file types:

- `batch_summary_*.csv`: one row per run with configuration, output paths, and
  final KPIs such as throughput, speed, energy, distance, delay, crossings,
  safety violations, and steady-state time.
- `vehicle_all*.csv`: per-time/per-vehicle kinematics, acceleration, gap and
  target gap, controller actions, blueprint, and power.
- `throughput_*.csv`: aggregate time series for throughput, average speed,
  energy, distance, safety, delay, and controller activity.
- `lap_times_*.csv`: vehicle/lap timing, spawn information, duration, and delay.
- `ev_debug_*.csv`: detailed force, motor speed/torque, efficiency, battery
  power, and accumulated energy for a selected electric vehicle.

## Historical paths

Fifty-three historical batch-summary CSVs store absolute paths beginning with
the former location:

```text
/home/rug/carla/PythonAPI/examples/simulation_results
```

Those research records remain byte-for-byte unchanged. The analysis scripts
that consume their `vehicle_csv`, `throughput_csv`, or `lap_csv` columns first
try the recorded path and then fall back to the matching file inside the local
`batch/run_NN` directory. Current publication-data discovery was verified for
all eight N=51/61/71/81 controller/demand combinations after relocation.

## Git and storage policy

`simulation_results/`, `results/`, logs, and Python caches are ignored by the
project `.gitignore`. The result tree is much too large for a normal GitHub
repository, and 311 individual files exceed 100 MB. Keep source code and this
README in Git, but use backed-up research storage, an artifact repository, or
another large-dataset system for the raw results.

The ignored results still exist locally; `.gitignore` only prevents accidental
Git staging. Before deleting or moving any batch, verify that another copy of
the research data exists.

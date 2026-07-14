#!/usr/bin/env python
# Copyright (c) 2025 Computer Vision Center (CVC)
# Figure-8 Simulation: Distance-Based Control with Traffic Light

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import carla
import networkx as nx
import numpy as np
from scipy.interpolate import RegularGridInterpolator

CARLA_SERVER_CMD = [
    '/home/rug/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor',
    '/home/rug/carla/Unreal/CarlaUE4/CarlaUE4.uproject',
    '-vulkan',
]
CARLA_MAP_NAME = '8ring'
CARLA_START_WAIT_S = 180


# ==========================================
# 0. SETUP PATHS
# ==========================================
AGENTS_PATH = '/home/rug/carla/PythonAPI/carla'
if os.path.exists(AGENTS_PATH):
    sys.path.append(AGENTS_PATH)

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    print("Agent module not found. Please ensure CARLA PythonAPI is correctly installed.")
    sys.exit(1)

# 1. MONKEY PATCH (Graph Build)

def patched_build_graph(self):
    self._graph = nx.DiGraph()
    self._road_id_to_edge = {}
    self._lane_id_to_edge = {}

    if hasattr(self, '_wmap'): topology = self._wmap.get_topology()
    else: topology = self._map.get_topology()

    for segment in topology:
        entry_wp, exit_wp = segment
        vec = exit_wp.transform.location - entry_wp.transform.location
        length = math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)
        if length > 0.001: net_carla_vector = carla.Vector3D(vec.x/length, vec.y/length, vec.z/length)
        else: net_carla_vector = carla.Vector3D(0, 0, 0)
        self._graph.add_node(entry_wp.id, vertex=entry_wp)
        self._graph.add_node(exit_wp.id, vertex=exit_wp)
        edge = {"entry_waypoint": entry_wp, "exit_waypoint": exit_wp, "entry_vector": net_carla_vector, "length": length, "path": []}
        self._graph.add_edge(entry_wp.id, exit_wp.id, **edge)
        if entry_wp.road_id not in self._road_id_to_edge: self._road_id_to_edge[entry_wp.road_id] = {}
        if entry_wp.section_id not in self._road_id_to_edge[entry_wp.road_id]: self._road_id_to_edge[entry_wp.road_id][entry_wp.section_id] = {}
        self._road_id_to_edge[entry_wp.road_id][entry_wp.section_id][entry_wp.lane_id] = edge
        self._lane_id_to_edge[entry_wp.id] = edge

GlobalRoutePlanner._build_graph = patched_build_graph

# 2. HELPER CLASSES

class DistanceMapper:
    """
    Handles the mapping between parametric t (radians) and physical distance S (meters).
    Now uses REVERSED math (-R*sin) to match natural car orientation.
    """
    def __init__(self, R, resolution=100000):
        self.R = R
        self.t_vals = np.linspace(0, 2 * np.pi, resolution)

        # --- REVERSE DIRECTION MATH ---
        # We invert the sine waves here: -R*sin(t)
        # This makes the "math path" flow in the same direction as the "spawn heading"
        x_vals = -self.R * np.sin(self.t_vals)
        y_vals = -self.R * np.sin(2 * self.t_vals)

        # Compute stepwise distances
        dists = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)

        # Cumulative sum to get S at each t (starting at 0)
        self.s_vals = np.concatenate(([0], np.cumsum(dists)))
        self.total_length = self.s_vals[-1]

        # Find distance at the intersection point (t = pi)
        idx_pi = np.argmin(np.abs(self.t_vals - np.pi))
        self.dist_at_pi = self.s_vals[idx_pi]

    def get_s_from_t(self, t):
        """Linearly interpolate distance S from parameter t."""
        t = t % (2 * np.pi)
        return np.interp(t, self.t_vals, self.s_vals)

    def get_t_from_s(self, s):
        """Linearly interpolate parameter t from distance S."""
        s = s % self.total_length
        return np.interp(s, self.s_vals, self.t_vals)


class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, integral_limit=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.prev_error = 0
        self.integral = 0

    def run(self, target_speed, current_speed, dt, derivative_input=None):
        error = target_speed - current_speed
        self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        if derivative_input is None:
            derivative = (error - self.prev_error) / dt if dt > 0 else 0
        else:
            derivative = derivative_input
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.prev_error = 0
        self.integral = 0


class HeadwayPIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, integral_limit=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_limit = integral_limit
        self.prev_error = 0
        self.integral = 0

    def run(self, current_gap, target_gap, dt, derivative_input=None):
        error = current_gap - target_gap
        self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        if derivative_input is None:
            derivative = (error - self.prev_error) / dt if dt > 0 else 0
        else:
            derivative = derivative_input
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.prev_error = 0
        self.integral = 0


class Figure8Tracker:
    def __init__(self, R, t_init):
        self.R = R
        self.t = t_init
        self._prev_loc = None

    def update_t(self, location):
        if self._prev_loc is not None:
            moved = math.sqrt(
                (location.x - self._prev_loc.x) ** 2
                + (location.y - self._prev_loc.y) ** 2
            )
            window = max(0.02, 3.0 * moved / self.R)
        else:
            window = 0.3
        self._prev_loc = location

        t_search = np.linspace(self.t - window, self.t + window, 1000)
        t_wrapped = t_search % (2 * np.pi)

        # --- REVERSE DIRECTION MATH ---
        dx = location.x - (-self.R * np.sin(t_wrapped))
        dy = location.y - (-self.R * np.sin(2 * t_wrapped))

        d2 = dx**2 + dy**2
        best_idx = np.argmin(d2)
        self.t = t_search[best_idx]           # store unwrapped
        return self.t % (2 * np.pi)           # callers expect [0, 2π)

    def get_math_location(self, t_val):
        # --- REVERSE DIRECTION MATH ---
        tx = -self.R * np.sin(t_val)
        ty = -self.R * np.sin(2 * t_val)
        return carla.Location(x=tx, y=ty, z=0.0)



def get_bb_distance(actor1, actor2):
    """Approximate gap between two vehicle bounding boxes (0 = touching)."""
    loc1 = actor1.get_location()
    loc2 = actor2.get_location()
    center_dist = loc1.distance(loc2)
    ext1 = actor1.bounding_box.extent
    ext2 = actor2.bounding_box.extent
    r1 = max(ext1.x, ext1.y)
    r2 = max(ext2.x, ext2.y)
    return max(0.0, center_dist - r1 - r2)


VEHICLE_CONFIGS = {
    'vehicle.tesla.model3': {
        'mass': 1931.0,
        'drag_coefficient': 0.219,
        'frontal_area': 2.22,
        'rolling_resistance': 0.010,
        'gear_ratio': 9.04,
        'wheel_radius': 0.3468,
        'gearbox_eff': 0.97,
    },
    'vehicle.audi.etron': {
        'mass': 2490.0,
        'drag_coefficient': 0.28,
        'frontal_area': 2.60,
        'rolling_resistance': 0.010,
        'gear_ratio': 9.205,
        'wheel_radius': 0.3815,
        'gearbox_eff': 0.97,
    },
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_results", "TrafficlightResults")
FIXED_RUN_COUNT = 30
RUN_SEED_BASE = 1000
KMH_PER_MS = 3.6


def build_vehicle_blueprint_sequence(num_vehicles, rng):
    tesla_count = (num_vehicles + 1) // 2
    etron_count = num_vehicles // 2
    blueprint_ids = (
        ['vehicle.tesla.model3'] * tesla_count
        + ['vehicle.audi.etron'] * etron_count
    )
    rng.shuffle(blueprint_ids)
    return blueprint_ids


def get_run_seed(run_index):
    return RUN_SEED_BASE + run_index


def update_ema(previous_value, value, alpha):
    """First-order low-pass filter; seeds from the first measured value."""
    if previous_value is None:
        return value
    return alpha * value + (1.0 - alpha) * previous_value


def create_vehicle_csv(output_dir=RESULTS_DIR, filename_prefix="vehicle"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "time_seconds",
        "vehicle_id",
        "blueprint_id",
        "velocity_ms",
        "velocity_kmh",
        "acceleration_ms2",
        "raw_acceleration_ms2",
        "gap_meters",
        "target_gap_meters",
        "headway_error_meters",
        "u_headway",
        "power_kw",
    ])
    return csv_file, writer, filename


def create_ev_debug_csv(output_dir=RESULTS_DIR, filename_prefix="ev_debug"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "time_s",
        "v_ms",
        "raw_accel_ms2",
        "F_inertia_N",
        "F_aero_N",
        "F_rolling_N",
        "F_total_N",
        "P_mech_W",
        "omega_motor_rads",
        "motor_rpm",
        "motor_torque_Nm",
        "motor_eta",
        "P_battery_W",
        "energy_vehicle_kWh",
    ])
    return csv_file, writer, filename


def create_throughput_csv(output_dir=RESULTS_DIR, filename_prefix="throughput"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "time_seconds",
        "throughput",
        "avg_speed_kmh",
        "Energy_kWh",
        "total_distance_km",
        "safety_violation_rate",
        "total_delay_s",
        "u_headway_avg",
        "steady_state",
    ])
    return csv_file, writer, filename


def create_lap_csv(output_dir=RESULTS_DIR, filename_prefix="lap_times"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["vehicle_id", "lap_number", "spawn_x", "spawn_y", "spawn_time_s",
                     "lap_start_time_s", "lap_end_time_s", "lap_duration_s", "delay_time_s"])
    return csv_file, writer, filename


def create_batch_summary_csv(output_dir, filename_prefix="batch_summary"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "run_index",
        "run_directory",
        "vehicles",
        "radius_m",
        "duration_s",
        "tracked_vehicle_ids",
        "throughput_csv",
        "vehicle_csv",
        "lap_csv",
        "final_time_seconds",
        "final_throughput",
        "avg_speed_kmh",
        "energy_kWh",
        "total_distance_km",
        "safety_violation_rate",
        "total_delay_s",
        "total_crossings",
        "total_safety_violations",
        "steady_state_time_s",
    ])
    return csv_file, writer, filename


def build_run_output_dir(base_output_dir, batch_label, run_index):
    return os.path.join(base_output_dir, batch_label, f"run_{run_index:02d}")


def write_batch_summary_row(writer, run_summary):
    writer.writerow([
        run_summary["run_index"],
        run_summary["run_directory"],
        run_summary["vehicles"],
        f"{run_summary['radius_m']:.2f}",
        f"{run_summary['duration_s']:.2f}",
        run_summary["tracked_vehicle_ids"],
        run_summary["throughput_csv"],
        run_summary["vehicle_csv"],
        run_summary["lap_csv"],
        f"{run_summary['final_time_seconds']:.2f}",
        f"{run_summary['final_throughput']:.6f}",
        f"{run_summary['avg_speed_kmh']:.2f}",
        f"{run_summary['energy_kWh']:.6f}",
        f"{run_summary['total_distance_km']:.4f}",
        f"{run_summary['safety_violation_rate']:.6f}",
        f"{run_summary['total_delay_s']:.2f}",
        run_summary["total_crossings"],
        run_summary["total_safety_violations"],
        f"{run_summary['steady_state_time_s']:.2f}",
    ])


# 3. MAIN SIMULATION

def run_simulation(args, run_index=1, run_output_dir=RESULTS_DIR):
    N = args.number_of_vehicles
    R = args.radius
    max_duration_s = args.duration_seconds
    run_seed = get_run_seed(run_index)
    rng = random.Random(run_seed)
    random.seed(run_seed)
    np.random.seed(run_seed)

    # Geometry and target spacing
    geom = DistanceMapper(R)
    TOTAL_TRACK_LENGTH = geom.total_length
    TARGET_GAP_METERS = (TOTAL_TRACK_LENGTH / N) if N > 0 else 0

    # Simulation timing
    DT = 0.05
    WARMUP_STEPS = 100
    RAMP_STEPS = 250
    RAMP_END_S = (WARMUP_STEPS + RAMP_STEPS) * DT
    TARGET_SPEED_KMH = 30.0
    COMFORTABLE_MAX_ACCEL_MS2 = 2.5  # limits commanded target-speed slew

    ACCEL_EMA_ALPHA = 0.5

    # Headway controller. Units: P -> km/h per meter of gap error,
    # D -> km/h per (m/s) of gap change rate.
    GAIN_HEADWAY_P = 3.50
    GAIN_HEADWAY_I = 0.00
    GAIN_HEADWAY_D = 1.00
    GAIN_INTEGRAL_LIMIT = 3.0
    MAX_HEADWAY_KMH = 10.0

  

    # Steady-state detection
    STEADY_STATE_WINDOW_S = 30.0       # rolling window length (seconds)
    STEADY_STATE_SPEED_STD_KMH = 0.5  # max fleet-speed std dev to declare steady state
    STEADY_STATE_SPEED_ERR_KMH = 2.0  # fleet mean must be within this of target speed

    # CROW Handboek Verkeerslichten-regelingen 2014 — standardized constants
    t_r      = 1.0    # perception-reaction time [s]
    a_af     = 2.5    # comfort deceleration [m/s²] (wet pavement baseline)
    t_weg    = 2.0    # queue start-up lost time [s]
    t_volg1  = 2.5    # discharge headway peak [s/vehicle]
    t_grens  = 4.0    # critical stream headway limit [s]
    L_vtg    = 5.0    # average vehicle length [m]
    d_tussen = 1.0    # buffer gap between stopped cars [m]
    t_ymax   = 1.0    # Max used yellow time (s)

    # Physical loop layout
    d      = 30.0   # stop-line to upstream edge of detection loop [m]
    L_loop = 5.0    # physical loop carpet length [m]

    # CROW equations — all phase durations derived from kinematics
    v0_ms          = TARGET_SPEED_KMH / KMH_PER_MS
    TL_YELLOW_TIME = math.ceil(t_r + v0_ms / (2.0 * a_af))                 # Eq A: t_gl [s]
    t_gg           = (d / (L_vtg + d_tussen)) * t_volg1  - t_ymax  # Eq B: fixed-green minimum [s]
    t_H            = t_grens - L_loop / v0_ms                               # Eq C: VAG2 gap limit [s]
    G_MAX          = 45.0

    # Vehicle-response constants
    STOP_LINE_DIST         = 12.0
    APPROACH_WINDOW_M      = 50.0
    MAX_DECEL_MS2          = a_af
    REACTION_TIME          = t_r

    # Lap timing
    LAP_MIN_TIME = 250.0
    MINIMAL_LAP_TIME = min(TOTAL_TRACK_LENGTH / (TARGET_SPEED_KMH / KMH_PER_MS), 281.78) # 281.78 found running simulation with N=1
    LAP_TRIGGER_RADIUS = 1.0
    LAP_CLEAR_RADIUS = 2.0

    # Safety distance violation parameter
    SAFETY_DISTANCE_M = 5.0

    # Queue following parameters
    VEHICLE_LENGTH = 5.0  # average of Tesla Model 3 and Audi e-tron lengths
    QUEUE_STOP_GAP = VEHICLE_LENGTH + 1.0
    QUEUE_SLOW_GAP = VEHICLE_LENGTH + 20.0

    # Green-discharge: seconds after TL release during which vehicles may ignore the
    # closing-gap headway penalty, allowing the queue to accelerate freely before
    # normal platoon spacing is enforced.  Duration matches fixed-green queue-clearance time.
    DISCHARGE_DURATION_S = t_gg

    # Energy calculation parameters
    GRAVITATIONAL_ACCELERATION = 9.81
    AIR_DENSITY = 1.225


    # Audi e-tron 55 asynchronous induction traction motor efficiency map
    # Source: Audi e-tron Motor Efficiency Map lookup table
    # Axes: RPM 0-15000 (1000 rpm steps), Torque 0-300 Nm (25 Nm steps)
    # Cells outside the power envelope (shown as — in the source table) filled with 0.0
    _eff_rpm_axis    = np.array([0., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000., 9000., 10000., 11000., 12000., 13000., 14000., 15000.])
    _eff_torque_axis = np.array([0., 25., 50., 75., 100., 125., 150., 175., 200., 225., 250., 275., 300.])
    _eff_values = np.array([
        # 0     25    50    75    100    125  150  175   200    225    250    275    300
        [0.80, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 0 RPM
        [0.93, 0.93, 0.92, 0.90, 0.88, 0.88, 0.86, 0.84, 0.0, 0.0, 0.0, 0.0, 0.0], # 1000 RPM
        [0.94, 0.95, 0.95, 0.94, 0.94, 0.93, 0.92, 0.92, 0.91, 0.90, 0.88, 0.88, 0.88], # 2000 RPM
        [0.95, 0.96, 0.96, 0.96, 0.95, 0.95, 0.94, 0.94, 0.93, 0.93, 0.93, 0.92, 0.92], # 3000 RPM
        [0.96, 0.97, 0.96, 0.96, 0.96, 0.96, 0.95, 0.95, 0.95, 0.94, 0.94, 0.93, 0.92], # 4000 RPM
        [0.96, 0.97, 0.97, 0.96, 0.96, 0.96, 0.96, 0.95, 0.95, 0.95, 0.94, 0.93, 0.92], # 5000 RPM
        [0.97, 0.97, 0.97, 0.97, 0.97, 0.96, 0.96, 0.96, 0.95, 0.95, 0.93, 0.92, 0.0], # 6000 RPM
        [0.97, 0.97, 0.97, 0.97, 0.97, 0.96, 0.96, 0.96, 0.95, 0.94, 0.93, 0.0, 0.0], # 7000 RPM
        [0.97, 0.97, 0.97, 0.97, 0.97, 0.96, 0.96, 0.96, 0.94, 0.93, 0.0, 0.0, 0.0], # 8000 RPM
        [0.97, 0.97, 0.97, 0.97, 0.97, 0.96, 0.95, 0.95, 0., 0.0, 0.0, 0.0, 0.0], # 9000 RPM
        [0.97, 0.98, 0.97, 0.97, 0.96, 0.96, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 10000 RPM
        [0.97, 0.98, 0.97, 0.97, 0.96, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 11000 RPM
        [0.97, 0.98, 0.97, 0.97, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 12000 RPM
        [0.97, 0.98, 0.97, 0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 13000 RPM
        [0.97, 0.98, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 14000 RPM
        [0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 15000 RPM
    ])
    audi_eff_map = RegularGridInterpolator(
        (_eff_rpm_axis, _eff_torque_axis),
        _eff_values,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    # Tesla Model 3 LR permanent magnet synchronous reluctance motor (PMSyRM) efficiency map
    # Source: Tesla Motor Efficiency Map lookup table
    # Axes: RPM 0-16000 (0, 500, 1500, then 1000-step to 16000), Torque 0-300 Nm (25 Nm steps)
    # Cells outside the power envelope (shown as — in the source table) filled with 0.0
    _t3_rpm_axis    = np.array([0., 1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000., 9000., 10000., 11000., 12000., 13000., 14000., 15000., 16000.])
    _t3_torque_axis = np.array([0., 25., 50., 75., 100., 125., 150., 175., 200., 225., 250., 275., 300.])
    _t3_eff_values  = np.array([
        #  0    25    50     75   100   125   150   175   200   225   250   275  300
        [0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60], # 0 RPM
        [0.88, 0.88, 0.88, 0.85, 0.85, 0.85, 0.85, 0.825, 0.80, 0.80, 0.80, 0.80, 0.70], # 1000 RPM
        [0.92, 0.92, 0.92, 0.90, 0.90, 0.90, 0.88, 0.88, 0.88, 0.85, 0.85, 0.85, 0.80], # 2000 RPM
        [0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.92, 0.92, 0.90, 0.90, 0.88, 0.88, 0.0], # 3000 RPM
        [0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.93, 0.93, 0.92, 0.90, 0.90, 0.0], # 4000 RPM
        [0.94, 0.94, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.94, 0.93, 0.93, 0.0, 0.0], # 5000 RPM
        [0.94, 0.95, 0.95, 0.96, 0.96, 0.96, 0.95, 0.95, 0.95, 0.95, 0.0, 0.0, 0.0], # 6000 RPM
        [0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0], # 7000 RPM
        [0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0], # 8000 RPM
        [0.94, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 9000 RPM
        [0.94, 0.96, 0.96, 0.96, 0.96, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 10000 RPM
        [0.96, 0.96, 0.97, 0.96, 0.96, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 11000 RPM
        [0.96, 0.97, 0.97, 0.96, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 12000 RPM
        [0.96, 0.97, 0.97, 0.96, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 13000 RPM
        [0.96, 0.97, 0.96, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 14000 RPM
        [0.96, 0.97, 0.97, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 15000 RPM
        [0.96, 0.97, 0.96, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 16000 RPM
    ])
    tesla_eff_map = RegularGridInterpolator(
        (_t3_rpm_axis, _t3_torque_axis),
        _t3_eff_values,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    MOTOR_EFF_MAPS = {
        'vehicle.tesla.model3': tesla_eff_map,
        'vehicle.audi.etron':   audi_eff_map,
    }

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    vehicles_list = []
    tl_actors = []

    throughput_csv_file = None
    throughput_csv_writer = None
    throughput_csv_filename = None

    vehicle_csv_file = None
    vehicle_csv_writer = None
    vehicle_csv_filename = None

    lap_csv_file = None
    lap_csv_writer = None
    lap_csv_filename = None

    ev_debug_csv_file = None
    ev_debug_csv_writer = None
    ev_debug_csv_filename = None

    tracked_vehicle_ids = []
    summary = None
    original_settings = None

    try:
        world = client.get_world()
        carla_map = world.get_map()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        # ── Traffic Light Actor Discovery ──────────────────────────────────────
        _center_wp = carla_map.get_waypoint(carla.Location(0.0, 0.0, 0.0), project_to_road=True)
        _center_loc = _center_wp.transform.location if _center_wp else carla.Location(0.0, 0.0, 0.0)

        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=_center_loc.x, y=_center_loc.y, z=100.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        ))

        _TL_SEARCH_RADIUS = 25.0
        tl_actors = [a for a in world.get_actors().filter('traffic.traffic_light')
                     if a.get_location().distance(_center_loc) < _TL_SEARCH_RADIUS]
        print(f"Found {len(tl_actors)} traffic light(s) within {_TL_SEARCH_RADIUS} m of the intersection.")

        def _tl_yaw(tl):
            wps = tl.get_affected_lane_waypoints()
            return wps[0].transform.rotation.yaw if wps else 0.0

        tl_group_1 = [tl for tl in tl_actors if abs(_tl_yaw(tl) % 180) >= 90]
        tl_group_2 = [tl for tl in tl_actors if abs(_tl_yaw(tl) % 180) < 90]
        print(f"  G1 actor IDs : {[a.id for a in tl_group_1]}")
        print(f"  G2 actor IDs : {[a.id for a in tl_group_2]}")

        # Map software state strings to CARLA enum values
        _STATE_MAP = {
            "green":  carla.TrafficLightState.Green,
            "yellow": carla.TrafficLightState.Yellow,
            "red":    carla.TrafficLightState.Red,
            "---":    carla.TrafficLightState.Red,
        }

        # Freeze all lights — we drive states manually every tick.
        for _tl in tl_actors:
            _tl.freeze(True)
            _tl.set_state(carla.TrafficLightState.Red)

        bp_lib = world.get_blueprint_library()
        vehicle_blueprint_ids = build_vehicle_blueprint_sequence(N, rng)

        _all_wps = carla_map.generate_waypoints(2.0)
        safe_waypoints = [w for w in _all_wps if not w.is_junction]
        safe_wp_xy = np.array([[w.transform.location.x, w.transform.location.y] for w in safe_waypoints])
        safe_wp_fwd = np.array([[w.transform.get_forward_vector().x, w.transform.get_forward_vector().y] for w in safe_waypoints])

        agent_ids = []
        agent_gap_pids = {}
        agent_pids = {}
        agent_trackers = {}
        spawn_times = {}
        vehicle_blueprints = {}
        vehicle_energy_params = {}
        vehicles_by_id = {}

        print(f"\n--- Run {run_index}: Spawning {N} EVs (Tesla Model 3 / Audi e-tron, Reverse Flow / Natural Heading) ---")
        print(f"Total Track Length: {TOTAL_TRACK_LENGTH:.1f} m")
        print(f"Target Gap: {TARGET_GAP_METERS:.1f} m")
        print(
            f"Headway PID: P={GAIN_HEADWAY_P:.3f}, I={GAIN_HEADWAY_I:.3f}, "
            f"D={GAIN_HEADWAY_D:.3f}, integral_limit={GAIN_INTEGRAL_LIMIT:.3f}, "
            f"cap=±{MAX_HEADWAY_KMH:.1f} km/h"
        )
        print(f"Run duration: {max_duration_s:.1f} s")
        print(f"Run seed: {run_seed}")

        # Spawning with noise
        for i in range(N):
            noise = rng.uniform(-0.20, 0.20) * TARGET_GAP_METERS
            s_target = (i * TARGET_GAP_METERS  + noise) % TOTAL_TRACK_LENGTH
            t_spawn = geom.get_t_from_s(s_target)

            start_x = -R * np.sin(t_spawn)
            start_y = -R * np.sin(2 * t_spawn)
            tang = np.array([-np.cos(t_spawn), -2.0 * np.cos(2.0 * t_spawn)])
            tang /= np.linalg.norm(tang)

            target_xy = np.array([start_x, start_y])
            dists = np.linalg.norm(safe_wp_xy - target_xy, axis=1)
            correct_loop = (safe_wp_fwd @ tang) > 0.8
            candidates = np.argsort(dists[correct_loop])
            candidates = np.where(correct_loop)[0][candidates]

            bp_id = vehicle_blueprint_ids[i]
            bp_veh = bp_lib.find(bp_id)
            if bp_veh.has_attribute('color'):
                bp_veh.set_attribute('color', '0,0,0' if bp_id == 'vehicle.audi.etron' else '255,0,0')

            trans = safe_waypoints[candidates[0]].transform
            trans.location.z += 0.5
            veh = world.try_spawn_actor(bp_veh, trans)

            if veh is None:
                print(f"  WARNING: could not spawn vehicle {i} ({bp_id}) — skipping")
                continue

            veh.set_target_velocity(carla.Vector3D(0, 0, 0))
            print(f"spawned vehicle {veh.id} as {bp_id}")

            agent_ids.append(veh.id)
            agent_trackers[veh.id] = Figure8Tracker(R, t_spawn)
            agent_pids[veh.id] = PIDController(Kp=0.5, Ki=0.25, Kd=0.02, integral_limit=10.0)
            agent_gap_pids[veh.id] = HeadwayPIDController(Kp=GAIN_HEADWAY_P, Ki=GAIN_HEADWAY_I, Kd=GAIN_HEADWAY_D, integral_limit=GAIN_INTEGRAL_LIMIT)
            spawn_times[veh.id] = 0.0
            vehicle_blueprints[veh.id] = bp_id

            vehicles_list.append(veh)
            vehicles_by_id[veh.id] = veh

        radius_label = str(R).replace('.', 'p')
        throughput_csv_file, throughput_csv_writer, throughput_csv_filename = create_throughput_csv(
            output_dir=run_output_dir,
            filename_prefix=f"throughput_N{N}_R{radius_label}"
        )
        print(f"Logging throughput to {throughput_csv_filename}")

        tracked_vehicle_ids = agent_ids
        vehicle_csv_file, vehicle_csv_writer, vehicle_csv_filename = create_vehicle_csv(
            output_dir=run_output_dir,
            filename_prefix=f"vehicle_allN{N}_R{radius_label}"
        )
        print(f"Logging all {len(tracked_vehicle_ids)} vehicles {tracked_vehicle_ids} to {vehicle_csv_filename}")

        lap_csv_file, lap_csv_writer, lap_csv_filename = create_lap_csv(
            output_dir=run_output_dir,
            filename_prefix=f"lap_times_N{N}_R{radius_label}"
        )
        print(f"Logging lap times to {lap_csv_filename}")
        if agent_ids:
            ev_debug_csv_file, ev_debug_csv_writer, ev_debug_csv_filename = create_ev_debug_csv(
                output_dir=run_output_dir,
                filename_prefix=f"ev_debug_vid{agent_ids[0]}_N{N}_R{radius_label}"
            )
            print(f"Logging EV debug for vehicle {agent_ids[0]} to {ev_debug_csv_filename}")
        print("Simulation Running...")

        # Lap and crossing state
        lap_counts = {vid: 0 for vid in agent_ids}
        lap_started_at = {vid: 0.0 for vid in agent_ids}
        lap_trigger_armed = {vid: False for vid in agent_ids}
        ramp_locs = {}
        t_previous = {vid: agent_trackers[vid].t for vid in agent_ids}
        total_crossings = 0
        ramp_complete = False
        safety_violation_active = {vid: False for vid in agent_ids}
        total_safety_violations = 0
        total_delay_s = 0.0

        # Speed, acceleration, and energy state
        prev_v_ms = {}
        accelerations = {}
        raw_accelerations = {}
        filtered_path_speeds = {}
        power_demands = {}
        prev_s_values = {}

        # Green-discharge state: per-vehicle countdown and previous override flag
        discharge_timer = {vid: 0.0 for vid in agent_ids}
        prev_in_tl_override = {vid: False for vid in agent_ids}

        # Aggregate run metrics
        avg_speed = 0.0
        elapsed_time = 0.0
        total_energy_kWh = 0.0
        total_distance_km = 0.0
        throughput = 0.0
        safety_violation_rate = 0.0

        # Controller and steady-state state
        GAP_EMA_ALPHA = 0.9
        gap_ema = {vid: None for vid in agent_ids}
        prev_target_speed = {}
        prev_speed = {}
        prev_steer = {}
        speed_history = deque(maxlen=int(STEADY_STATE_WINDOW_S / DT))
        steady_state_reached = False
        steady_state_time = None

        # Headway evaluation metrics
        # Per-vehicle yellow light decision: "stop" or "go".
        # Set once when a vehicle first enters yellow; cleared when it passes the stop line.
        yellow_decisions = {}

        # Stop line positions (constant — computed once before the loop)
        stop_line_1 = geom.dist_at_pi - STOP_LINE_DIST
        stop_line_2 = geom.total_length - STOP_LINE_DIST

        # Adaptive TL state (CROW 4-state machine)
        phase              = 0               # 0 = G1 serving, 1 = G2 serving
        phase_state        = "fixed_green"   # "fixed_green" / "ext_green" / "yellow"
        phase_elapsed      = 0.0
        gap_timer          = 0.0             # gap counter in VAG2 / ext_green state
        extensions_granted = 0

        # --- CROW Aanvraag Registers ---
        # aanvraag_fase[0] is voor richting 1, aanvraag_fase[1] is voor richting 2
        aanvraag_fase = [False, False]

        _GRN = "\033[30;42m"
        _YLW = "\033[30;43m"
        _RED = "\033[97;41m"
        _RST = "\033[0m"

        def _badge(state, t_left):
            c = {"green": _GRN, "yellow": _YLW, "red": _RED}.get(state, "")
            return f"{c} {state.upper():<6} {t_left:4.1f}s {_RST}"

        ev_debug_vid = agent_ids[0] if agent_ids else None
        ev_energy_vehicle_kWh = 0.0

        step_count = 0
        while True:
            world.tick()
            step_count += 1
            elapsed_time = step_count * DT

            # --- 1. State Estimation (t -> S) ---
            current_s_values = {}
            current_locations = {}
            velocities = {}

            for veh in vehicles_list:
                loc = veh.get_location()
                current_locations[veh.id] = loc
                t_new = agent_trackers[veh.id].update_t(loc)

                s = geom.get_s_from_t(t_new)
                current_s_values[veh.id] = s

                v = veh.get_velocity()
                speed = KMH_PER_MS * math.sqrt(v.x**2 + v.y**2)
                velocities[veh.id] = speed



                if step_count > WARMUP_STEPS:
                    old_t = t_previous[veh.id]
                    # Pass through t = π (mid-lap origin crossing)
                    if old_t <= math.pi < t_new:
                        total_crossings += 1
                    # Pass through t = 0/2π (wrap-around origin crossing)
                    # Detected by t_new being much smaller than old_t (vehicle just wrapped)
                    if t_new < old_t - math.pi:
                        total_crossings += 1
                    t_previous[veh.id] = t_new

            sorted_agents = sorted(agent_ids, key=lambda vid: current_s_values[vid])

            # Determine Global Nominal Speed (Ramping) — calculated before TL logic since TL uses it
            if step_count <= WARMUP_STEPS: current_nominal_speed = 0.0
            elif step_count <= (WARMUP_STEPS + RAMP_STEPS):
                progress = (step_count - WARMUP_STEPS) / RAMP_STEPS
                current_nominal_speed = TARGET_SPEED_KMH * progress
            else: current_nominal_speed = TARGET_SPEED_KMH

            # --- CROW Koplus Aanvraag Registratie ---
            for f in [0, 1]:
                # Een fase kan alleen een aanvraag indienen als hij zelf GEEN groen heeft
                if phase != f or phase_state == "yellow":
                    s_streep = stop_line_1 if f == 0 else stop_line_2

                    # Koplus: ligt tussen 0 en 10 meter vóór de stopstreep
                    koplus_bezet = any(
                        (s_streep - 10.0) <= current_s_values[vid] <= s_streep
                        for vid in agent_ids
                    )

                    # Onthoud de aanvraag permanent (totdat de richting groen krijgt)
                    if koplus_bezet:
                        aanvraag_fase[f] = True

            # Als een fase momenteel groen is (fixed of ext), is de aanvraag ingewilligd en wordt deze gereset
            if phase_state in ["fixed_green", "ext_green"]:
                aanvraag_fase[phase] = False

            # --- 2. Adaptive Traffic Light Control ---
            conflict_pair = None
            tl_speed_overrides = {}
            tl_g1      = "---"
            tl_g2      = "---"
            tl_g1_left = 0.0
            tl_g2_left = 0.0
            candidates_1_set = set()
            candidates_2_set = set()

            if step_count > WARMUP_STEPS:
                phase_elapsed += DT

                # Which stop line is the active phase serving?
                stop_s = stop_line_1 if phase == 0 else stop_line_2

                # CROW loop detector: occupancy within the physical detection carpet.
                # d is the distance from the stop line to the UPSTREAM (far) edge of the loop,
                # so the carpet spans (stop_s - d) to (stop_s - d + L_loop):
                #   far  edge (upstream)  : d        metres before stop line  = 30 m
                #   near edge (downstream): d-L_loop metres before stop line  = 25 m
                loop_far_edge  = stop_s - d
                loop_near_edge = stop_s - d + L_loop
                loop_occupied  = any(
                    loop_far_edge <= current_s_values[vid] <= loop_near_edge
                    for vid in agent_ids
                )

                # --- CROW 4-state machine ---
                if phase_state == "fixed_green":
                    # State 2: deterministic t_gg timer; ignore loop entirely
                    if phase_elapsed >= G_MAX:
                        print(f"[PHASE] phase={phase} reason=max-out(fixed) elapsed={phase_elapsed:.1f}s")
                        phase_state   = "yellow"
                        phase_elapsed = 0.0
                    elif phase_elapsed >= t_gg:
                        # Queue discharged — hand off to gap-extension logic
                        phase_state = "ext_green"
                        gap_timer   = 0.0

                elif phase_state == "ext_green":
                    # State 3: VAG2 / Hiaatmeting — occupancy resets gap counter
                    if loop_occupied:
                        if gap_timer > 0.0:
                            extensions_granted += 1
                        gap_timer = 0.0
                    else:
                        gap_timer += DT

                    terminate = False
                    reason    = ""
                    if phase_elapsed >= G_MAX:
                        terminate = True
                        reason    = "max-out"
                    elif gap_timer > t_H:
                        terminate = True
                        reason    = "gap-out"

                    if terminate:
                        print(f"[PHASE] phase={phase} reason={reason} "
                              f"elapsed={phase_elapsed:.1f}s gap_timer={gap_timer:.2f}s "
                              f"extensions={extensions_granted}")
                        phase_state   = "yellow"
                        phase_elapsed = 0.0

                elif phase_state == "yellow":
                    # State 4: deterministic t_gl clearance timer
                    if phase_elapsed >= TL_YELLOW_TIME:
                        phase              = 1 - phase
                        phase_state        = "fixed_green"
                        phase_elapsed      = 0.0
                        gap_timer          = 0.0
                        extensions_granted = 0

                # Map phase + state to G1 / G2 CARLA signal strings
                _is_green  = phase_state in ("fixed_green", "ext_green")
                _sig_state = "green" if _is_green else phase_state

                if phase == 0:
                    tl_g1 = _sig_state
                    tl_g2 = "red"
                else:
                    tl_g1 = "red"
                    tl_g2 = _sig_state

                # Time left for the dashboard display
                if phase_state == "fixed_green":
                    t_left = max(t_gg - phase_elapsed, 0.0)
                elif phase_state == "ext_green":
                    t_left = max(t_H - gap_timer, 0.0)
                else:
                    t_left = max(TL_YELLOW_TIME - phase_elapsed, 0.0)

                if phase == 0:
                    tl_g1_left = t_left
                    tl_g2_left = 0.0
                else:
                    tl_g1_left = 0.0
                    tl_g2_left = t_left

                # Approach-window speed overrides (vehicles braking for red / yellow)
                candidates_1 = [vid for vid in agent_ids
                                if (geom.dist_at_pi - APPROACH_WINDOW_M) < current_s_values[vid] < geom.dist_at_pi]
                candidates_2 = [vid for vid in agent_ids
                                if (geom.total_length - APPROACH_WINDOW_M) < current_s_values[vid] < geom.total_length]
                candidates_1_set = set(candidates_1)
                candidates_2_set = set(candidates_2)

                # Safety: violation if geodesic gap through crossing drops below safety distance
                if candidates_1 and candidates_2:
                    c1 = max(candidates_1, key=lambda vid: current_s_values[vid])
                    c2 = max(candidates_2, key=lambda vid: current_s_values[vid])
                    conflict_pair = (c1, c2)
                    dist_to_cross_1 = geom.dist_at_pi - current_s_values[c1]
                    dist_to_cross_2 = geom.total_length - current_s_values[c2]
                    gap_at_conflict = dist_to_cross_1 + dist_to_cross_2 - VEHICLE_LENGTH
                    if gap_at_conflict < SAFETY_DISTANCE_M:
                        if not safety_violation_active[c1] and not safety_violation_active[c2]:
                            total_safety_violations += 1
                        safety_violation_active[c1] = True
                        safety_violation_active[c2] = True
                    else:
                        safety_violation_active[c1] = False
                        safety_violation_active[c2] = False
                else:
                    for vid in agent_ids:
                        safety_violation_active[vid] = False

                for group_cands, stop_line_s, tl_state in [
                    (candidates_1, stop_line_1, tl_g1),
                    (candidates_2, stop_line_2, tl_g2),
                ]:
                    for vid in group_cands:
                        dist_to_stop = stop_line_s - current_s_values[vid]

                        if dist_to_stop <= 0:
                            if dist_to_stop > -3.0 and tl_state == "red":
                                tl_speed_overrides[vid] = 0.0
                            else:
                                yellow_decisions.pop(vid, None)
                            continue

                        if tl_state == "green":
                            yellow_decisions.pop(vid, None)
                            continue

                        if tl_state == "yellow" and vid not in yellow_decisions:
                            v_ms = velocities[vid] / KMH_PER_MS
                            stop_dist_needed = v_ms * REACTION_TIME + v_ms**2 / (2 * MAX_DECEL_MS2) if v_ms > 0 else 0.0
                            yellow_decisions[vid] = "stop" if stop_dist_needed < dist_to_stop else "go"

                        should_stop = tl_state == "red" or yellow_decisions.get(vid) == "stop"
                        if should_stop:
                            tl_speed_overrides[vid] = 0.0 if dist_to_stop <= 1.0 else current_nominal_speed * min(dist_to_stop / APPROACH_WINDOW_M, 1.0)

            # Push software TL states to CARLA actors every tick.
            for _tl in tl_group_1:
                _tl.set_state(_STATE_MAP[tl_g1])
            for _tl in tl_group_2:
                _tl.set_state(_STATE_MAP[tl_g2])

            throughput = total_crossings / max(elapsed_time, 1)

            dash_lines = []
            u_headway_log = {}
            gap_meters_log = {}
            headway_error_log = {}
            target_speed_log = {}

            # Arm lap timing once, at the moment ramp-up completes
            if not ramp_complete and step_count > WARMUP_STEPS + RAMP_STEPS:
                ramp_complete = True
                ramp_end_time = step_count * DT
                for vid in agent_ids:
                    lap_started_at[vid] = ramp_end_time
                    lap_trigger_armed[vid] = True
                    ramp_locs[vid] = current_locations[vid]

            # --- Steady-state detection (after ramp-up only) ---
            if ramp_complete and velocities:
                fleet_avg_kmh = sum(velocities.values()) / len(velocities)
                speed_history.append(fleet_avg_kmh)
                if not steady_state_reached and len(speed_history) == speed_history.maxlen:
                    arr = np.array(speed_history)
                    if arr.std() < STEADY_STATE_SPEED_STD_KMH and abs(arr.mean() - TARGET_SPEED_KMH) < STEADY_STATE_SPEED_ERR_KMH:
                        steady_state_reached = True
                        steady_state_time = elapsed_time
                        print(f"\n*** STEADY STATE REACHED at t={elapsed_time:.2f}s "
                              f"(fleet avg={arr.mean():.2f} km/h, std={arr.std():.3f} km/h) ***")

            for i, vid in enumerate(sorted_agents):
                veh = vehicles_by_id[vid]

                # --- 3. Headway Sensing ---
                u_headway = 0.0
                gap_meters = TOTAL_TRACK_LENGTH
                headway_error = 0.0
                leader_id = None

                if N > 1:
                    leader_idx = (i + 1) % N
                    leader_id = sorted_agents[leader_idx]
                    gap_meters = (current_s_values[leader_id] - current_s_values[vid]) % TOTAL_TRACK_LENGTH
                    if gap_ema[vid] is None:
                        gap_ema[vid] = gap_meters
                    else:
                        gap_ema[vid] = GAP_EMA_ALPHA * gap_ema[vid] + (1.0 - GAP_EMA_ALPHA) * gap_meters
                    gap_meters = gap_ema[vid]
                    headway_error = gap_meters - TARGET_GAP_METERS

                gap_meters_log[vid] = gap_meters
                headway_error_log[vid] = headway_error

                # --- 4. Combine Controls ---
                if step_count <= WARMUP_STEPS:
                    target_speed = 0.0
                    prev_target_speed[vid] = 0.0
                    agent_pids[vid].reset()
                    agent_gap_pids[vid].reset()
                elif step_count <= WARMUP_STEPS + RAMP_STEPS:
                    target_speed = current_nominal_speed
                    if vid in tl_speed_overrides:
                        target_speed = min(target_speed, tl_speed_overrides[vid])
                    target_speed = max(0.0, target_speed)
                    prev_target_speed[vid] = target_speed
                else:
                    # u_headway: gap PID correction
                    if N > 1 and leader_id is not None:
                        u_headway = agent_gap_pids[vid].run(
                            gap_meters, TARGET_GAP_METERS, DT,
                        )
                        u_headway = float(np.clip(u_headway, -MAX_HEADWAY_KMH, MAX_HEADWAY_KMH))

                    # Green-discharge: suppress negative headway so queue re-accelerates freely
                    if prev_in_tl_override[vid] and vid not in tl_speed_overrides and velocities[vid] < 5.0:
                        discharge_timer[vid] = DISCHARGE_DURATION_S
                    prev_in_tl_override[vid] = vid in tl_speed_overrides
                    if discharge_timer[vid] > 0.0:
                        u_headway = max(u_headway, 0.0)
                        discharge_timer[vid] = max(0.0, discharge_timer[vid] - DT)

                    u_headway_log[vid] = u_headway

                    # Additive target, then hard caps: TL stop, queue proximity
                    target_speed = current_nominal_speed + u_headway
                    if vid in tl_speed_overrides:
                        target_speed = min(target_speed, tl_speed_overrides[vid])
                    if N > 1 and gap_meters <= QUEUE_SLOW_GAP:
                        ratio = (gap_meters - QUEUE_STOP_GAP) / (QUEUE_SLOW_GAP - QUEUE_STOP_GAP)
                        target_speed = 0.0 if gap_meters <= QUEUE_STOP_GAP else min(target_speed, current_nominal_speed * ratio)

                    target_speed = max(0.0, target_speed)
                    max_delta_kmh = COMFORTABLE_MAX_ACCEL_MS2 * DT * KMH_PER_MS
                    prev = prev_target_speed.get(vid, target_speed)
                    target_speed = float(np.clip(target_speed, prev - max_delta_kmh, prev + max_delta_kmh))
                    prev_target_speed[vid] = target_speed

                target_speed_log[vid] = target_speed

                # --- 5. Lateral Control (Steering) ---
                tracker = agent_trackers[vid]
                curr_loc = veh.get_location()
                veh_trans = veh.get_transform()
                veh_fwd = veh_trans.get_forward_vector()

                dt_lookahead = 5.0 / R
                t_future = tracker.t + dt_lookahead
                math_target = tracker.get_math_location(t_future)
                wp_target = carla_map.get_waypoint(math_target, project_to_road=True, lane_type=carla.LaneType.Driving)
                wp_curr = carla_map.get_waypoint(curr_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

                # Discard wp_target that snapped to the opposite-direction road at the intersection.
                # This happens when both loops cross the same physical junction and project_to_road
                # picks the incoming road of the other loop instead of the correct exit road.
                if wp_target is not None:
                    wp_fwd = wp_target.transform.get_forward_vector()
                    if veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y < 0.0:
                        wp_target = None

                at_junction = (wp_curr is not None and wp_curr.is_junction) or (wp_target is not None and wp_target.is_junction)
                if at_junction or wp_target is None:
                    # Use the figure-8 math tangent directly — immune to waypoint-snap errors
                    # and safe when prev_steer was 0 (vehicle stopped at red before junction).
                    tang_x = -math.cos(t_future)
                    tang_y = -2.0 * math.cos(2.0 * t_future)
                    tang_norm = math.sqrt(tang_x ** 2 + tang_y ** 2)
                    if tang_norm > 0:
                        tang_x /= tang_norm
                        tang_y /= tang_norm
                    steer_cmd = float(np.clip(veh_fwd.x * tang_y - veh_fwd.y * tang_x, -1.0, 1.0))
                else:
                    steer_target = wp_target.transform.location
                    vec_to = np.array([steer_target.x - curr_loc.x, steer_target.y - curr_loc.y])
                    norm = np.linalg.norm(vec_to)
                    if norm > 0: vec_to /= norm
                    steer_cmd = float(np.clip(veh_fwd.x * vec_to[1] - veh_fwd.y * vec_to[0], -1.0, 1.0))
                prev_steer[vid] = steer_cmd

                # --- 6. Actuation + Energy Calculation ---
                curr_speed = velocities[vid]
                v_ms = curr_speed / KMH_PER_MS
                raw_acceleration = (v_ms - prev_v_ms.get(vid, v_ms)) / DT
                prev_v_ms[vid] = v_ms
                raw_accelerations[vid] = raw_acceleration
                accelerations[vid] = update_ema(accelerations.get(vid), raw_acceleration, ACCEL_EMA_ALPHA)
                bp_id = vehicle_blueprints[vid]
                cfg = VEHICLE_CONFIGS.get(bp_id, VEHICLE_CONFIGS['vehicle.tesla.model3'])
                F_inertia  = cfg['mass'] * accelerations[vid]
                F_aero     = 0.5 * AIR_DENSITY * cfg['drag_coefficient'] * cfg['frontal_area'] * v_ms**2
                F_rolling  = cfg['mass'] * GRAVITATIONAL_ACCELERATION * cfg['rolling_resistance']
                F_total    = F_inertia + F_aero + F_rolling
                Power_mech = F_total * v_ms
                motor_rpm = cfg['gear_ratio'] * max(v_ms, 0.0) * 60.0 / (2.0 * math.pi * cfg['wheel_radius'])
                if v_ms > 0.0:
                    if F_total >= 0:
                        motor_torque = (F_total * cfg['wheel_radius'] / cfg['gear_ratio']) / cfg['gearbox_eff']
                    else:
                        motor_torque = (F_total * cfg['wheel_radius'] / cfg['gear_ratio']) * cfg['gearbox_eff']
                else:
                    motor_torque = 0.0
                eff_map = MOTOR_EFF_MAPS.get(bp_id, tesla_eff_map)
                rpm_q = float(np.clip(motor_rpm, 0.0, 18000.0))
                tq_q = float(np.clip(abs(motor_torque), 0.0, 375.0))
                eta = max(float(eff_map([[rpm_q, tq_q]])[0]), 0.60)
                if Power_mech >= 0:
                    power_demands[vid] = Power_mech / (cfg['gearbox_eff'] * eta)
                else:
                    power_demands[vid] = Power_mech * cfg['gearbox_eff'] * eta

                if ev_debug_csv_writer is not None and vid == ev_debug_vid:
                    ev_energy_vehicle_kWh += power_demands[vid] * DT / 3_600_000
                    ev_debug_csv_writer.writerow([
                        f"{elapsed_time:.4f}",
                        f"{v_ms:.6f}",
                        f"{raw_acceleration:.6f}",
                        f"{F_inertia:.4f}",
                        f"{F_aero:.4f}",
                        f"{F_rolling:.4f}",
                        f"{F_total:.4f}",
                        f"{Power_mech:.4f}",
                        f"{motor_rpm * 2.0 * math.pi / 60.0:.6f}",
                        f"{motor_rpm:.4f}",
                        f"{motor_torque:.6f}",
                        f"{eta:.6f}",
                        f"{power_demands[vid]:.4f}",
                        f"{0.0:.1f}",
                        f"{ev_energy_vehicle_kWh:.8f}",
                    ])
                    if step_count % int(1 / DT) == 0:
                        ev_debug_csv_file.flush()

                control = carla.VehicleControl()
                control.steer = float(steer_cmd)

                if step_count <= WARMUP_STEPS:
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.hand_brake = True
                elif step_count <= WARMUP_STEPS + RAMP_STEPS:
                    throttle_brake = agent_pids[vid].run(target_speed, curr_speed, DT)
                    control.hand_brake = False
                    if throttle_brake >= 0:
                        control.throttle = min(throttle_brake, 1.0)
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0
                        control.brake = min(abs(throttle_brake), 1.0)
                else:
                    throttle_brake = agent_pids[vid].run(target_speed, curr_speed, DT)
                    control.hand_brake = False
                    if throttle_brake >= 0:
                        control.throttle = min(throttle_brake, 1.0)
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0
                        control.brake = min(abs(throttle_brake), 1.0)

                veh.apply_control(control)
                throughput = total_crossings / max(elapsed_time, 1)

                # TL state + time-left for this vehicle's crossing direction
                if vid in candidates_1_set:
                    tl_col = _badge(tl_g1, tl_g1_left)
                elif vid in candidates_2_set:
                    tl_col = _badge(tl_g2, tl_g2_left)
                else:
                    tl_col = f" {'---':<6} {'':>4}  "


                # Lap timing
                if lap_csv_writer is not None and ramp_complete:
                    dist_from_ramp = curr_loc.distance(ramp_locs[vid])
                    time_since_last_lap = elapsed_time - lap_started_at[vid]

                    if dist_from_ramp > LAP_CLEAR_RADIUS:
                        lap_trigger_armed[vid] = True
                    elif (
                        lap_trigger_armed[vid]
                        and dist_from_ramp <= LAP_TRIGGER_RADIUS
                        and time_since_last_lap >= LAP_MIN_TIME
                    ):
                        lap_counts[vid] += 1
                        lap_start_time = lap_started_at[vid]
                        lap_duration = elapsed_time - lap_start_time
                        delay_time = lap_duration - MINIMAL_LAP_TIME
                        total_delay_s += delay_time
                        ramp_loc = ramp_locs[vid]
                        lap_csv_writer.writerow([
                            vid,
                            lap_counts[vid],
                            f"{ramp_loc.x:.4f}",
                            f"{ramp_loc.y:.4f}",
                            f"{spawn_times[vid]:.2f}",
                            f"{lap_start_time:.2f}",
                            f"{elapsed_time:.2f}",
                            f"{lap_duration:.2f}",
                            f"{delay_time:.2f}",
                        ])
                        lap_csv_file.flush()
                        lap_started_at[vid] = elapsed_time
                        lap_trigger_armed[vid] = False

                safety_flag = "VIOLATION" if safety_violation_active.get(vid) else "ok"
                dash_lines.append(
                    f"{vid:<4} | {gap_meters:6.1f} | {headway_error:7.2f} | {u_headway:6.2f} | {curr_speed:5.1f} | {raw_accelerations[vid]:6.2f} | {tl_col} | {throughput:10.3f} | {power_demands[vid]/1000:7.2f} kW | {safety_flag:>9}"
                )

            # Crash detection: adjacent pairs on the ring + active intersection pair
            crash_detected = False
            if step_count > WARMUP_STEPS and len(sorted_agents) > 1:
                for j in range(len(sorted_agents)):
                    va = sorted_agents[j]
                    vb = sorted_agents[(j + 1) % len(sorted_agents)]
                    if get_bb_distance(vehicles_by_id[va], vehicles_by_id[vb]) <= 0.0:
                        print(f"\n*** CRASH at t={elapsed_time:.2f}s: vehicle {va} and vehicle {vb} "
                              f"bounding boxes overlap — stopping simulation ***")
                        crash_detected = True
                        break
                if not crash_detected and conflict_pair is not None:
                    c1, c2 = conflict_pair
                    if get_bb_distance(vehicles_by_id[c1], vehicles_by_id[c2]) <= 0.0:
                        print(f"\n*** CRASH at t={elapsed_time:.2f}s: vehicle {c1} and vehicle {c2} "
                              f"bounding boxes overlap — stopping simulation ***")
                        crash_detected = True
            if crash_detected:
                break

            if elapsed_time >= max_duration_s:
                avg_speed = sum(velocities.values()) / len(velocities) if velocities else 0.0
                safety_violation_rate = total_safety_violations / max(elapsed_time, 1)
                summary = {
                    "run_index": run_index,
                    "run_directory": run_output_dir,
                    "vehicles": N,
                    "radius_m": R,
                    "duration_s": max_duration_s,
                    "tracked_vehicle_ids": ",".join(str(vid) for vid in tracked_vehicle_ids),
                    "throughput_csv": throughput_csv_filename,
                    "vehicle_csv": vehicle_csv_filename,
                    "lap_csv": lap_csv_filename,
                    "final_time_seconds": elapsed_time,
                    "final_throughput": throughput,
                    "avg_speed_kmh": avg_speed,
                    "energy_kWh": total_energy_kWh,
                    "total_distance_km": total_distance_km,
                    "safety_violation_rate": safety_violation_rate,
                    "total_delay_s": total_delay_s,
                    "total_crossings": total_crossings,
                    "total_safety_violations": total_safety_violations,
                    "steady_state_time_s": steady_state_time if steady_state_time is not None else -1.0,
                }
                print(f"Run {run_index} completed at {elapsed_time:.2f} s")
                break

            # Throughput CSV (every tick, flush once per second)
            if throughput_csv_writer is not None:
                total_energy_kWh += sum(power_demands.values()) * DT / (1000 * 3600)
                total_distance_km += sum(velocities.values()) * DT / 3600
                avg_speed = sum(velocities.values()) / len(velocities) if velocities else 0.0
                safety_violation_rate = total_safety_violations / max(elapsed_time, 1)
                u_headway_avg = sum(u_headway_log.values()) / len(u_headway_log) if u_headway_log else 0.0
                throughput_csv_writer.writerow([
                    f"{elapsed_time:.2f}", f"{throughput:.6f}", f"{avg_speed:.2f}",
                    f"{total_energy_kWh:.6f}", f"{total_distance_km:.4f}",
                    f"{safety_violation_rate:.6f}", f"{total_delay_s:.2f}",
                    f"{u_headway_avg:.4f}",
                    1 if steady_state_reached else 0,
                ])
                if step_count % int(1 / DT) == 0:
                    throughput_csv_file.flush()

            # Vehicle CSV (every tick, flush once per second)
            if vehicle_csv_writer is not None and tracked_vehicle_ids:
                for tracked_vid in tracked_vehicle_ids:
                    if tracked_vid not in power_demands:
                        continue
                    vehicle_csv_writer.writerow([
                        f"{elapsed_time:.2f}",
                        tracked_vid,
                        vehicle_blueprints.get(tracked_vid, ""),
                        f"{velocities[tracked_vid] / KMH_PER_MS:.4f}",
                        f"{velocities[tracked_vid]:.4f}",
                        f"{accelerations.get(tracked_vid, 0.0):.4f}",
                        f"{raw_accelerations.get(tracked_vid, 0.0):.4f}",
                        f"{gap_meters_log.get(tracked_vid, TOTAL_TRACK_LENGTH):.4f}",
                        f"{TARGET_GAP_METERS:.4f}",
                        f"{headway_error_log.get(tracked_vid, 0.0):.4f}",
                        f"{u_headway_log.get(tracked_vid, 0.0):.4f}",
                        f"{power_demands[tracked_vid] / 1000:.4f}",
                    ])
                if step_count % int(1 / DT) == 0:
                    vehicle_csv_file.flush()

            # --- 7. Visualization ---
            if step_count % 5 == 0:
                print("\033[H\033[J")
                if step_count <= WARMUP_STEPS: status = "SETTLING..."
                elif step_count <= WARMUP_STEPS + RAMP_STEPS: status = f"RAMPING UP ({current_nominal_speed:.1f} km/h)"
                else: status = "ACTIVE"
                throughput = total_crossings / max(elapsed_time, 1)
                safety_violation_rate = total_safety_violations / max(elapsed_time, 1)
                print(f"--- RUN {run_index} | STEP {step_count} [{status}] ---")
                print(f"Throughput: {throughput:.4f} cross/s | SafetyViol: {safety_violation_rate:.4f} viol/s ({total_safety_violations}) | Energy: {total_energy_kWh:+.4f} kWh | Target gap: {TARGET_GAP_METERS:.1f} m")
                print(f"TL  G1 (π) : {_badge(tl_g1, tl_g1_left)}    G2 (2π): {_badge(tl_g2, tl_g2_left)}")
                print(f"Phase {phase} [{phase_state}]  el={phase_elapsed:.1f}s  gap_tmr={gap_timer:.2f}s  ext={extensions_granted}  |  t_gg={t_gg:.1f}s t_H={t_H:.2f}s t_gl={TL_YELLOW_TIME:.0f}s G_MAX={G_MAX:.0f}s")
                print(f"{'ID':<4} | {'Gap(m)':>6} | {'Err(m)':>7} | {'u_Head':>6} | {'Speed':>5} | {'Accel':>6} | {'TL':^14} | {'Throughput':>10} | {'Power':>10} | {'Safety':>9}")
                for line in dash_lines:
                    print(line)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if throughput_csv_file is not None:
            throughput_csv_file.close()
            print(f"Throughput log saved to {throughput_csv_filename}")
        if vehicle_csv_file is not None:
            vehicle_csv_file.close()
            print(f"Vehicle log saved to {vehicle_csv_filename}")
        if lap_csv_file is not None:
            lap_csv_file.close()
            print(f"Lap times log saved to {lap_csv_filename}")
        if ev_debug_csv_file is not None:
            ev_debug_csv_file.close()
            print(f"EV debug log saved to {ev_debug_csv_filename}")
        for v in vehicles_list:
            try:
                v.destroy()
            except Exception:
                pass
        # Unfreeze map traffic lights so CARLA resumes normal control (don't destroy them)
        for tl in tl_actors:
            try:
                tl.freeze(False)
            except Exception:
                pass
        if original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass

    return summary


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--number-of-vehicles', default=57, type=int)
    argparser.add_argument('--radius', default=250.0, type=float)
    argparser.add_argument('--duration-seconds', default=1800.0, type=float)
    argparser.add_argument('--output-dir', default=RESULTS_DIR)
    argparser.add_argument('--start-run', default=1, type=int)
    argparser.add_argument('--end-run', default=FIXED_RUN_COUNT, type=int)
    args = argparser.parse_args()
    args.runs = FIXED_RUN_COUNT
    args.start_run = max(1, args.start_run)
    args.end_run = min(FIXED_RUN_COUNT, args.end_run)

    if args.start_run > args.end_run:
        raise ValueError(f"start-run ({args.start_run}) must be <= end-run ({args.end_run})")

    batch_label = (
        f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_N{args.number_of_vehicles}_T{int(args.duration_seconds)}_R{str(args.radius).replace('.', 'p')}"
    )
    batch_output_dir = os.path.join(args.output_dir, batch_label)
    os.makedirs(batch_output_dir, exist_ok=True)

    summary_file, summary_writer, summary_filename = create_batch_summary_csv(batch_output_dir)
    print(f"Batch results directory: {batch_output_dir}")
    print(f"Batch summary will be stored in {summary_filename}")

    MAX_RUN_RETRIES = 10
    RETRY_WAIT_S = 90

    def wait_for_simulator():
        print(f"Waiting up to {RETRY_WAIT_S}s for simulator to come back online...")
        for _ in range(RETRY_WAIT_S):
            try:
                probe = carla.Client('127.0.0.1', 2000)
                probe.set_timeout(3.0)
                probe.get_server_version()
                print("Simulator is back online.")
                time.sleep(5)
                return
            except Exception:
                time.sleep(1)

        # Server didn't recover on its own — kill it and restart
        print("Simulator did not respond. Restarting CARLA...")
        subprocess.run(['pkill', '-f', 'CarlaUE4'], check=False)
        time.sleep(5)
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carla_server.log')
        with open(log_path, 'w') as log_f:
            subprocess.Popen(CARLA_SERVER_CMD, stdout=log_f, stderr=log_f)
        print(f"CARLA started (log → {log_path}). Waiting up to {CARLA_START_WAIT_S}s...")
        for _ in range(CARLA_START_WAIT_S):
            try:
                probe = carla.Client('127.0.0.1', 2000)
                probe.set_timeout(3.0)
                probe.get_server_version()
                print("CARLA is online. Loading 8ring map...")
                probe.load_world(CARLA_MAP_NAME)
                print(f"Map '{CARLA_MAP_NAME}' loaded.")
                time.sleep(10)
                return
            except Exception:
                time.sleep(1)
        print("CARLA restart timed out — retrying run anyway.")

    def purge_world_vehicles():
        """Destroy all vehicle and sensor actors still alive in the world before spawning."""
        try:
            probe = carla.Client('127.0.0.1', 2000)
            probe.set_timeout(10.0)
            w = probe.get_world()
            actors = w.get_actors()
            to_destroy = list(actors.filter('vehicle.*')) + list(actors.filter('sensor.*'))
            if to_destroy:
                print(f"  Purging {len(to_destroy)} leftover actors from previous attempt...")
            for a in to_destroy:
                try:
                    a.destroy()
                except Exception:
                    pass
            if to_destroy:
                time.sleep(2)
        except Exception as exc:
            print(f"  Warning: world purge failed ({exc})")

    try:
        run_index = args.start_run
        while run_index <= args.end_run:
            run_output_dir = build_run_output_dir(args.output_dir, batch_label, run_index)
            os.makedirs(run_output_dir, exist_ok=True)

            run_summary = None
            for attempt in range(1, MAX_RUN_RETRIES + 1):
                print(f"\n=== Starting run {run_index}/{args.runs}"
                      + (f" (attempt {attempt})" if attempt > 1 else "") + " ===")
                try:
                    run_summary = run_simulation(args, run_index=run_index, run_output_dir=run_output_dir)
                    break
                except Exception as e:
                    print(f"\n[ERROR] Simulator crash on run {run_index} attempt {attempt}: {e}")
                    if os.path.exists(run_output_dir):
                        shutil.rmtree(run_output_dir)
                        print(f"Removed partial output: {run_output_dir}")
                    if attempt == MAX_RUN_RETRIES:
                        print(f"Run {run_index} failed after {MAX_RUN_RETRIES} attempts, skipping.")
                        break
                    wait_for_simulator()
                    purge_world_vehicles()
                    os.makedirs(run_output_dir, exist_ok=True)

            if run_summary is not None:
                write_batch_summary_row(summary_writer, run_summary)
                summary_file.flush()
            else:
                print(f"Run {run_index} did not complete normally; no summary row recorded.")

            run_index += 1
    finally:
        summary_file.close()
        print(f"Batch summary saved to {summary_filename}")


if __name__ == '__main__':
    main()

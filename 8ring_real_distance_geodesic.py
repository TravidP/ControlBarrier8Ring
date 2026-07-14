#!/usr/bin/env python
# Copyright (c) 2025 Computer Vision Center (CVC)
# Figure-8 Simulation: Distance-Based Control with Reverse Flow

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_results", "BarrierControlResults")
FIXED_RUN_COUNT = 1   # Number of simulation runs
RUN_SEED_BASE = 1000  # Fixed seed for reproducible simulations  

# Spawning N/2 + 1 Teslas and N/2 Audis
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

# Creating csv files for logging per vehicle values, KPI values, lap timing, and batch csv for evaluating if everythign simulated without errors
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
        "u_barrier",
        "power_kw",
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
        "u_barrier_firings",
        "u_barrier_avg",
        "u_headway_avg",
    ])
    return csv_file, writer, filename


def create_lap_csv(output_dir=RESULTS_DIR, filename_prefix="lap_times"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "vehicle_id",
        "lap_number",
        "spawn_x",
        "spawn_y",
        "spawn_time_s",
        "lap_start_time_s",
        "lap_end_time_s",
        "lap_duration_s",
        "delay_time_s",
    ])
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
        "tracked_vehicle_id",
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
# Function used to run simulations with fixed seeds
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
    TOTAL_TRACK_LENGTH = geom.total_length                          # Calculation of total track length based on distance mapper
    TARGET_GAP_METERS = (TOTAL_TRACK_LENGTH / N) if N > 0 else 0    # Calculation of target gap based on vehicle count and total track length

    # Simulation timing
    DT = 0.05                        # Simulation time step
    WARMUP_STEPS = 100               # Simulation steps during warm-up
    RAMP_STEPS = 250                 # Simulation steps during ramp-up
    TARGET_SPEED_KMH = 30.0          # Target speed of vehicle fleet
    COMFORTABLE_MAX_ACCEL_MS2 = 2.5  # Limits commanded target-speed slew

    # Headway controller parameters
    GAIN_HEADWAY_P =  3.50      # Proportional gain (km * s / m * h)
    GAIN_HEADWAY_I =  0.00      # Integral gain disabled
    GAIN_HEADWAY_D =  1.00      # Derivative gain (km/ h * m)
    GAIN_INTEGRAL_LIMIT =  3.0  # larger limit to allow meaningful accumulation
    MAX_HEADWAY_KMH = 10.0      # Max headway correction for realisitc speed
 
    # Barrier controller parameters
    GAIN_BARRIER = 25.0             # Gain barrier, higher is agressive correction/ lower is less correction, greater chance of crash
    BARRIER_ACTIVATION_DIST =  10   # Activation distance threshold
    APPROACH_WINDOW_M = 20          # Search window for barrier candidates
    EPSILON_DIFF = 1.0                 
    MAX_BARRIER_KMH = 7.0           # Maximum speed up 

    # Lap timing parameters
    LAP_MIN_TIME = 250.0       # Minimal lap time to avoid lap count of short laps
    MINIMAL_LAP_TIME =  281.78 # 281.78 found running simulation with N=1 at target speed
    LAP_TRIGGER_RADIUS = 1.0   # Triggering space for lap recording
    LAP_CLEAR_RADIUS = 2.0     # Triggering space for starting new lao

    # Safety distance parameters
    VEHICLE_LENGTH = 5.0       # Average vehicle length for adjustment of middle of vehicle to front
    SAFETY_DISTANCE_M = 5.0

    # EMA parameters used for smoothing
    ACCEL_EMA_ALPHA = 0.5

    # Energy consumption parameters 
    AIR_DENSITY = 1.225
    GRAVITATIONAL_ACCELERATION = 9.81

    # Vehicle specific parameters based on table 1 from report
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
    # Efficiency lookup table Audi e-tron 55 created from efficiency map in report
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

    # Efficiency lookup table Tesla model 3 created from efficiency map in report
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
    # Assigning map to vehicle type
    MOTOR_EFF_MAPS = {
        'vehicle.tesla.model3': tesla_eff_map,
        'vehicle.audi.etron':   audi_eff_map,
    }

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    # Initialization of parameters
    vehicles_list = []
    tracked_vehicle_ids = []
    plot_vehicle_ids = []
    gap_history = {}
    gap_time_history = []
    speed_history_plot = {}
    target_speed_history_plot = {}

    # Setting all csv files to none
    throughput_csv_file = None
    throughput_csv_writer = None
    throughput_csv_filename = None

    vehicle_csv_file = None
    vehicle_csv_writer = None
    vehicle_csv_filename = None

    lap_csv_file = None
    lap_csv_writer = None
    lap_csv_filename = None

    summary = None
    original_settings = None
   
    # Retrieving map from CARLA, setting synchronous mode and timestep
    try:
        world = client.get_world()
        carla_map = world.get_map()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)
        
        # Retrieving vehicle blueprints from CARLA
        bp_lib = world.get_blueprint_library()
        vehicle_blueprint_ids = build_vehicle_blueprint_sequence(N, rng) # Spawning selected blueprints random

        # Generating safe waypoints to avoid spawning simultaneously at junction, based on junction coordinates in xodr
        _all_wps = carla_map.generate_waypoints(2.0)
        safe_waypoints = [w for w in _all_wps if not w.is_junction] 
        safe_wp_xy = np.array([[w.transform.location.x, w.transform.location.y] for w in safe_waypoints])
        safe_wp_fwd = np.array([[w.transform.get_forward_vector().x, w.transform.get_forward_vector().y] for w in safe_waypoints])

        # Initialization of parameters for vehicles
        agent_ids = []
        agent_gap_pids = {}
        agent_pids = {}
        agent_trackers = {}
        spawn_times = {}
        vehicle_blueprints = {}
        vehicles_by_id = {}

        print(f"\n--- Run {run_index}: Spawning {N} EVs (Tesla Model 3 / Audi e-tron ---")
        print(f"Total Track Length: {TOTAL_TRACK_LENGTH:.1f} m")
        print(f"Target Gap: {TARGET_GAP_METERS:.1f} m")
        print(f"Run duration: {max_duration_s:.1f} s")
        print(f"Run seed: {run_seed}")

   
     
        # Spawning with noise
        for i in range(N):
            noise = rng.uniform(-0.2, 0.2) * TARGET_GAP_METERS   # Spawning with random uniform noise
            s_target = (i * TARGET_GAP_METERS  + noise) % TOTAL_TRACK_LENGTH # Defining the target spawn location in s
            t_spawn = geom.get_t_from_s(s_target)  # Using distance mapper to move from s to t, which is used by CARLA for spawning

            start_x = -R * np.sin(t_spawn)      # Defining x-position based on t        
            start_y = -R * np.sin(2 * t_spawn)  # Defining y-position based on t    
            tang = np.array([-np.cos(t_spawn), -2.0 * np.cos(2.0 * t_spawn)])   # Calculating tang for straight spawning
            tang /= np.linalg.norm(tang)    # Normalizing tang to 1

            target_xy = np.array([start_x, start_y]) # Retrieving taregt xy
            dists = np.linalg.norm(safe_wp_xy - target_xy, axis=1) # Calculating distance from spawn xy to all safe spawnpoints in CARLA
            correct_loop = (safe_wp_fwd @ tang) > 0.8  # Checking tang with direction of safe waypoints, only points that are almost similar are allowed
            candidates = np.argsort(dists[correct_loop]) # Sorting correct loop coordinates
            candidates = np.where(correct_loop)[0][candidates] # Setting closest loop coordinate as the correct spawn point

            bp_id = vehicle_blueprint_ids[i] # Retrieving from specific vehicle the blueprint
            bp_veh = bp_lib.find(bp_id)
            if bp_veh.has_attribute('color'):
                bp_veh.set_attribute('color', '0,0,0' if bp_id == 'vehicle.audi.etron' else '255,0,0') # Making Teslas red and Audis black
            
            # Spawning of vehicle at safe spawnpoint with correct blueprint and color
            trans = safe_waypoints[candidates[0]].transform
            trans.location.z += 0.5   # Spawning 0.5m above map  
            veh = world.try_spawn_actor(bp_veh, trans) # Spawning the vehicle

            if veh is None:
                print(f"  WARNING: could not spawn vehicle {i} ({bp_id}) — skipping")
                continue
            
            # Setting velocity to 0 after spawning
            veh.set_target_velocity(carla.Vector3D(0, 0, 0))
            print(f"spawned vehicle {veh.id} as {bp_id}")

            # Assigning vehicle specific tracker, PIDs and spawn time
            agent_ids.append(veh.id)
            agent_trackers[veh.id] = Figure8Tracker(R, t_spawn)
            agent_pids[veh.id] = PIDController(Kp=0.5, Ki=0.25, Kd=0.02, integral_limit=10.0)
            agent_gap_pids[veh.id] = HeadwayPIDController(Kp=GAIN_HEADWAY_P, Ki=GAIN_HEADWAY_I, Kd=GAIN_HEADWAY_D, integral_limit=GAIN_INTEGRAL_LIMIT)
            spawn_times[veh.id] = 0.0
            vehicle_blueprints[veh.id] = bp_id
            vehicles_list.append(veh)
            vehicles_by_id[veh.id] = veh

        # Creating all csv files for logging
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
        print("Simulation Running...")

        # Lap and crossing state
        ramp_locs = {}                                                  # Location of vehicle after rampup is complete, used for lap counting
        lap_counts = {vid: 0 for vid in agent_ids}                      # Triggered if vehicle passed rampup location, =+1 lap
        lap_started_at = {vid: 0.0 for vid in agent_ids}                # Triggered if vehicle leaves rampup location
        lap_trigger_armed = {vid: False for vid in agent_ids}           # Triggered if vehicle passed rampup location, time logging
        t_previous = {vid: agent_trackers[vid].t for vid in agent_ids}  # Logging time of starting lap to compute delay time
        safety_violation_active = {vid: False for vid in agent_ids}

        # Throuhgput KPI initialization
        total_crossings = 0                                             # Initializing total crossings as 0

        # Speed, acceleration, and energy initialization
        accelerations = {}
        raw_accelerations = {}
        power_demands = {}
        prev_v_ms = {}

        # Aggregate run metrics
        avg_speed = 0.0
        elapsed_time = 0.0
        total_energy_kWh = 0.0
        total_distance_km = 0.0
        total_delay_s = 0.0
        total_safety_violations = 0
        throughput = 0.0
        safety_violation_rate = 0.0

        # Parameter initialization
        GAP_EMA_ALPHA = 0.9                          # Alpha used for headway error smoothing 
        gap_ema = {vid: None for vid in agent_ids}
        prev_target_speed = {}
        prev_steer = {}
        ramp_complete = False

        # Initialization of step count and starting while loop
        step_count = 0
        while True:
            world.tick()
            step_count += 1
            elapsed_time = step_count * DT

            # 1. State Estimation (t -> S)
            current_s_values = {}
            current_locations = {}
            velocities = {}
            for veh in vehicles_list:
                loc = veh.get_location()
                current_locations[veh.id] = loc
                if step_count > WARMUP_STEPS:
                    t_new = agent_trackers[veh.id].update_t(loc)
                else:
                    t_new = agent_trackers[veh.id].t % (2 * np.pi)
                # Converting current locations from t to s    
                s = geom.get_s_from_t(t_new)
                current_s_values[veh.id] = s


                v = veh.get_velocity()
                speed = 3.6 * math.sqrt(v.x**2 + v.y**2) # Vehicle speed converted to km/h
                velocities[veh.id] = speed

            # Throughput counting based on t
                if step_count > WARMUP_STEPS:
                    old_t = t_previous[veh.id]
                    # Pass through intersection at t = π 
                    if old_t <= math.pi < t_new:
                        total_crossings += 1
                    # Pass through intersection at t = 0/2π 
                    if t_new < old_t - math.pi:
                        total_crossings += 1
                    t_previous[veh.id] = t_new      
            
            # Throughput rate     
            throughput = total_crossings / max(elapsed_time, 1)

            # Sorting vehicles based on current position along the track
            sorted_agents = sorted(agent_ids, key=lambda vid: current_s_values[vid])

            # 2. Barrier Control
            conflict_pair = None
            barrier_controls = {vid: 0.0 for vid in agent_ids}

            if step_count > WARMUP_STEPS:        
                # Identifying candidates for barrier control if within approach window from conflict point   
                candidates_1 = [
                    vid for vid in agent_ids
                    if (geom.dist_at_pi - APPROACH_WINDOW_M) < current_s_values[vid] < geom.dist_at_pi
                ]
                candidates_2 = [
                    vid for vid in agent_ids
                    if (geom.total_length - APPROACH_WINDOW_M) < current_s_values[vid] < geom.total_length
                ]
                # Sorting list of candidates and selecting vehicle closest as candidate conflict point
                if candidates_1 and candidates_2:
                    c1 = max(candidates_1, key=lambda vid: current_s_values[vid])
                    c2 = max(candidates_2, key=lambda vid: current_s_values[vid])
                    conflict_pair = (c1, c2)

                    # Calculating distance from conflict point to vehicle
                    dist_to_cross_1 = geom.dist_at_pi - current_s_values[c1]
                    dist_to_cross_2 = geom.total_length - current_s_values[c2]
                    diff_meters = dist_to_cross_2 - dist_to_cross_1         # Difference between both approaching vehicles

                    if abs(diff_meters) < BARRIER_ACTIVATION_DIST: # If absolute value of difference is below the activation distance barrier is computed
                        if abs(diff_meters) < EPSILON_DIFF:
                            safe_diff = EPSILON_DIFF * (1.0 if diff_meters >= 0 else -1.0)
                        else:
                            safe_diff = diff_meters
                        # Gradient calculation based on safe difference    
                        gradient = -1.0/ safe_diff 
                        barrier_controls[c2] = gradient 
                        barrier_controls[c1] = -gradient

                    # Safety violation calculation for candidate pair
                    gap_at_conflict = dist_to_cross_1 + dist_to_cross_2 - VEHICLE_LENGTH  # Subtracting full vehicle length to adjust for measurement from both vehicles centers
                    if gap_at_conflict < SAFETY_DISTANCE_M:         # If below safety distance the violation is counted, only if the vehicle is not yet registered 
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

            # Initialize logging parameters    
            dash_lines = []
            u_headway_log = {}
            u_barrier_log = {}
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

            # Determine Global Nominal Speed (Ramping)
            if step_count <= WARMUP_STEPS: current_nominal_speed = 0.0
            elif step_count <= (WARMUP_STEPS + RAMP_STEPS):
                progress = (step_count - WARMUP_STEPS) / RAMP_STEPS
                current_nominal_speed = TARGET_SPEED_KMH * progress
            else: current_nominal_speed = TARGET_SPEED_KMH

            for i, vid in enumerate(sorted_agents):
                veh = vehicles_by_id[vid]

                # 3. Headway Controller 
                # Initialie parameters
                u_headway = 0.0
                headway_error = 0.0
                gap_meters = TOTAL_TRACK_LENGTH 
                leader_id = None

                # Per vehicle calculation only if more than 1 one vehicles, otherwise gap meters is the total track length
                if N > 1:
                    leader_idx = (i + 1) % N  # Leader is vehicle in front, %N handles wraparound           
                    leader_id = sorted_agents[leader_idx]
                    gap_meters = (current_s_values[leader_id] - current_s_values[vid]) % TOTAL_TRACK_LENGTH # Calculating gap in meters using the distance mapper
                    
                    # Use EMA to smooth the gap meters error
                    if gap_ema[vid] is None:    
                        gap_ema[vid] = gap_meters
                    else:
                        gap_ema[vid] = GAP_EMA_ALPHA * gap_ema[vid] + (1.0 - GAP_EMA_ALPHA) * gap_meters
                    gap_meters = gap_ema[vid]
                    headway_error = gap_meters - TARGET_GAP_METERS    # Calculate headway error by subtracting the target value

                # Logging gap meters and headway error per vehicle
                gap_meters_log[vid] = gap_meters
                headway_error_log[vid] = headway_error

                #  4. Combine Controls 
                # Calculating barrier correction based on gain and gradient, limited to the max correction
                u_barrier_raw = GAIN_BARRIER * barrier_controls[vid]
                u_barrier = max(min(u_barrier_raw, MAX_BARRIER_KMH), -MAX_BARRIER_KMH)
                u_barrier_log[vid] = u_barrier
                
                # Speeding up to taregt soeed during warmup and rampup
                if step_count <= WARMUP_STEPS:
                    target_speed = 0.0
                    prev_target_speed[vid] = 0.0
                    agent_pids[vid].reset()
                    agent_gap_pids[vid].reset()
                elif step_count <= WARMUP_STEPS + RAMP_STEPS:
                    target_speed = current_nominal_speed + u_barrier
                    if target_speed < 0: target_speed = 0
                    prev_target_speed[vid] = target_speed
                    # After warmup and rampup use also headway controller
                else:
                    if N > 1 and leader_id is not None:
                        # Use Headway PD for headway correction
                        u_headway = agent_gap_pids[vid].run(
                                gap_meters,
                                TARGET_GAP_METERS,
                                DT,
                            )
                        if MAX_HEADWAY_KMH > 0.0:
                            u_headway = float(np.clip(u_headway, -MAX_HEADWAY_KMH, MAX_HEADWAY_KMH)) # Limit headway correction for realism

                    u_headway_log[vid] = u_headway
                    target_speed = current_nominal_speed + u_headway + u_barrier

                    if target_speed < 0: target_speed = 0
                    max_delta_kmh = COMFORTABLE_MAX_ACCEL_MS2 * DT * 3.6
                    prev = prev_target_speed.get(vid, target_speed)
                    target_speed = float(np.clip(target_speed, prev - max_delta_kmh, prev + max_delta_kmh)) # Capping speed correction at max comfortable acceleration, but vehicles do not adhere completely
                    prev_target_speed[vid] = target_speed
                target_speed_log[vid] = target_speed

                # 5. Lateral Control 
                tracker = agent_trackers[vid]
                dt_lookahead = 15.0 / R 
                t_future = tracker.t + dt_lookahead
                math_target = tracker.get_math_location(t_future)
                wp_target = carla_map.get_waypoint(math_target, project_to_road=True, lane_type=carla.LaneType.Driving)
                steer_target = wp_target.transform.location if wp_target else math_target 
                
                curr_loc = veh.get_location()
                veh_trans = veh.get_transform()
                veh_fwd = veh_trans.get_forward_vector()
                vec_to = np.array([steer_target.x - curr_loc.x, steer_target.y - curr_loc.y])
                norm = np.linalg.norm(vec_to)
                if norm > 0: vec_to /= norm
                cross = veh_fwd.x*vec_to[1] - veh_fwd.y*vec_to[0]
                steer_cmd = np.clip(cross, -1.0, 1.0)
                
                

                #  6. Actuation and Energy Calculation
                curr_speed = velocities[vid] # Retrieving speed from vehicle
                v_ms = curr_speed / 3.6 # Converting speed to m/s
                raw_acceleration = (v_ms - prev_v_ms.get(vid, v_ms)) / DT # Calculating raw acceleration
                prev_v_ms[vid] = v_ms
                raw_accelerations[vid] = raw_acceleration
                accelerations[vid] = update_ema(accelerations.get(vid), raw_acceleration, ACCEL_EMA_ALPHA) # Smooth acceleration
                
                # Energy calculation based on specific vehicle type
                bp_id = vehicle_blueprints[vid]
                cfg = VEHICLE_CONFIGS.get(bp_id)
                # Computing forces for mechanical power
                F_inertia  = cfg['mass'] * accelerations[vid]
                F_aero     = 0.5 * AIR_DENSITY * cfg['drag_coefficient'] * cfg['frontal_area'] * v_ms**2
                F_rolling  = cfg['mass'] * GRAVITATIONAL_ACCELERATION * cfg['rolling_resistance']
                F_total    = F_inertia + F_aero + F_rolling
                Power_mech = F_total * v_ms

                # Calculating motor speed
                motor_rpm = cfg['gear_ratio'] * max(v_ms, 0.0) * 60.0 / (2.0 * math.pi * cfg['wheel_radius'])

                # Calculating motor torque
                if v_ms > 0.0:
                    if F_total >= 0:
                        motor_torque = (F_total * cfg['wheel_radius'] / cfg['gear_ratio']) / cfg['gearbox_eff']
                    else:
                        motor_torque = (F_total * cfg['wheel_radius'] / cfg['gear_ratio']) * cfg['gearbox_eff']
                else:
                    motor_torque = 0.0
                
                eff_map = MOTOR_EFF_MAPS.get(bp_id)
                rpm_q = float(np.clip(motor_rpm, 0.0, 18000.0))
                tq_q = float(np.clip(abs(motor_torque), 0.0, 375.0))
                eta = max(float(eff_map([[rpm_q, tq_q]])[0]), 0.60) # If value is outside the efficiency map a low efficiency of 0.6 is assigned

                # Compute power demand based on propulsion or regeneration
                if Power_mech >= 0:
                    power_demands[vid] = Power_mech / (cfg['gearbox_eff'] * eta) 
                else:
                    power_demands[vid] = Power_mech * cfg['gearbox_eff'] * eta 

                # Control the vehicle
                control = carla.VehicleControl()
                control.steer = float(steer_cmd)

                # During warmup vehicle is held at spawn position
                if step_count <= WARMUP_STEPS:
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.hand_brake = True
                # During rampup throttle and brake are based on longitudinal controller    
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

                # Updating the lap timing after rampup
                if lap_csv_writer is not None and ramp_complete:
                    dist_from_ramp = curr_loc.distance(ramp_locs[vid])
                    time_since_last_lap = elapsed_time - lap_started_at[vid]

                    # Starting lap if outside lap clear radius
                    if dist_from_ramp > LAP_CLEAR_RADIUS:
                        lap_trigger_armed[vid] = True
                    # Count lap if within lap trigger radius
                    elif (
                        lap_trigger_armed[vid]
                        and dist_from_ramp <= LAP_TRIGGER_RADIUS
                        and time_since_last_lap >= LAP_MIN_TIME
                    ):
                        lap_counts[vid] += 1
                    
                    # Calculate lap time from start time and elapsed time
                        lap_start_time = lap_started_at[vid]
                        lap_duration = elapsed_time - lap_start_time
                        delay_time = lap_duration - MINIMAL_LAP_TIME  # Calculate delay time
                        total_delay_s += delay_time # Cumulative delay for all vehicles

                        # Write to lap csv
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
                    f"{vid:<4} | {gap_meters:6.1f} | {headway_error:7.2f} | {u_headway:6.2f} | {u_barrier:6.2f} | {curr_speed:5.1f} | {raw_accelerations[vid]:6.2f} | {throughput:10.3f} | {power_demands[vid]/1000:7.2f} kW | {safety_flag:>9}"
                )
            # Final calculations at end of simulation run
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
                }
                print(f"Run {run_index} completed at {elapsed_time:.2f} s")
                break

            # Logging throughput CSV, calculating once per timestep but writing every second for speed 
            if throughput_csv_writer is not None:
                total_energy_kWh += sum(power_demands.values()) * DT / (1000 * 3600)
                total_distance_km += sum(velocities.values()) * DT / 3600
                avg_speed = sum(velocities.values()) / len(velocities) if velocities else 0.0
                safety_violation_rate = total_safety_violations / max(elapsed_time, 1)
                u_barrier_firings = sum(1 for v in u_barrier_log.values() if v != 0.0)
                u_barrier_avg = sum(u_barrier_log.values()) / len(u_barrier_log) if u_barrier_log else 0.0
                u_headway_avg = sum(u_headway_log.values()) / len(u_headway_log) if u_headway_log else 0.0
                throughput_csv_writer.writerow([
                    f"{elapsed_time:.2f}", f"{throughput:.6f}", f"{avg_speed:.2f}",
                    f"{total_energy_kWh:.6f}", f"{total_distance_km:.4f}",
                    f"{safety_violation_rate:.6f}", f"{total_delay_s:.2f}",
                    f"{u_barrier_firings}", f"{u_barrier_avg:.4f}", f"{u_headway_avg:.4f}",
                ])
                if step_count % int(1 / DT) == 0:
                    throughput_csv_file.flush()

            # Logging vehicle CSV, calculating once per timestep but writing every second for speed 
            if vehicle_csv_writer is not None and tracked_vehicle_ids:
                for tracked_vid in tracked_vehicle_ids:
                    if tracked_vid not in power_demands:
                        continue
                    vehicle_csv_writer.writerow([
                        f"{elapsed_time:.2f}",
                        tracked_vid,
                        vehicle_blueprints.get(tracked_vid, ""),
                        f"{velocities[tracked_vid] / 3.6:.4f}",
                        f"{velocities[tracked_vid]:.4f}",
                        f"{accelerations.get(tracked_vid, 0.0):.4f}",
                        f"{raw_accelerations.get(tracked_vid, 0.0):.4f}",
                        f"{gap_meters_log.get(tracked_vid, TOTAL_TRACK_LENGTH):.4f}",
                        f"{TARGET_GAP_METERS:.4f}",
                        f"{headway_error_log.get(tracked_vid, 0.0):.4f}",
                        f"{u_headway_log.get(tracked_vid, 0.0):.4f}",
                        f"{u_barrier_log.get(tracked_vid, 0.0):.4f}",
                        f"{power_demands[tracked_vid] / 1000:.4f}",
                    ])
                if step_count % int(1 / DT) == 0:
                    vehicle_csv_file.flush()

            #  7. Visualization in terminal every 5  timesteps
            if step_count % 5 == 0:
                print("\033[H\033[J")
                if step_count <= WARMUP_STEPS: status = "SETTLING..."
                elif step_count <= WARMUP_STEPS + RAMP_STEPS: status = f"RAMPING UP ({current_nominal_speed:.1f} km/h)"
                else: status = "ACTIVE"
                throughput = total_crossings / max(elapsed_time, 1)
                safety_violation_rate = total_safety_violations / max(elapsed_time, 1)
                print(f"--- RUN {run_index} | STEP {step_count} [{status}] ---")
                print(f"Throughput: {throughput:.4f} cross/s | SafetyViol: {safety_violation_rate:.4f} viol/s ({total_safety_violations}) | Energy: {total_energy_kWh:+.4f} kWh | Target gap: {TARGET_GAP_METERS:.1f} m")
                print(f"{'ID':<4} | {'Gap(m)':<6} | {'Err(m)':<7} | {'u_Head':<6} | {'u_Bar':<6} | {'Speed':<5} | {'Accel':>6} | {'Throughput':>10} | {'Power':>10} | {'Safety':>9}")
                for line in dash_lines: print(line)

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
            print(f"Lap log saved to {lap_csv_filename}")
    
        for v in vehicles_list:
            try: v.destroy()
            except Exception: pass
        if original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass

    return summary

# Simulation setup
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--number-of-vehicles', default=21, type=int)
    argparser.add_argument('--radius', default=250, type=float)
    argparser.add_argument('--duration-seconds', default=1800.0, type=float)
    argparser.add_argument('--output-dir', default=RESULTS_DIR)
    argparser.add_argument('--start-run', default=1, type=int)
    argparser.add_argument('--end-run', default=FIXED_RUN_COUNT, type=int)
    argparser.add_argument('--max-headway-kmh', default=10.0, type=float)
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
                except RuntimeError as e:
                    print(f"\n[ERROR] Simulator crash on run {run_index} attempt {attempt}: {e}")
                    if os.path.exists(run_output_dir):
                        shutil.rmtree(run_output_dir)
                        print(f"Removed partial output: {run_output_dir}")
                    if attempt == MAX_RUN_RETRIES:
                        print(f"Run {run_index} failed after {MAX_RUN_RETRIES} attempts, skipping.")
                        break
                    wait_for_simulator()
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

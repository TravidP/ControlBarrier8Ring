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

CARLA_SERVER_CMD = [
    '/home/rug/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor',
    '/home/rug/carla/Unreal/CarlaUE4/CarlaUE4.uproject',
    '-vulkan',
]
CARLA_MAP_NAME = '8ring'
CARLA_START_WAIT_S = 180  # UE4Editor is slow to start


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
    Uses a 3:1 Lissajous curve: x(t) = -R*cos(3t), y(t) = R*sin(t).
    Two self-intersections at (0, +R/2) and (0, -R/2), each approached from two arms.
    """
    # t-values at which the curve passes through a self-intersection (4 per lap)
    CROSSING_T_VALUES = np.array([np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6])

    def __init__(self, R, resolution=100000):
        self.R = R
        self.t_vals = np.linspace(0, 2 * np.pi, resolution)

        x_vals = -self.R * np.cos(3 * self.t_vals)
        y_vals =  self.R * np.sin(self.t_vals)

        dists = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
        self.s_vals = np.concatenate(([0], np.cumsum(dists)))
        self.total_length = self.s_vals[-1]

        # Arc-length at each of the 4 crossing t-values
        self.crossing_s = np.array([
            self.s_vals[np.argmin(np.abs(self.t_vals - tc))]
            for tc in self.CROSSING_T_VALUES
        ])
        # For each intersection: (s_approach_A, s_approach_B)
        self.intersection_approach_s = [
            (self.crossing_s[0], self.crossing_s[1]),  # upper: (0, +R/2)
            (self.crossing_s[2], self.crossing_s[3]),  # lower: (0, -R/2)
        ]

    def get_s_from_t(self, t):
        t = t % (2 * np.pi)
        return np.interp(t, self.t_vals, self.s_vals)

    def get_t_from_s(self, s):
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

        dx = location.x - (-self.R * np.cos(3 * t_wrapped))
        dy = location.y - ( self.R * np.sin(t_wrapped))

        d2 = dx**2 + dy**2
        best_idx = np.argmin(d2)
        self.t = t_search[best_idx]           # store unwrapped
        return self.t % (2 * np.pi)           # callers expect [0, 2π)

    def get_math_location(self, t_val):
        tx = -self.R * np.cos(3 * t_val)
        ty =  self.R * np.sin(t_val)
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


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_results", "BarrierControlResults")
FIXED_RUN_COUNT = 1
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


def get_signed_delta_s(current_s, previous_s, track_length):
    """Shortest signed progress delta on a closed track."""
    delta_s = current_s - previous_s
    half_track = track_length / 2.0
    if delta_s > half_track:
        delta_s -= track_length
    elif delta_s < -half_track:
        delta_s += track_length
    return delta_s


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
        "steady_state",
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

    # Headway controller. Units: P -> km/h per meter of gap error,
    # D -> km/h per km/h of relative geodesic speed.
    GAIN_HEADWAY_P =  3.50 # 3.50
    GAIN_HEADWAY_I =  0.00
    GAIN_HEADWAY_D =  1.00
    GAIN_INTEGRAL_LIMIT =  3.0  # larger limit to allow meaningful accumulation
    MAX_HEADWAY_KMH = 10.0 

    # The D term uses a smoothed geodesic speed derived from track-distance progress.
    # This matches the derivative of headway error better than raw world-speed magnitude.
    VELOCITY_EMA_TAU_S = 0.95
    VELOCITY_EMA_ALPHA = DT / (DT + VELOCITY_EMA_TAU_S)

    # Steady-state detection
    STEADY_STATE_WINDOW_S = 30.0       # rolling window length (seconds)
    STEADY_STATE_SPEED_STD_KMH = 0.5  # max fleet-speed std dev to declare steady state
    STEADY_STATE_SPEED_ERR_KMH = 2.0  # fleet mean must be within this of target speed

    # Intersection barrier logic
    GAIN_BARRIER = 25.0
    BARRIER_ACTIVATION_DIST = min(TARGET_GAP_METERS / 2.0, 10)
    APPROACH_WINDOW_M = 20 # [Request: closer to intersection point in 10 meters]
    EPSILON_DIFF = 1.0
    MAX_BARRIER_KMH = 7.0 # Maximum speed up

    # Lap timing
    LAP_MIN_TIME = 200.0
    MINIMAL_LAP_TIME = min(TOTAL_TRACK_LENGTH / (TARGET_SPEED_KMH / KMH_PER_MS), 281.78) # 281.78 found running simulation with N=1
    LAP_TRIGGER_RADIUS = 1.0
    LAP_CLEAR_RADIUS = 2.0

    # Safety distance violation parametersdREGEN
    SAFETY_DISTANCE_M = 2.5  # minimum allowed BB-to-BB gap at intersection (m)

    # Energy consumption parameters Tesla Model 3
    MASS_TESLA_MODEL3 = 1791.5
    AIR_DENSITY = 1.225
    DRAG_COEFFICIENT_TESLA = 0.23
    FRONTAL_AREA_TESLA = 2.22
    ROLLING_RESISTANCE_COEFFICIENT_TESLA = 0.010
    GRAVITATIONAL_ACCELERATION = 9.81
    TESLA_REGEN_EFFICIENCY = 0.80
    TESLA_DRIVETRAIN_EFFICIENCY = 0.91

    # Energy consumption parameters Audi Etron
    MASS_AUDI_ETRON = 2490.0
    DRAG_COEFFICIENT_ETRON = 0.28
    FRONTAL_AREA_ETRON = 2.6
    ROLLING_RESISTANCE_COEFFICIENT_ETRON = 0.010
    AUDI_REGEN_EFFICIENCY = 0.80
    AUDI_DRIVETRAIN_EFFICIENCY = 0.88

    energy_params_by_blueprint = {
        'vehicle.tesla.model3': {
            'mass': MASS_TESLA_MODEL3,
            'drag_coefficient': DRAG_COEFFICIENT_TESLA,
            'frontal_area': FRONTAL_AREA_TESLA,
            'rolling_resistance': ROLLING_RESISTANCE_COEFFICIENT_TESLA,
            'drivetrain_efficiency': TESLA_DRIVETRAIN_EFFICIENCY,
            'regen_efficiency': TESLA_REGEN_EFFICIENCY,
        },
        'vehicle.audi.etron': {
            'mass': MASS_AUDI_ETRON,
            'drag_coefficient': DRAG_COEFFICIENT_ETRON,
            'frontal_area': FRONTAL_AREA_ETRON,
            'rolling_resistance': ROLLING_RESISTANCE_COEFFICIENT_ETRON,
            'drivetrain_efficiency': AUDI_DRIVETRAIN_EFFICIENCY,
            'regen_efficiency': AUDI_REGEN_EFFICIENCY,
        },
    }

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    vehicles_list = []

    throughput_csv_file = None
    throughput_csv_writer = None
    throughput_csv_filename = None

    vehicle_csv_file = None
    vehicle_csv_writer = None
    vehicle_csv_filename = None

    lap_csv_file = None
    lap_csv_writer = None
    lap_csv_filename = None

    tracked_vehicle_ids = []
    summary = None
    original_settings = None
    plot_vehicle_ids = []
    gap_history = {}
    gap_time_history = []
    speed_history_plot = {}
    target_speed_history_plot = {}

    try:
        world = client.get_world()
        carla_map = world.get_map()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        # Position spectator at the figure-8 intersection (0, 0) looking straight down
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=0.0, y=0.0, z=300.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        ))

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

        # Phase shifts the entire formation to keep N=4 away from intersections
        start_dist_offset = TARGET_GAP_METERS / 3.0

        for i in range(N):
            noise = rng.uniform(-0.2, 0.2) * TARGET_GAP_METERS
            s_target = (i * TARGET_GAP_METERS + start_dist_offset + noise) % TOTAL_TRACK_LENGTH
            t_spawn = geom.get_t_from_s(s_target)

            start_x = -R * np.cos(3 * t_spawn)
            start_y =  R * np.sin(t_spawn)
            tang = np.array([3.0 * np.sin(3 * t_spawn), np.cos(t_spawn)])
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
            vehicle_energy_params[veh.id] = energy_params_by_blueprint[bp_id]
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
        print("Simulation Running...")

        # Lap and crossing state
        lap_counts = {vid: 0 for vid in agent_ids}
        lap_started_at = {vid: 0.0 for vid in agent_ids}
        lap_trigger_armed = {vid: False for vid in agent_ids}
        ramp_locs = {}
        t_previous = {vid: agent_trackers[vid].t for vid in agent_ids}
        total_crossings = 0
        safety_violation_active = {vid: False for vid in agent_ids}

        # Speed, acceleration, and energy state
        ACCEL_EMA_TAU_S = 0.95
        ACCEL_ALPHA = DT / (DT + ACCEL_EMA_TAU_S)
        accelerations = {}
        raw_accelerations = {}
        filtered_path_speeds = {}
        power_demands = {}
        prev_s_values = {}
        prev_v_ms = {}
        smooth_acceleration = {}

        # Aggregate run metrics
        avg_speed = 0.0
        elapsed_time = 0.0
        total_energy_kWh = 0.0
        total_distance_km = 0.0
        total_delay_s = 0.0
        total_safety_violations = 0
        throughput = 0.0
        safety_violation_rate = 0.0

        # Controller and steady-state state
        GAP_EMA_TAU = 0.95
        gap_ema = {vid: None for vid in agent_ids}
        prev_target_speed = {}
        prev_steer = {}
        ramp_complete = False
        speed_history = deque(maxlen=int(STEADY_STATE_WINDOW_S / DT))
        steady_state_reached = False
        steady_state_time = None

        # Gap history for end-of-run plot (first 4 vehicles)
        plot_vehicle_ids = agent_ids[:min(4, len(agent_ids))]
        gap_history = {vid: [] for vid in plot_vehicle_ids}
        speed_history_plot = {vid: [] for vid in plot_vehicle_ids}
        target_speed_history_plot = {vid: [] for vid in plot_vehicle_ids}
        barrier_firings_history = []

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
                # Don't re-search while stationary: physics jitter shifts t slightly
                # each tick even with hand-brake on, causing gap_meters to flutter.
                if step_count > WARMUP_STEPS:
                    t_new = agent_trackers[veh.id].update_t(loc)
                else:
                    t_new = agent_trackers[veh.id].t % (2 * np.pi)

                s = geom.get_s_from_t(t_new)
                current_s_values[veh.id] = s

                v = veh.get_velocity()
                speed = KMH_PER_MS * math.sqrt(v.x**2 + v.y**2)
                velocities[veh.id] = speed

                prev_s = prev_s_values.get(veh.id)
                if prev_s is None:
                    path_speed = speed
                else:
                    delta_s = get_signed_delta_s(s, prev_s, TOTAL_TRACK_LENGTH)
                    path_speed = (delta_s / DT) * KMH_PER_MS
                prev_s_values[veh.id] = s

                filtered_path_speeds[veh.id] = update_ema(
                    filtered_path_speeds.get(veh.id),
                    path_speed,
                    VELOCITY_EMA_ALPHA,
                )

                if step_count > WARMUP_STEPS:
                    old_t = t_previous[veh.id]
                    # Pass through t = π/6 (first intersection crossing)
                    if old_t <= np.pi/6 < t_new:
                        total_crossings += 1
                    elif t_new < old_t and (np.pi/6 >= old_t or np.pi/6 < t_new):
                        total_crossings += 1
                    # Pass through t = 7π/6 (second intersection crossing)
                    if old_t <= 7*np.pi/6 < t_new:
                        total_crossings += 1
                    elif t_new < old_t and (7*np.pi/6 >= old_t or 7*np.pi/6 < t_new):
                        total_crossings += 1
                    t_previous[veh.id] = t_new

            sorted_agents = sorted(agent_ids, key=lambda vid: current_s_values[vid])

            # --- 2. Barrier Control + Safety Distance Check (Intersection Logic) ---
            conflict_pairs = []
            barrier_controls = {vid: 0.0 for vid in agent_ids}
            active_conflict_vids = set()

            if step_count > WARMUP_STEPS:
                for s_a, s_b in geom.intersection_approach_s:
                    cands_a = [vid for vid in agent_ids
                               if (s_a - APPROACH_WINDOW_M) < current_s_values[vid] < s_a]
                    cands_b = [vid for vid in agent_ids
                               if (s_b - APPROACH_WINDOW_M) < current_s_values[vid] < s_b]

                    if not cands_a or not cands_b:
                        continue

                    ca = max(cands_a, key=lambda vid: current_s_values[vid])
                    cb = max(cands_b, key=lambda vid: current_s_values[vid])
                    conflict_pairs.append((ca, cb))

                    dist_to_cross_a = s_a - current_s_values[ca]
                    dist_to_cross_b = s_b - current_s_values[cb]
                    diff_meters = dist_to_cross_b - dist_to_cross_a

                    if abs(diff_meters) < BARRIER_ACTIVATION_DIST:
                        safe_diff = diff_meters if abs(diff_meters) >= EPSILON_DIFF \
                            else EPSILON_DIFF * (1.0 if diff_meters >= 0 else -1.0)
                        gradient = -1.0 / safe_diff
                        barrier_controls[cb] += gradient
                        barrier_controls[ca] += -gradient

                    # Safety: BB check on this conflict pair
                    bb_dist_conflict = get_bb_distance(vehicles_by_id[ca], vehicles_by_id[cb])
                    if bb_dist_conflict < SAFETY_DISTANCE_M:
                        if not safety_violation_active[ca] and not safety_violation_active[cb]:
                            total_safety_violations += 1
                        safety_violation_active[ca] = True
                        safety_violation_active[cb] = True
                        active_conflict_vids.add(ca)
                        active_conflict_vids.add(cb)
                    else:
                        safety_violation_active[ca] = False
                        safety_violation_active[cb] = False

                for vid in agent_ids:
                    if vid not in active_conflict_vids:
                        safety_violation_active[vid] = False

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

            # Determine Global Nominal Speed (Ramping)
            if step_count <= WARMUP_STEPS: current_nominal_speed = 0.0
            elif step_count <= (WARMUP_STEPS + RAMP_STEPS):
                progress = (step_count - WARMUP_STEPS) / RAMP_STEPS
                current_nominal_speed = TARGET_SPEED_KMH * progress
            else: current_nominal_speed = TARGET_SPEED_KMH

            for i, vid in enumerate(sorted_agents):
                veh = vehicles_by_id[vid]

                # --- 3. Headway Sensing ---
                u_headway = 0.0
                gap_meters = TOTAL_TRACK_LENGTH
                headway_error = 0.0
                leader_id = None

                if N > 1:
                    leader_idx = (i + 1) % len(sorted_agents)
                    leader_id = sorted_agents[leader_idx]
                    gap_meters = (current_s_values[leader_id] - current_s_values[vid]) % TOTAL_TRACK_LENGTH
                    if gap_ema[vid] is None:
                        gap_ema[vid] = gap_meters
                    else:
                        gap_ema[vid] = GAP_EMA_TAU * gap_ema[vid] + (1.0 - GAP_EMA_TAU) * gap_meters
                    gap_meters = gap_ema[vid]
                    headway_error = gap_meters - TARGET_GAP_METERS

                gap_meters_log[vid] = gap_meters
                headway_error_log[vid] = headway_error

                # --- 4. Combine Controls ---
                u_barrier_raw = GAIN_BARRIER * barrier_controls[vid]
                u_barrier = max(min(u_barrier_raw, MAX_BARRIER_KMH), -MAX_BARRIER_KMH)
                u_barrier_log[vid] = u_barrier

                if step_count <= WARMUP_STEPS:
                    target_speed = 0.0
                    prev_target_speed[vid] = 0.0
                    agent_pids[vid].reset()
                    agent_gap_pids[vid].reset()
                elif step_count <= WARMUP_STEPS + RAMP_STEPS:
                    target_speed = current_nominal_speed + u_barrier
                    if target_speed < 0: target_speed = 0
                    prev_target_speed[vid] = target_speed
                else:
                    if N > 1 and leader_id is not None:
                        relative_velocity = filtered_path_speeds[leader_id] - filtered_path_speeds[vid]
                        u_headway = agent_gap_pids[vid].run(
                            gap_meters,
                            TARGET_GAP_METERS,
                            DT,
                            derivative_input=relative_velocity,
                        )
                        if MAX_HEADWAY_KMH > 0.0:
                            u_headway = float(np.clip(u_headway, -MAX_HEADWAY_KMH, MAX_HEADWAY_KMH))
                    u_headway_log[vid] = u_headway
                    target_speed = current_nominal_speed + u_headway + u_barrier
                    if target_speed < 0: target_speed = 0
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

                dt_lookahead = 15.0 / R
                t_future = tracker.t + dt_lookahead
                math_target = tracker.get_math_location(t_future)
                wp_target = carla_map.get_waypoint(math_target, project_to_road=True, lane_type=carla.LaneType.Driving)
                wp_curr = carla_map.get_waypoint(curr_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

                # Discard wp_target that snapped to the opposite-direction road at the intersection.
                if wp_target is not None:
                    wp_fwd = wp_target.transform.get_forward_vector()
                    if veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y < 0.0:
                        wp_target = None

                at_junction = (wp_curr is not None and wp_curr.is_junction) or (wp_target is not None and wp_target.is_junction)
                if at_junction or wp_target is None:
                    # Steer from Lissajous math tangent (dx/dt = 3R·sin(3t), dy/dt = R·cos(t))
                    tang_x = 3.0 * math.sin(3.0 * t_future)
                    tang_y = math.cos(t_future)
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
                raw_accelerations[vid] = raw_acceleration
                smooth_acceleration[vid] = update_ema(
                    smooth_acceleration.get(vid),
                    raw_acceleration,
                    ACCEL_ALPHA,
                )
                acceleration = smooth_acceleration[vid]
                accelerations[vid] = acceleration
                prev_v_ms[vid] = v_ms
                energy_params = vehicle_energy_params[vid]
                Power_mech = (
                    energy_params['mass'] * acceleration
                    + 0.5 * AIR_DENSITY * energy_params['drag_coefficient'] * energy_params['frontal_area'] * v_ms**2
                    + energy_params['mass'] * GRAVITATIONAL_ACCELERATION * energy_params['rolling_resistance']
                ) * v_ms
                if Power_mech >= 0:
                    power_demands[vid] = Power_mech / energy_params['drivetrain_efficiency']
                else:
                    power_demands[vid] = Power_mech * energy_params['regen_efficiency']

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
                    f"{vid:<4} | {gap_meters:6.1f} | {headway_error:7.2f} | {u_headway:6.2f} | {u_barrier:6.2f} | {curr_speed:5.1f} | {accelerations[vid]:6.2f} | {throughput:10.3f} | {power_demands[vid]/1000:7.2f} kW | {safety_flag:>9}"
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
                if not crash_detected:
                    for c1, c2 in conflict_pairs:
                        if get_bb_distance(vehicles_by_id[c1], vehicles_by_id[c2]) <= 0.0:
                            print(f"\n*** CRASH at t={elapsed_time:.2f}s: vehicle {c1} and vehicle {c2} "
                                  f"bounding boxes overlap — stopping simulation ***")
                            crash_detected = True
                            break
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
                u_barrier_firings = sum(1 for v in u_barrier_log.values() if v != 0.0)
                u_barrier_avg = sum(u_barrier_log.values()) / len(u_barrier_log) if u_barrier_log else 0.0
                u_headway_avg = sum(u_headway_log.values()) / len(u_headway_log) if u_headway_log else 0.0
                throughput_csv_writer.writerow([
                    f"{elapsed_time:.2f}", f"{throughput:.6f}", f"{avg_speed:.2f}",
                    f"{total_energy_kWh:.6f}", f"{total_distance_km:.4f}",
                    f"{safety_violation_rate:.6f}", f"{total_delay_s:.2f}",
                    f"{u_barrier_firings}", f"{u_barrier_avg:.4f}", f"{u_headway_avg:.4f}",
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
                        f"{u_barrier_log.get(tracked_vid, 0.0):.4f}",
                        f"{power_demands[tracked_vid] / 1000:.4f}",
                    ])
                if step_count % int(1 / DT) == 0:
                    vehicle_csv_file.flush()

            # Gap history for end-of-run plot (every simulated second)
            if step_count % int(1 / DT) == 0 and gap_meters_log:
                gap_time_history.append(elapsed_time)
                for pv in plot_vehicle_ids:
                    gap_history[pv].append(gap_meters_log.get(pv, TOTAL_TRACK_LENGTH))
                    speed_history_plot[pv].append(velocities.get(pv, 0.0))
                    target_speed_history_plot[pv].append(target_speed_log.get(pv, 0.0))
                barrier_firings_history.append(sum(1 for v in u_barrier_log.values() if v != 0.0))

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
        if gap_time_history and plot_vehicle_ids:
            fig, ax = plt.subplots(figsize=(12, 5))
            for vid in plot_vehicle_ids:
                ax.plot(gap_time_history, gap_history[vid], label=f"Vehicle {vid}")
            ax.axhline(TARGET_GAP_METERS, color='k', linestyle='--', linewidth=1.5,
                       label=f"Target ({TARGET_GAP_METERS:.1f} m)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Gap (m)")
            ax.set_title(f"Gap vs Target Gap — first {len(plot_vehicle_ids)} vehicles")
            ax2 = ax.twinx()
            ax2.bar(gap_time_history, barrier_firings_history, width=1.0,
                    alpha=0.25, color='orange', label='Barrier firings')
            ax2.set_ylabel("Barrier firings (vehicles)", color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax.grid(True)
            plot_path = os.path.join(run_output_dir, "gap_plot.png")
            fig.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"Gap plot saved to {plot_path}")

            fig2, ax3 = plt.subplots(figsize=(12, 5))
            prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for idx, vid in enumerate(plot_vehicle_ids):
                color = prop_cycle[idx % len(prop_cycle)]
                ax3.plot(gap_time_history, speed_history_plot[vid],
                         color=color, label=f"Vehicle {vid}")
            ax3.axhline(TARGET_SPEED_KMH, color='k', linestyle='--', linewidth=1.5,
                        label=f"Target ({TARGET_SPEED_KMH:.0f} km/h)")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Speed (km/h)")
            ax3.set_title(f"Speed vs Target Speed — first {len(plot_vehicle_ids)} vehicles")
            ax4 = ax3.twinx()
            ax4.bar(gap_time_history, barrier_firings_history, width=1.0,
                    alpha=0.25, color='orange', label='Barrier firings')
            ax4.set_ylabel("Barrier firings (vehicles)", color='orange')
            ax4.tick_params(axis='y', labelcolor='orange')
            lines3, labels3 = ax3.get_legend_handles_labels()
            lines4, labels4 = ax4.get_legend_handles_labels()
            ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')
            ax3.yaxis.set_minor_locator(MultipleLocator(2.5))
            ax3.grid(True)
            ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
            speed_plot_path = os.path.join(run_output_dir, "speed_plot.png")
            fig2.savefig(speed_plot_path, dpi=120, bbox_inches='tight')
            plt.close(fig2)
            print(f"Speed plot saved to {speed_plot_path}")
        for v in vehicles_list:
            try: v.destroy()
            except Exception: pass
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
    argparser.add_argument('--gain-p', default=0.90, type=float)
    argparser.add_argument('--gain-i', default=0.0, type=float)
    argparser.add_argument('--gain-d', default=1.10, type=float)
    argparser.add_argument('--gain-integral-limit', default=0.0, type=float)
    argparser.add_argument('--max-headway-kmh', default=20.0, type=float)
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
                except RuntimeError as e:
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

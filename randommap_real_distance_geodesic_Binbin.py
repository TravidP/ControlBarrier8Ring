#!/usr/bin/env python
# Copyright (c) 2025 Computer Vision Center (CVC)
# Custom Track Simulation: Distance-Based Control with Barrier Function

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

import carla
import networkx as nx
import numpy as np

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
    if hasattr(self, '_wmap'):
        topology = self._wmap.get_topology()
    else:
        topology = self._map.get_topology()
    for segment in topology:
        entry_wp, exit_wp = segment
        vec = exit_wp.transform.location - entry_wp.transform.location
        length = math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)
        if length > 0.001:
            net_carla_vector = carla.Vector3D(vec.x/length, vec.y/length, vec.z/length)
        else:
            net_carla_vector = carla.Vector3D(0, 0, 0)
        self._graph.add_node(entry_wp.id, vertex=entry_wp)
        self._graph.add_node(exit_wp.id, vertex=exit_wp)
        edge = {"entry_waypoint": entry_wp, "exit_waypoint": exit_wp,
                "entry_vector": net_carla_vector, "length": length, "path": []}
        self._graph.add_edge(entry_wp.id, exit_wp.id, **edge)
        if entry_wp.road_id not in self._road_id_to_edge:
            self._road_id_to_edge[entry_wp.road_id] = {}
        if entry_wp.section_id not in self._road_id_to_edge[entry_wp.road_id]:
            self._road_id_to_edge[entry_wp.road_id][entry_wp.section_id] = {}
        self._road_id_to_edge[entry_wp.road_id][entry_wp.section_id][entry_wp.lane_id] = edge
        self._lane_id_to_edge[entry_wp.id] = edge

GlobalRoutePlanner._build_graph = patched_build_graph

# 2. HELPER CLASSES

class DistanceMapper:
    """
    Arc-length mapper for the custom piecewise track 
    """
    TARGET_LENGTH = 650.0
    _N_ARC = 80
    _N_LINE = 80
    _R_ARC = 0.5

    def __init__(self):
        X, Y = self._build_raw_centerline()

        Xc = np.append(X, X[0])
        Yc = np.append(Y, Y[0])
        sf = self.TARGET_LENGTH / np.sqrt(np.diff(Xc)**2 + np.diff(Yc)**2).sum()
        X = X * sf
        Y = Y * sf

        Xc = np.append(X, X[0])
        Yc = np.append(Y, Y[0])
        ds = np.sqrt(np.diff(Xc)**2 + np.diff(Yc)**2)
        self.s_vals = np.concatenate(([0.0], np.cumsum(ds)))
        self.total_length = float(self.s_vals[-1])
        self.X = X
        self.Y = Y
        self.N = len(X)

        dxV = np.roll(X, -1) - np.roll(X, 1)
        dyV = np.roll(Y, -1) - np.roll(Y, 1)
        self.hdg = np.arctan2(dyV, dxV)

        cx, cy = 2.0 * sf, 2.0 * sf
        dists = np.hypot(X - cx, Y - cy)
        order = np.argsort(dists)
        idx1 = int(order[0])
        min_sep = self.N // 8
        idx2 = None
        for j in order[1:]:
            sep = abs(int(j) - idx1)
            if min_sep < sep < self.N - min_sep:
                idx2 = int(j)
                break
        if idx2 is None:
            idx2 = int(order[1])
        if self.s_vals[idx1] > self.s_vals[idx2]:
            idx1, idx2 = idx2, idx1
        self.dist_at_crossing_1 = float(self.s_vals[idx1])
        self.dist_at_crossing_2 = float(self.s_vals[idx2])
        self.dist_at_pi = self.dist_at_crossing_1

        self.Y = -self.Y
        dxV = np.roll(self.X, -1) - np.roll(self.X, 1)
        dyV = np.roll(self.Y, -1) - np.roll(self.Y, 1)
        self.hdg = np.arctan2(dyV, dxV)

    def _build_raw_centerline(self):
        r, nA, nL = self._R_ARC, self._N_ARC, self._N_LINE
        X, Y = np.array([]), np.array([])
        X, Y = self._arc(X, Y, 4.5, 1.5, r,  np.pi/2,       0,           nA)
        X, Y = self._arc(X, Y, 4.5, 1.5, r,  0,            -np.pi/2,     nA)
        X, Y = self._arc(X, Y, 4.5, 0.5, r,  np.pi/2,       np.pi,       nA)
        X, Y = self._arc(X, Y, 3.5, 0.5, r,  0,            -np.pi/2,     nA)
        X, Y = self._line(X, Y, 3.5, 0.0,  1.5, 0.0,        nL)
        X, Y = self._arc(X, Y, 1.5, 0.5, r,  3*np.pi/2,    np.pi/2,     nA)
        X, Y = self._arc(X, Y, 1.5, 1.5, r,  3*np.pi/2,    2*np.pi,     nA)
        X, Y = self._line(X, Y, 2.0, 1.5,  2.0, 4.5,        nL)
        X, Y = self._arc(X, Y, 1.5, 4.5, r,  0,             np.pi/2,     nA)
        X, Y = self._line(X, Y, 1.5, 5.0,  0.5, 5.0,        nL)
        X, Y = self._arc(X, Y, 0.5, 4.5, r,  np.pi/2,       3*np.pi/2,  nA)
        X, Y = self._arc(X, Y, 0.5, 3.5, r,  np.pi/2,      -np.pi/2,    nA)
        X, Y = self._arc(X, Y, 0.5, 2.5, r,  np.pi/2,       3*np.pi/2,  nA)
        X, Y = self._line(X, Y, 0.5, 2.0,  4.5, 2.0,        nL)
        if np.hypot(X[-1] - X[0], Y[-1] - Y[0]) < 1e-10:
            X, Y = X[:-1], Y[:-1]
        return X, Y

    @staticmethod
    def _arc(X, Y, xc, yc, r, t1, t2, n):
        th = np.linspace(t1, t2, n)
        return DistanceMapper._append(X, Y, xc + r * np.cos(th), yc + r * np.sin(th))

    @staticmethod
    def _line(X, Y, x1, y1, x2, y2, n):
        return DistanceMapper._append(X, Y, np.linspace(x1, x2, n), np.linspace(y1, y2, n))

    @staticmethod
    def _append(X, Y, xN, yN):
        if len(X) == 0:
            return xN.copy(), yN.copy()
        if np.hypot(X[-1] - xN[0], Y[-1] - yN[0]) < 1e-10:
            xN, yN = xN[1:], yN[1:]
        return np.concatenate([X, xN]), np.concatenate([Y, yN])

    def get_idx_from_s(self, s):
        s = float(s) % self.total_length
        return int(np.searchsorted(self.s_vals[:-1], s, side='right') - 1) % self.N

    def get_s_from_idx(self, idx):
        return float(self.s_vals[int(idx) % self.N])

    def get_xy_from_idx(self, idx):
        i = int(idx) % self.N
        return float(self.X[i]), float(self.Y[i])

    def get_xy_from_s(self, s):
        return self.get_xy_from_idx(self.get_idx_from_s(s))

    def get_nearest(self, x, y, hint_idx=None, window=60):
        if hint_idx is not None:
            idxs = np.arange(hint_idx - window, hint_idx + window + 1) % self.N
        else:
            idxs = np.arange(self.N)
        d2 = (self.X[idxs] - x)**2 + (self.Y[idxs] - y)**2
        best = int(idxs[np.argmin(d2)])
        return float(self.s_vals[best]), best


class TrackTracker:
    def __init__(self, geom, idx_init):
        self.geom = geom
        self.idx = int(idx_init)
        self._prev_loc = None

    def update(self, location):
        if self._prev_loc is not None:
            moved = math.sqrt(
                (location.x - self._prev_loc.x)**2
                + (location.y - self._prev_loc.y)**2
            )
            ds_avg = self.geom.total_length / self.geom.N
            window = max(20, int(3.0 * moved / ds_avg))
        else:
            window = 100
        self._prev_loc = location
        _, self.idx = self.geom.get_nearest(location.x, location.y,
                                            hint_idx=self.idx, window=window)
        return self.idx

    def get_s(self):
        return self.geom.get_s_from_idx(self.idx)

    def get_lookahead(self, lookahead_m):
        s_ahead = (self.get_s() + lookahead_m) % self.geom.total_length
        idx_ahead = self.geom.get_idx_from_s(s_ahead)
        x, y = self.geom.get_xy_from_idx(idx_ahead)
        return carla.Location(x=x, y=y, z=0.0), idx_ahead


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
        self.integral = np.clip(self.integral + error * dt,
                                -self.integral_limit, self.integral_limit)
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
        self.integral = np.clip(self.integral + error * dt,
                                -self.integral_limit, self.integral_limit)
        if derivative_input is None:
            derivative = (error - self.prev_error) / dt if dt > 0 else 0
        else:
            derivative = derivative_input
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.prev_error = 0
        self.integral = 0

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "simulation_results", "BarrierControlResults"
)
FIXED_RUN_COUNT = 1
RUN_SEED_BASE = 1000
KMH_PER_MS = 3.6

_BLUEPRINT_POOL = [
    'vehicle.tesla.model3',
    'vehicle.audi.etron',
    'vehicle.toyota.prius',
    'vehicle.bmw.grandtourer',
]

_BLUEPRINT_COLOR = {
    'vehicle.tesla.model3':    '255,0,0',
    'vehicle.audi.etron':      '0,0,0',
    'vehicle.toyota.prius':    '0,200,0',
    'vehicle.bmw.grandtourer': '0,0,255',
}

def build_vehicle_blueprint_sequence(num_vehicles, rng):
    base = _BLUEPRINT_POOL * (num_vehicles // len(_BLUEPRINT_POOL) + 1)
    blueprint_ids = base[:num_vehicles]
    rng.shuffle(blueprint_ids)
    return blueprint_ids

def get_run_seed(run_index):
    return RUN_SEED_BASE + run_index

def create_state_csv(agent_ids, output_dir=RESULTS_DIR, filename_prefix="state"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    header = ["time_s"]
    for vid in agent_ids:
        header.append(f"pos_{vid}_m")
    for vid in agent_ids:
        header.append(f"headway_{vid}_m")
    for vid in agent_ids:
        header.append(f"speed_{vid}_ms")
    for vid in agent_ids:
        header.append(f"speed_{vid}_kmh")
    header += [
        "inter_loop1_vid", "inter_loop1_dist_m",
        "inter_loop2_vid", "inter_loop2_dist_m",
        "d_jk_m",
    ]
    writer.writerow(header)
    return csv_file, writer, filename


# 3. MAIN SIMULATION

def run_simulation(args, run_index=1, run_output_dir=RESULTS_DIR):
    N = args.number_of_vehicles
    max_duration_s = args.duration_seconds
    run_seed = get_run_seed(run_index)
    rng = random.Random(run_seed)
    random.seed(run_seed)
    np.random.seed(run_seed)

    geom = DistanceMapper()
    TOTAL_TRACK_LENGTH = geom.total_length
    TARGET_GAP_METERS = (TOTAL_TRACK_LENGTH / N) if N > 0 else 0.0

    DT = 0.05
    WARMUP_STEPS = 100
    RAMP_STEPS = 250
    TARGET_SPEED_KMH = 30.0
    COMFORTABLE_MAX_ACCEL_MS2 = 2.5

    GAIN_HEADWAY_P = 3.50
    GAIN_HEADWAY_I = 0.00
    GAIN_HEADWAY_D = 1.00
    GAIN_INTEGRAL_LIMIT = 0.0
    MAX_HEADWAY_KMH = 10.0

    GAIN_BARRIER = 25.0
    BARRIER_ACTIVATION_DIST = 10.0
    APPROACH_WINDOW_M = 50.0
    EPSILON_DIFF = 1.0
    MAX_BARRIER_KMH = 7.0

    LOOKAHEAD_M = 5.0
    GAP_EMA_ALPHA = 0.9

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(20.0)

    vehicles_list = []
    state_csv_file = state_csv_writer = state_csv_filename = None
    original_settings = None
    summary = None

    try:
        world = client.get_world()
        carla_map = world.get_map()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        vehicle_blueprint_ids = build_vehicle_blueprint_sequence(N, rng)

        _all_wps = carla_map.generate_waypoints(2.0)
        safe_waypoints = [w for w in _all_wps if not w.is_junction]
        safe_wp_xy = np.array([[w.transform.location.x, w.transform.location.y]
                                for w in safe_waypoints])
        safe_wp_fwd = np.array([[w.transform.get_forward_vector().x,
                                  w.transform.get_forward_vector().y]
                                 for w in safe_waypoints])

        agent_ids = []
        agent_pids = {}
        agent_gap_pids = {}
        agent_trackers = {}
        vehicle_blueprints = {}
        vehicles_by_id = {}

        print(f"\n--- Run {run_index}: Spawning {N} vehicles ---")
        print(f"Track length: {TOTAL_TRACK_LENGTH:.1f} m  |  Target gap: {TARGET_GAP_METERS:.1f} m")
        print(f"Crossing 1 at s={geom.dist_at_crossing_1:.1f} m  |  "
              f"Crossing 2 at s={geom.dist_at_crossing_2:.1f} m")
        print(f"[coord-check] DM  X:[{geom.X.min():.1f},{geom.X.max():.1f}] "
              f"Y:[{geom.Y.min():.1f},{geom.Y.max():.1f}]")
        print(f"[coord-check] MAP X:[{safe_wp_xy[:,0].min():.1f},{safe_wp_xy[:,0].max():.1f}] "
              f"Y:[{safe_wp_xy[:,1].min():.1f},{safe_wp_xy[:,1].max():.1f}]")

        for i in range(N):
            noise = rng.uniform(-0.3, 0.3) * TARGET_GAP_METERS
            s_target = (i * TARGET_GAP_METERS + noise) % TOTAL_TRACK_LENGTH
            idx_spawn = geom.get_idx_from_s(s_target)
            start_x, start_y = geom.get_xy_from_idx(idx_spawn)
            hdg_spawn = float(geom.hdg[idx_spawn])
            tang = np.array([math.cos(hdg_spawn), math.sin(hdg_spawn)])

            dists = np.linalg.norm(safe_wp_xy - np.array([start_x, start_y]), axis=1)
            correct_loop = (safe_wp_fwd @ tang) > 0.8
            if not correct_loop.any():
                correct_loop = np.ones(len(safe_waypoints), dtype=bool)
            candidates = np.where(correct_loop)[0][np.argsort(dists[correct_loop])]

            bp_id = vehicle_blueprint_ids[i]
            bp_veh = bp_lib.find(bp_id)
            if bp_veh.has_attribute('color') and bp_id in _BLUEPRINT_COLOR:
                bp_veh.set_attribute('color', _BLUEPRINT_COLOR[bp_id])

            trans = safe_waypoints[candidates[0]].transform
            trans.location.z += 0.5
            veh = world.try_spawn_actor(bp_veh, trans)
            if veh is None:
                print(f"  WARNING: could not spawn vehicle {i} ({bp_id}) — skipping")
                continue

            veh.set_target_velocity(carla.Vector3D(0, 0, 0))
            print(f"  spawned vehicle {veh.id} as {bp_id}")

            agent_ids.append(veh.id)
            agent_trackers[veh.id] = TrackTracker(geom, idx_spawn)
            agent_pids[veh.id] = PIDController(Kp=0.5, Ki=0.25, Kd=0.02, integral_limit=10.0)
            agent_gap_pids[veh.id] = HeadwayPIDController(
                Kp=GAIN_HEADWAY_P, Ki=GAIN_HEADWAY_I, Kd=GAIN_HEADWAY_D,
                integral_limit=GAIN_INTEGRAL_LIMIT)
            vehicle_blueprints[veh.id] = bp_id
            vehicles_list.append(veh)
            vehicles_by_id[veh.id] = veh

        state_csv_file, state_csv_writer, state_csv_filename = create_state_csv(
            agent_ids,
            output_dir=run_output_dir,
            filename_prefix=f"state_N{N}"
        )
        print(f"Logging state to {state_csv_filename}")
        print("Simulation running...")

        gap_ema = {vid: None for vid in agent_ids}
        prev_target_speed = {}
        prev_steer = {}

        step_count = 0
        while True:
            world.tick()
            step_count += 1
            elapsed_time = step_count * DT

            # --- 1. State Estimation ---
            current_s_values = {}
            velocities = {}
            for veh in vehicles_list:
                loc = veh.get_location()
                if step_count > WARMUP_STEPS:
                    agent_trackers[veh.id].update(loc)
                current_s_values[veh.id] = agent_trackers[veh.id].get_s()
                v = veh.get_velocity()
                velocities[veh.id] = KMH_PER_MS * math.sqrt(v.x**2 + v.y**2)

            sorted_agents = sorted(agent_ids, key=lambda vid: current_s_values[vid])

            # --- 2. Barrier Control (Intersection Logic) ---
            barrier_controls = {vid: 0.0 for vid in agent_ids}
            conflict_pair = None
            inter_j = None
            inter_k = None
            dist_to_inter_j = None
            dist_to_inter_k = None
            d_jk_meters = None

            if step_count > WARMUP_STEPS:
                c1_s = geom.dist_at_crossing_1
                c2_s = geom.dist_at_crossing_2
                cands_1 = [vid for vid in agent_ids
                           if (c1_s - APPROACH_WINDOW_M) < current_s_values[vid] < c1_s]
                cands_2 = [vid for vid in agent_ids
                           if (c2_s - APPROACH_WINDOW_M) < current_s_values[vid] < c2_s]

                if cands_1 and cands_2:
                    j = max(cands_1, key=lambda vid: current_s_values[vid])
                    k = max(cands_2, key=lambda vid: current_s_values[vid])
                    conflict_pair = (j, k)
                    d_j = c1_s - current_s_values[j]
                    d_k = c2_s - current_s_values[k]
                    diff = d_k - d_j
                    inter_j = j
                    inter_k = k
                    dist_to_inter_j = d_j
                    dist_to_inter_k = d_k
                    d_jk_meters = diff
                    if abs(diff) < BARRIER_ACTIVATION_DIST:
                        safe_diff = (EPSILON_DIFF if diff >= 0 else -EPSILON_DIFF) \
                                    if abs(diff) < EPSILON_DIFF else diff
                        gradient = -1.0 / safe_diff
                        barrier_controls[k] = gradient
                        barrier_controls[j] = -gradient

            dash_lines = []
            u_headway_log = {}
            u_barrier_log = {}
            gap_meters_log = {}
            target_speed_log = {}

            if step_count <= WARMUP_STEPS:
                current_nominal_speed = 0.0
            elif step_count <= WARMUP_STEPS + RAMP_STEPS:
                current_nominal_speed = TARGET_SPEED_KMH * (step_count - WARMUP_STEPS) / RAMP_STEPS
            else:
                current_nominal_speed = TARGET_SPEED_KMH

            for i, vid in enumerate(sorted_agents):
                veh = vehicles_by_id[vid]
                curr_speed = velocities[vid]

                # --- 3. Headway Sensing ---
                u_headway = 0.0
                gap_meters = TOTAL_TRACK_LENGTH
                leader_id = None
                if N > 1:
                    leader_id = sorted_agents[(i + 1) % N]
                    raw_gap = (current_s_values[leader_id] - current_s_values[vid]) % TOTAL_TRACK_LENGTH
                    if gap_ema[vid] is None:
                        gap_ema[vid] = raw_gap
                    else:
                        gap_ema[vid] = GAP_EMA_ALPHA * gap_ema[vid] + (1.0 - GAP_EMA_ALPHA) * raw_gap
                    gap_meters = gap_ema[vid]
                gap_meters_log[vid] = gap_meters

                # --- 4. Combine Controls ---
                u_barrier = float(np.clip(GAIN_BARRIER * barrier_controls[vid],
                                          -MAX_BARRIER_KMH, MAX_BARRIER_KMH))
                u_barrier_log[vid] = u_barrier

                if step_count <= WARMUP_STEPS:
                    target_speed = 0.0
                    prev_target_speed[vid] = 0.0
                    agent_pids[vid].reset()
                    agent_gap_pids[vid].reset()
                elif step_count <= WARMUP_STEPS + RAMP_STEPS:
                    target_speed = max(0.0, current_nominal_speed + u_barrier)
                    prev_target_speed[vid] = target_speed
                else:
                    if N > 1 and leader_id is not None:
                        u_headway = float(np.clip(
                            agent_gap_pids[vid].run(gap_meters, TARGET_GAP_METERS, DT),
                            -MAX_HEADWAY_KMH, MAX_HEADWAY_KMH))
                    u_headway_log[vid] = u_headway
                    target_speed = max(0.0, current_nominal_speed + u_headway + u_barrier)
                    max_delta = COMFORTABLE_MAX_ACCEL_MS2 * DT * KMH_PER_MS
                    prev = prev_target_speed.get(vid, target_speed)
                    target_speed = float(np.clip(target_speed,
                                                 prev - max_delta, prev + max_delta))
                    prev_target_speed[vid] = target_speed
                target_speed_log[vid] = target_speed

                # --- 5. Lateral Control ---
                curr_loc = veh.get_location()
                veh_fwd = veh.get_transform().get_forward_vector()
                tracker = agent_trackers[vid]

                math_target, idx_ahead = tracker.get_lookahead(LOOKAHEAD_M)
                wp_target = carla_map.get_waypoint(math_target, project_to_road=True,
                                                   lane_type=carla.LaneType.Driving)
                wp_curr = carla_map.get_waypoint(curr_loc, project_to_road=True,
                                                 lane_type=carla.LaneType.Driving)

                if wp_target is not None:
                    wp_fwd = wp_target.transform.get_forward_vector()
                    if veh_fwd.x * wp_fwd.x + veh_fwd.y * wp_fwd.y < 0.7:
                        wp_target = None

                at_junction = ((wp_curr is not None and wp_curr.is_junction) or
                               (wp_target is not None and wp_target.is_junction))
                if at_junction or wp_target is None:
                    hdg_a = float(geom.hdg[idx_ahead])
                    tang_x, tang_y = math.cos(hdg_a), math.sin(hdg_a)
                    steer_cmd = float(np.clip(
                        veh_fwd.x * tang_y - veh_fwd.y * tang_x, -1.0, 1.0))
                else:
                    st = wp_target.transform.location
                    vec_to = np.array([st.x - curr_loc.x, st.y - curr_loc.y])
                    norm = np.linalg.norm(vec_to)
                    if norm > 0:
                        vec_to /= norm
                    steer_cmd = float(np.clip(
                        veh_fwd.x * vec_to[1] - veh_fwd.y * vec_to[0], -1.0, 1.0))
                prev_steer[vid] = steer_cmd

                # --- 6. Actuation ---
                control = carla.VehicleControl()
                control.steer = steer_cmd
                if step_count <= WARMUP_STEPS:
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.hand_brake = True
                else:
                    tb = agent_pids[vid].run(target_speed, curr_speed, DT)
                    control.hand_brake = False
                    if tb >= 0:
                        control.throttle = min(tb, 1.0)
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0
                        control.brake = min(abs(tb), 1.0)
                veh.apply_control(control)

                dash_lines.append(
                    f"{vid} - {u_headway_log.get(vid, 0.0):6.2f} - {u_barrier:6.2f} - {curr_speed:5.1f}"
                )

            if elapsed_time >= max_duration_s:
                avg_speed = sum(velocities.values()) / len(velocities) if velocities else 0.0
                summary = {
                    "run_index": run_index,
                    "run_directory": run_output_dir,
                    "vehicles": N,
                    "duration_s": max_duration_s,
                    "final_time_seconds": elapsed_time,
                    "avg_speed_kmh": avg_speed,
                }
                print(f"Run {run_index} completed at {elapsed_time:.2f} s")
                break

            # State CSV (every tick, flush once per second)
            if state_csv_writer is not None and current_s_values:
                row = [f"{elapsed_time:.4f}"]
                for vid in agent_ids:
                    row.append(f"{current_s_values.get(vid, 0.0):.4f}")
                sorted_rank = {vid: idx for idx, vid in enumerate(sorted_agents)}
                for vid in agent_ids:
                    if N > 1:
                        rank = sorted_rank.get(vid, 0)
                        ldr = sorted_agents[(rank + 1) % N]
                        raw_hw = (current_s_values.get(ldr, 0.0) - current_s_values.get(vid, 0.0)) % TOTAL_TRACK_LENGTH
                    else:
                        raw_hw = TOTAL_TRACK_LENGTH
                    row.append(f"{raw_hw:.4f}")
                for vid in agent_ids:
                    row.append(f"{velocities.get(vid, 0.0) / KMH_PER_MS:.4f}")
                for vid in agent_ids:
                    row.append(f"{velocities.get(vid, 0.0):.4f}")
                row += [
                    "" if inter_j is None else str(inter_j),
                    "" if dist_to_inter_j is None else f"{dist_to_inter_j:.4f}",
                    "" if inter_k is None else str(inter_k),
                    "" if dist_to_inter_k is None else f"{dist_to_inter_k:.4f}",
                    "" if d_jk_meters is None else f"{d_jk_meters:.4f}",
                ]
                state_csv_writer.writerow(row)
                if step_count % int(1 / DT) == 0:
                    state_csv_file.flush()

            # Dashboard
            if step_count % 5 == 0:
                print("\033[H\033[J")
                if step_count <= WARMUP_STEPS:
                    status = "SETTLING..."
                elif step_count <= WARMUP_STEPS + RAMP_STEPS:
                    status = f"RAMPING ({current_nominal_speed:.1f} km/h)"
                else:
                    status = "ACTIVE"
                print(f"--- RUN {run_index} | step {step_count} [{status}] | t={elapsed_time:.1f}s ---")
                print(f"{'vehicle_id':<12} {'u_headway':>10} {'u_barrier':>10} {'speed(kmh)':>10}")
                for line in dash_lines:
                    print(line)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if state_csv_file is not None:
            state_csv_file.close()
            print(f"State log → {state_csv_filename}")

        for v in vehicles_list:
            try:
                v.destroy()
            except Exception:
                pass
        if original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass

    return summary

def main():
    parser = argparse.ArgumentParser(description='Random Map Geodesic Barrier Simulation')
    parser.add_argument('--number-of-vehicles', '-n', type=int, default=4,
                        help='Number of vehicles to spawn (default: 4)')
    parser.add_argument('--duration-seconds', '-d', type=float, default=1800.0,
                        help='Simulation duration in seconds (default: 1800)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_simulation(args, run_index=1, run_output_dir=RESULTS_DIR)


if __name__ == '__main__':
    main()

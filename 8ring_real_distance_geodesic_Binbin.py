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
    Handles the mapping between parametric t (radians) and physical distance S (meters).
    Uses REVERSED math (-R*sin) to match natural car orientation.
    """
    def __init__(self, R, resolution=20000):
        self.R = R
        self.t_vals = np.linspace(0, 2 * np.pi, resolution)

        x_vals = -self.R * np.sin(self.t_vals)
        y_vals = -self.R * np.sin(2 * self.t_vals)

        dists = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
        self.s_vals = np.concatenate(([0], np.cumsum(dists)))
        self.total_length = self.s_vals[-1]

        self.hdg = np.arctan2(
            -2.0 * np.cos(2.0 * self.t_vals),
            -np.cos(self.t_vals)
        )

        idx_pi = np.argmin(np.abs(self.t_vals - np.pi))
        self.dist_at_pi = self.s_vals[idx_pi]

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


class Figure8Tracker:
    def __init__(self, R, t_init, geom):
        self.R = R
        self.t = t_init
        self.geom = geom
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

        dx = location.x - (-self.R * np.sin(t_wrapped))
        dy = location.y - (-self.R * np.sin(2 * t_wrapped))

        d2 = dx**2 + dy**2
        best_idx = np.argmin(d2)
        self.t = t_search[best_idx]
        return self.t % (2 * np.pi)

    def get_math_location(self, t_val):
        tx = -self.R * np.sin(t_val)
        ty = -self.R * np.sin(2 * t_val)
        return carla.Location(x=tx, y=ty, z=0.0)

    def get_lookahead(self, lookahead_m):
        s_current = self.geom.get_s_from_t(self.t % (2 * np.pi))
        s_ahead = (s_current + lookahead_m) % self.geom.total_length
        t_ahead = self.geom.get_t_from_s(s_ahead)
        idx_ahead = int(np.argmin(np.abs(self.geom.t_vals - t_ahead)))
        return self.get_math_location(t_ahead), idx_ahead

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "simulation_results", "BarrierControlResults"
)

KMH_PER_MS = 3.6

_BLUEPRINT_POOL = [
    'vehicle.tesla.model3',
    'vehicle.audi.etron',
    'vehicle.toyota.prius',
    'vehicle.bmw.grandtourer',
]


def build_vehicle_blueprint_sequence(num_vehicles, rng):
    base = num_vehicles // 4
    remainder = num_vehicles % 4
    counts = [base + (1 if i < remainder else 0) for i in range(4)]
    blueprint_ids = [bp for bp, n in zip(_BLUEPRINT_POOL, counts) for _ in range(n)]
    rng.shuffle(blueprint_ids)
    return blueprint_ids



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
    R = args.radius
    max_duration_s = args.duration_seconds
    rng = random

    geom = DistanceMapper(R)
    TOTAL_TRACK_LENGTH = geom.total_length
    TARGET_GAP_METERS = (TOTAL_TRACK_LENGTH / N) if N > 0 else 0

    DT = 0.05
    WARMUP_STEPS = 100
    RAMP_STEPS = 250
    TARGET_SPEED_KMH = 30.0
    COMFORTABLE_MAX_ACCEL_MS2 = 2.5

    GAIN_HEADWAY_P = 3.50
    GAIN_HEADWAY_I = 0.00
    GAIN_HEADWAY_D = 1.00
    GAIN_INTEGRAL_LIMIT = 3.0
    MAX_HEADWAY_KMH = 10.0

    GAIN_BARRIER = 25.0
    BARRIER_ACTIVATION_DIST = min(TARGET_GAP_METERS / 2.0, 10)
    APPROACH_WINDOW_M = 50
    EPSILON_DIFF = 1.0
    MAX_BARRIER_KMH = 7.0

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
        agent_gap_pids = {}
        agent_pids = {}
        agent_trackers = {}
        vehicle_blueprints = {}
        vehicles_by_id = {}

        print(f"\n--- Run {run_index}: Spawning {N} EVs ---")
        print(f"Total Track Length: {TOTAL_TRACK_LENGTH:.1f} m")
        print(f"Target Gap: {TARGET_GAP_METERS:.1f} m")

        for i in range(N):
            noise = rng.uniform(-0.35, 0.35) * TARGET_GAP_METERS
            s_target = (i * TARGET_GAP_METERS + noise) % TOTAL_TRACK_LENGTH
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

            _BLUEPRINT_COLORS = {
                'audi':  '0,0,0',
                'tesla': '255,0,0',
                'prius': '0,255,0',
                'bmw':   '0,0,255',
            }
            bp_id = vehicle_blueprint_ids[i]
            bp_veh = bp_lib.find(bp_id)
            if bp_veh.has_attribute('color'):
                color = next((c for k, c in _BLUEPRINT_COLORS.items() if k in bp_id), '255,255,255')
                bp_veh.set_attribute('color', color)

            trans = safe_waypoints[candidates[0]].transform
            trans.location.z += 0.5
            veh = world.try_spawn_actor(bp_veh, trans)

            if veh is None:
                print(f"  WARNING: could not spawn vehicle {i} ({bp_id}) — skipping")
                continue

            veh.set_target_velocity(carla.Vector3D(0, 0, 0))
            print(f"spawned vehicle {veh.id} as {bp_id}")

            agent_ids.append(veh.id)
            agent_trackers[veh.id] = Figure8Tracker(R, t_spawn, geom)
            agent_pids[veh.id] = PIDController(Kp=0.5, Ki=0.25, Kd=0.02, integral_limit=10.0)
            agent_gap_pids[veh.id] = HeadwayPIDController(
                Kp=GAIN_HEADWAY_P, Ki=GAIN_HEADWAY_I, Kd=GAIN_HEADWAY_D,
                integral_limit=GAIN_INTEGRAL_LIMIT)
            vehicle_blueprints[veh.id] = bp_id
            vehicles_list.append(veh)
            vehicles_by_id[veh.id] = veh

        N = len(agent_ids)
        if N > 0:
            TARGET_GAP_METERS = TOTAL_TRACK_LENGTH / N

        radius_label = str(R).replace('.', 'p')
        state_csv_file, state_csv_writer, state_csv_filename = create_state_csv(
            agent_ids,
            output_dir=run_output_dir,
            filename_prefix=f"state_N{N}_R{radius_label}"
        )
        print(f"Logging state to {state_csv_filename}")
        print("Simulation Running...")

        gap_ema = {vid: None for vid in agent_ids}
        prev_target_speed = {}

        step_count = 0
        while True:
            world.tick()
            step_count += 1
            elapsed_time = step_count * DT
            if elapsed_time >= max_duration_s:
                print(f"\nReached duration limit ({max_duration_s}s). Stopping.")
                break

            # --- 1. State Estimation (t -> S) ---
            current_s_values = {}
            velocities = {}
            for veh in vehicles_list:
                loc = veh.get_location()
                if step_count > WARMUP_STEPS:
                    t_new = agent_trackers[veh.id].update_t(loc)
                else:
                    t_new = agent_trackers[veh.id].t % (2 * np.pi)
                current_s_values[veh.id] = geom.get_s_from_t(t_new)
                v = veh.get_velocity()
                velocities[veh.id] = KMH_PER_MS * math.sqrt(v.x**2 + v.y**2)

            sorted_agents = sorted(agent_ids, key=lambda vid: current_s_values[vid])

            # --- 2. Barrier Control (Intersection Logic) ---
            conflict_pair = None
            barrier_controls = {vid: 0.0 for vid in agent_ids}
            inter_j = None
            inter_k = None
            dist_to_inter_j = None
            dist_to_inter_k = None
            d_jk_meters = None

            if step_count > WARMUP_STEPS:
                candidates_1 = [
                    vid for vid in agent_ids
                    if (geom.dist_at_pi - APPROACH_WINDOW_M) < current_s_values[vid] < geom.dist_at_pi
                ]
                candidates_2 = [
                    vid for vid in agent_ids
                    if (geom.total_length - APPROACH_WINDOW_M) < current_s_values[vid] < geom.total_length
                ]

                if candidates_1 and candidates_2:
                    c1 = max(candidates_1, key=lambda vid: current_s_values[vid])
                    c2 = max(candidates_2, key=lambda vid: current_s_values[vid])
                    conflict_pair = (c1, c2)

                    dist_to_cross_1 = geom.dist_at_pi - current_s_values[c1]
                    dist_to_cross_2 = geom.total_length - current_s_values[c2]
                    diff_meters = dist_to_cross_2 - dist_to_cross_1
                    inter_j = c1
                    inter_k = c2
                    dist_to_inter_j = dist_to_cross_1
                    dist_to_inter_k = dist_to_cross_2
                    d_jk_meters = diff_meters

                    if abs(diff_meters) < BARRIER_ACTIVATION_DIST:
                        if abs(diff_meters) < EPSILON_DIFF:
                            safe_diff = EPSILON_DIFF * (1.0 if diff_meters >= 0 else -1.0)
                        else:
                            safe_diff = diff_meters
                        gradient = -1.0 / safe_diff
                        barrier_controls[c2] = gradient
                        barrier_controls[c1] = -gradient

            dash_lines = []
            u_headway_log = {}
            u_barrier_log = {}
            gap_meters_log = {}
            target_speed_log = {}

            if step_count <= WARMUP_STEPS:
                current_nominal_speed = 0.0
            elif step_count <= (WARMUP_STEPS + RAMP_STEPS):
                progress = (step_count - WARMUP_STEPS) / RAMP_STEPS
                current_nominal_speed = TARGET_SPEED_KMH * progress
            else:
                current_nominal_speed = TARGET_SPEED_KMH

            for i, vid in enumerate(sorted_agents):
                veh = vehicles_by_id[vid]

                # --- 3. Headway Sensing ---
                u_headway = 0.0
                gap_meters = TOTAL_TRACK_LENGTH
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

                gap_meters_log[vid] = gap_meters

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
                    if target_speed < 0:
                        target_speed = 0
                    prev_target_speed[vid] = target_speed
                else:
                    if N > 1 and leader_id is not None:
                        u_headway = agent_gap_pids[vid].run(gap_meters, TARGET_GAP_METERS, DT)
                        if MAX_HEADWAY_KMH > 0.0:
                            u_headway = float(np.clip(u_headway, -MAX_HEADWAY_KMH, MAX_HEADWAY_KMH))
                    u_headway_log[vid] = u_headway
                    target_speed = current_nominal_speed + u_headway + u_barrier
                    if target_speed < 0:
                        target_speed = 0
                    max_delta_kmh = COMFORTABLE_MAX_ACCEL_MS2 * DT * KMH_PER_MS
                    prev = prev_target_speed.get(vid, target_speed)
                    target_speed = float(np.clip(target_speed, prev - max_delta_kmh, prev + max_delta_kmh))
                    prev_target_speed[vid] = target_speed
                target_speed_log[vid] = target_speed

               # 5. Lateral Control 
                prev_steer = {}
                LOOKAHEAD_M = 5.0
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
                curr_speed = velocities[vid]
                control = carla.VehicleControl()
                control.steer = float(steer_cmd)

                if step_count <= WARMUP_STEPS:
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.hand_brake = True
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

                dash_lines.append(
                    f"{vid} - {u_headway_log.get(vid, 0.0):6.2f} - {u_barrier:6.2f} - {curr_speed:5.1f}"
                )

            # State CSV 
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
                    status = f"RAMPING UP ({current_nominal_speed:.1f} km/h)"
                else:
                    status = "ACTIVE"
                print(f"--- RUN {run_index} | STEP {step_count} [{status}] | t={elapsed_time:.1f}s ---")
                print(f"{'vehicle_id':<12} {'u_headway':>10} {'u_barrier':>10} {'speed(kmh)':>10}")
                for line in dash_lines:
                    print(line)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if state_csv_file is not None:
            state_csv_file.close()
            print(f"State log saved to {state_csv_filename}")
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--number-of-vehicles', default=21, type=int)
    argparser.add_argument('--radius', default=66.2, type=float)
    argparser.add_argument('--duration-seconds', default=1800.0, type=float)
    argparser.add_argument('--output-dir', default=RESULTS_DIR)
    args = argparser.parse_args()

    run_simulation(args)


if __name__ == '__main__':
    main()


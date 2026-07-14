#!/usr/bin/env python
# Copyright (c) 2025 Computer Vision Center (CVC)
# Figure-8 Simulation: Distance-Based Control with Reverse Flow

import glob
import os
import sys
import csv
import random
import carla
import argparse
import math
import numpy as np
import networkx as nx
from datetime import datetime


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
# ==========================================
# 1. MONKEY PATCH (Graph Build)
# ==========================================
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

# ==========================================
# 2. HELPER CLASSES
# ==========================================

class DistanceMapper:
    """
    Handles the mapping between parametric t (radians) and physical distance S (meters).
    Now uses REVERSED math (-R*sin) to match natural car orientation.
    """
    def __init__(self, R, resolution=4000):
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
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.integral_limit = integral_limit
        self.prev_error = 0; self.integral = 0

    def run(self, target_speed, current_speed, dt):
        error = target_speed - current_speed
        self.integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.prev_error = 0; self.integral = 0

class Figure8Tracker:
    def __init__(self, R, t_init):
        self.R = R; self.t = t_init 
    
    def update_t(self, location):
        window = 0.5
        t_search = np.linspace(self.t - window, self.t + window, 1000) % (2 * np.pi)
        
        # --- REVERSE DIRECTION MATH ---
        dx = location.x - (-self.R * np.sin(t_search))
        dy = location.y - (-self.R * np.sin(2 * t_search))
        
        d2 = dx**2 + dy**2
        best_idx = np.argmin(d2)
        self.t = t_search[best_idx]
        return self.t

    def get_math_location(self, t_val):
        # --- REVERSE DIRECTION MATH ---
        tx = -self.R * np.sin(t_val)
        ty = -self.R * np.sin(2 * t_val)
        return carla.Location(x=tx, y=ty, z=0.0)

radar_data_buffer = {}
def _radar_callback(sensor_data, vehicle_id):
    min_dist = 999.0
    for detect in sensor_data:
        # Filter for objects directly in front
        if -0.1 < detect.azimuth < 0.1: 
            if detect.altitude > -0.05:
                if detect.depth < min_dist: min_dist = detect.depth
    radar_data_buffer[vehicle_id] = min_dist

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def create_speed_csv(output_dir=RESULTS_DIR, filename_prefix="vehicle_speeds"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
    csv_file = open(filename, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "time_seconds",
        "step",
        "vehicle_id",
        "speed_kmh",
        "target_speed_kmh",
        "track_position_m",
        "actual_track_position_m",
        "gap_meters",
        "headway_control_kmh",
        "barrier_control_kmh",
        "lateral_error_m",
    ])
    return csv_file, writer, filename

# ==========================================
# 3. MAIN SIMULATION
# ==========================================

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--number-of-vehicles', default=4, type=int)
    argparser.add_argument('--radius', default=250.0, type=float)
    argparser.add_argument(
        '--actual-track-length',
        default=None,
        type=float,
        help='Loop length used for odometry-based gaps. Defaults to the reference figure-8 length.'
    )
    args = argparser.parse_args()

    N = args.number_of_vehicles
    R = args.radius
    
    # Initialize Geometry Helper
    geom = DistanceMapper(R)
    TOTAL_TRACK_LENGTH = geom.total_length
    
    # Desired distance gap in meters
    TARGET_GAP_METERS = TOTAL_TRACK_LENGTH / N if N > 0 else 0
    
    # Tuning Parameters
    GAIN_HEADWAY = 0 
    GAIN_BARRIER = 0  # High gain for barrier [cite: 2]
    TARGET_SPEED_KMH = 30.0  
    DT = 0.05
    WARMUP_STEPS = 100       
    RAMP_STEPS = 250        
    
    # Barrier Logic Thresholds
    BARRIER_ACTIVATION_DIST = 10    # TARGET_GAP_METERS / 2.0
    APPROACH_WINDOW_M = 50.0 # [Request: closer to intersection point in 10 meters]
    EPSILON_DIFF = 1.0  
    MAX_BARRIER_KMH = 20.0

    client = carla.Client('127.0.0.1', 2000); client.set_timeout(10.0)
    
    vehicles_list = []; sensors_list = []
    speed_csv_file = None
    speed_csv_writer = None
    speed_csv_filename = None
    
    try:
        world = client.get_world()
        carla_map = world.get_map()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        # Global Planner (used for topology patching)
        grp = GlobalRoutePlanner(carla_map, 2.0)
        
        bp_lib = world.get_blueprint_library()
        bp_veh = bp_lib.find('vehicle.tesla.model3')
        bp_veh.set_attribute('color', '255,0,0') 
        bp_radar = bp_lib.find('sensor.other.radar')
        bp_radar.set_attribute('horizontal_fov', '20')
        bp_radar.set_attribute('vertical_fov', '20')
        bp_radar.set_attribute('range', '50')

        agent_trackers = {}; agent_pids = {}; agent_ids = []
        actual_s_values = {}; prev_locations = {}

        print(f"\n--- Spawning {N} Red Teslas (Reverse Flow / Natural Heading) ---")
        print(f"Total Track Length: {TOTAL_TRACK_LENGTH:.1f} m")
        print(f"Actual/Odom Track Length: {TOTAL_TRACK_LENGTH:.1f} m")
        print(f"Target Gap: {TARGET_GAP_METERS:.1f} m")

        # --- SPAWNING LOGIC (METERS + OFFSET) ---
        # Offset is critical for N=4 to avoid spawning directly on the intersection points
        start_dist_offset = TARGET_GAP_METERS / 3.0
        
        for i in range(N):
            # 1. Calculate target distance in the odometry coordinate, then map
            # proportionally onto the reference curve for spawning.
            actual_s_start = (i * TARGET_GAP_METERS + start_dist_offset) % TOTAL_TRACK_LENGTH
            s_target = (actual_s_start / TOTAL_TRACK_LENGTH) * TOTAL_TRACK_LENGTH
            
            # 2. Convert S -> t using our new mapper function
            t_spawn = geom.get_t_from_s(s_target)
            
            # 3. Get Cartesian Coordinates (Reversed Math)
            start_x = -R * np.sin(t_spawn)
            start_y = -R * np.sin(2 * t_spawn)
            
            math_loc = carla.Location(x=start_x, y=start_y, z=2.0)
            wp = carla_map.get_waypoint(math_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            if wp:
                spawn_trans = wp.transform
                spawn_trans.location.z += 0.5 
                
                # [Request: Do not rotate yaw 180]
                # We use the natural road heading. Since math is reversed, cars drive 'forward' naturally.
                
                veh = world.spawn_actor(bp_veh, spawn_trans)
                veh.set_target_velocity(carla.Vector3D(0, 0, 0))
                
                agent_ids.append(veh.id)
                agent_trackers[veh.id] = Figure8Tracker(R, t_spawn)
                agent_pids[veh.id] = PIDController(Kp=0.5, Ki=0.25, Kd=0.02, integral_limit=10.0)
                actual_s_values[veh.id] = actual_s_start
                prev_locations[veh.id] = None
                
                radar = world.spawn_actor(bp_radar, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=veh)
                radar.listen(lambda data, vid=veh.id: _radar_callback(data, vid))
                
                vehicles_list.append(veh)
                sensors_list.append(radar)
                radar_data_buffer[veh.id] = 999.0

        log_vehicle_ids = set(agent_ids[:4])
        radius_label = str(R).replace('.', 'p')
        speed_csv_file, speed_csv_writer, speed_csv_filename = create_speed_csv(
            filename_prefix=f"vehicle_speeds_N{N}_R{radius_label}"
        )
        print(f"Logging vehicle speeds to {speed_csv_filename}")
        print(f"CSV control logging enabled for first {len(log_vehicle_ids)} vehicle(s): {sorted(log_vehicle_ids)}")
        print("Simulation Running...")

        step_count = 0
        while True:
            world.tick()
            step_count += 1
            elapsed_time = step_count * DT
            
            # --- 1. State Estimation (t -> S) ---
            current_s_values = {} 
            current_actual_s_values = {}
            velocities = {}
            
            for veh in vehicles_list:
                loc = veh.get_location()
                # Update tracker (finds t based on reversed math)
                t = agent_trackers[veh.id].update_t(loc)
                s = geom.get_s_from_t(t)

                prev_loc = prev_locations.get(veh.id)
                if prev_loc is not None and step_count > WARMUP_STEPS:
                    actual_step = math.hypot(loc.x - prev_loc.x, loc.y - prev_loc.y)
                    actual_s_values[veh.id] = (actual_s_values[veh.id] + actual_step) % TOTAL_TRACK_LENGTH
                prev_locations[veh.id] = carla.Location(x=loc.x, y=loc.y, z=loc.z)
                
                v = veh.get_velocity()
                speed = 3.6 * math.sqrt(v.x**2 + v.y**2)
                
                current_s_values[veh.id] = s
                current_actual_s_values[veh.id] = actual_s_values[veh.id]
                velocities[veh.id] = speed

            sorted_agents = sorted(agent_ids, key=lambda vid: current_actual_s_values[vid])
            
            # --- 2. Barrier Control (Intersection Logic) ---
            barrier_controls = {vid: 0.0 for vid in agent_ids}
            
            if step_count > WARMUP_STEPS:
                # [Request: Filter by 10 meters physical distance]
                # Group 1: Approaching the Crossing at ~L/2 (pi)
                candidates_1 = [
                    vid for vid in agent_ids 
                    if (geom.dist_at_pi - APPROACH_WINDOW_M) < current_s_values[vid] < geom.dist_at_pi
                ]
                
                # Group 2: Approaching the Crossing at ~L (2pi)
                candidates_2 = [
                    vid for vid in agent_ids 
                    if (geom.total_length - APPROACH_WINDOW_M) < current_s_values[vid] < geom.total_length
                ]

                if candidates_1 and candidates_2:
                    # Pick the car closest to the crash point in each group
                    c1 = max(candidates_1, key=lambda vid: current_s_values[vid])
                    c2 = max(candidates_2, key=lambda vid: current_s_values[vid])
                    
                    # [cite_start]Calculate physical distance remaining to crossing [cite: 8]
                    dist_to_cross_1 = geom.dist_at_pi - current_s_values[c1]
                    dist_to_cross_2 = geom.total_length - current_s_values[c2]
                    
                    # [cite_start]Diff in meters [cite: 9]
                    diff_meters = dist_to_cross_2 - dist_to_cross_1
                    
                    # [cite_start]Apply Barrier Control [cite: 14, 15]
                    if abs(diff_meters) < BARRIER_ACTIVATION_DIST:
                        # Singularity avoidance
                        if abs(diff_meters) < EPSILON_DIFF: 
                            safe_diff = EPSILON_DIFF * (1.0 if diff_meters >= 0 else -1.0)
                        else: 
                            safe_diff = diff_meters
                            
                        # [cite_start]Gradient Calculation [cite: 17]
                        gradient = -1.0 / safe_diff
                        
                        barrier_controls[c2] = gradient
                        barrier_controls[c1] = -gradient # [cite: 18]

            dash_lines = []
            
            # Determine Global Nominal Speed (Ramping)
            if step_count <= WARMUP_STEPS: current_nominal_speed = 0.0
            elif step_count <= (WARMUP_STEPS + RAMP_STEPS):
                progress = (step_count - WARMUP_STEPS) / RAMP_STEPS
                current_nominal_speed = TARGET_SPEED_KMH * progress
            else: current_nominal_speed = TARGET_SPEED_KMH

            for i, vid in enumerate(sorted_agents):
                veh = [v for v in vehicles_list if v.id == vid][0]
                
                # --- 3. Headway Control (Distance Based) ---
                u_headway = 0.0
                gap_meters = TOTAL_TRACK_LENGTH
                
                if N > 1:
                    leader_idx = (i + 1) % N
                    leader_id = sorted_agents[leader_idx]
                    
                    actual_s_me = current_actual_s_values[vid]
                    actual_s_leader = current_actual_s_values[leader_id]
                    
                    # Calculate gap from actual distance travelled.
                    gap_meters = (actual_s_leader - actual_s_me) % TOTAL_TRACK_LENGTH
                    
                    if step_count > WARMUP_STEPS:
                        headway_error_meters = gap_meters - TARGET_GAP_METERS
                        u_headway = GAIN_HEADWAY * headway_error_meters
                
                # --- 4. Combine Controls ---
                u_barrier_raw = GAIN_BARRIER * barrier_controls[vid]
                u_barrier = max(min(u_barrier_raw, MAX_BARRIER_KMH), -MAX_BARRIER_KMH)

                if step_count <= WARMUP_STEPS:
                    target_speed = 0.0
                    agent_pids[vid].reset()
                else:
                    # [cite_start]Base Speed + Headway + Barrier [cite: 19]
                    target_speed = current_nominal_speed + u_headway + u_barrier
                    if target_speed < 0: target_speed = 0 # [cite: 22]

                # --- 5. Lateral Control (Steering) ---
                tracker = agent_trackers[vid]
                dt_lookahead = 15.0 / R 
                t_future = tracker.t + dt_lookahead
                math_target = tracker.get_math_location(t_future)
                wp_target = carla_map.get_waypoint(math_target, project_to_road=True, lane_type=carla.LaneType.Driving)
                steer_target = wp_target.transform.location if wp_target else math_target 
                
                curr_loc = veh.get_location()
                math_pt = tracker.get_math_location(tracker.t)
                lateral_error = curr_loc.distance(math_pt)
                veh_trans = veh.get_transform()
                veh_fwd = veh_trans.get_forward_vector()
                vec_to = np.array([steer_target.x - curr_loc.x, steer_target.y - curr_loc.y])
                norm = np.linalg.norm(vec_to)
                if norm > 0: vec_to /= norm
                cross = veh_fwd.x*vec_to[1] - veh_fwd.y*vec_to[0]
                steer_cmd = np.clip(cross, -1.0, 1.0)
                
                
                # --- 6. Actuation ---
                curr_speed = velocities[vid]
                throttle_brake = agent_pids[vid].run(target_speed, curr_speed, DT)
                
                control = carla.VehicleControl()
                control.steer = float(steer_cmd)
                
                if step_count <= WARMUP_STEPS:
                    control.throttle = 0.0; control.brake = 1.0; control.hand_brake = True
                else:
                    control.hand_brake = False
                    if throttle_brake >= 0:
                        control.throttle = min(throttle_brake, 1.0); control.brake = 0.0
                    else:
                        control.throttle = 0.0; control.brake = min(abs(throttle_brake), 1.0)
                
                veh.apply_control(control)
                if speed_csv_writer is not None and vid in log_vehicle_ids:
                    speed_csv_writer.writerow([
                        f"{elapsed_time:.2f}",
                        step_count,
                        vid,
                        f"{curr_speed:.3f}",
                        f"{target_speed:.3f}",
                        f"{current_s_values[vid]:.1f}",
                        f"{current_actual_s_values[vid]:.1f}",
                        f"{gap_meters:.1f}",
                        f"{u_headway:.3f}",
                        f"{u_barrier:.3f}",
                        f"{lateral_error:.3f}",
                    ])
                dash_lines.append(f"{vid:<4} | {gap_meters:6.1f} | {u_headway:6.2f} | {u_barrier:6.2f} | {curr_speed:5.1f}")

            if speed_csv_file is not None:
                speed_csv_file.flush()

            # --- 7. Visualization ---
            if step_count % 5 == 0:
                print("\033[H\033[J")
                if step_count <= WARMUP_STEPS: status = "SETTLING..."
                elif step_count <= WARMUP_STEPS + RAMP_STEPS: status = f"RAMPING UP ({current_nominal_speed:.1f} km/h)"
                else: status = "ACTIVE"
                
                print(f"--- STEP {step_count} [{status}] ---")
                print(f"Target Gap: {TARGET_GAP_METERS:.1f} m")
                print(f"{'ID':<4} | {'Gap(m)':<6} | {'u_Head':<6} | {'u_Bar':<6} | {'Speed':<5}")
                for line in dash_lines: print(line)

    except KeyboardInterrupt: print("\nStopping...")
    finally:
        if speed_csv_file is not None:
            speed_csv_file.close()
            print(f"Vehicle speed log saved to {speed_csv_filename}")
        for s in sensors_list: s.destroy()
        for v in vehicles_list: v.destroy()

if __name__ == '__main__':
    main()

import os
import sys
import time
import random
import math
import numpy as np


# Ensure CARLA Python API can be found
try:
    sys.path.append('C:/CARLA_0.9.15/PythonAPI/carla')
except IndexError:
    pass

import carla

def get_unique_waypoints(town_map, distance_interval=0.3):
    """
    Generates waypoints and filters duplicates cleanly.
    Optimized to handle custom maps quickly.
    """
    print(f"Generating raw waypoints every {distance_interval} meters...")
    all_waypoints = town_map.generate_waypoints(distance_interval)
    print(f"Generated {len(all_waypoints)} raw waypoints.")
    
    print("Filtering unique waypoints...")
    unique_waypoints = []
    
    # Using a spatial grid/threshold tracking approach for performance
    for wp in all_waypoints:
        if len(unique_waypoints) == 0:
            unique_waypoints.append(wp)
            continue
            
        found = False
        for uwp in unique_waypoints:
            # Check if this waypoint is spatially identical to an existing one
            if (abs(uwp.transform.location.x - wp.transform.location.x) < 0.1 and
                abs(uwp.transform.location.y - wp.transform.location.y) < 0.1 and
                abs(uwp.transform.rotation.yaw - wp.transform.rotation.yaw) < 20):
                found = True
                break
                
        if not found:
            unique_waypoints.append(wp)
            
    print(f"Found {len(unique_waypoints)} unique waypoints total.")
    return unique_waypoints

def main():
    # 1. Connect to the CARLA Simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Retrieves whatever map is currently loaded on the server (works for custom maps)
    world = client.get_world()
    town_map = world.get_map()
    print(f"Successfully connected to map: {town_map.name}")
    
    # Clear existing vehicles before spawning to avoid map clutter
    print("Cleaning up existing vehicles...")
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()

    # 2. Extract Waypoints
    unique_waypoints = get_unique_waypoints(town_map, distance_interval=0.3)

    # 3. Spawn Tracker Vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('*model3*')[0]
    
    spawn_points = town_map.get_spawn_points()
    if not spawn_points:
        print("CRITICAL ERROR: No spawn points found. Ensure your custom map has an OpenDRIVE (.xodr) definition.")
        return
        
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    
    if vehicle is None:
        print("Failed to spawn vehicle. Rerun the script to attempt another spawn position.")
        return
    print(f"Spawned tracking vehicle ID: {vehicle.id}")

    # Enable autopilot so vehicle drives on its own
    vehicle.set_autopilot(True)
    print("Autopilot enabled — vehicle will drive itself.")

    spectator = world.get_spectator()

    # Draw a lower-density grid of all waypoints so you can see the paths
    print("Rendering map layout visualizer...")
    all_display_wps = town_map.generate_waypoints(2.5)  # Sparsely spaced for performance
    for wp in all_display_wps:
        world.debug.draw_point(
            wp.transform.location,
            size=0.05,
            color=carla.Color(r=0, g=255, b=0),  # Bright green layout dots
            life_time=120.0  # Visible for 2 minutes
        )

    # Move spectator to a corner of the map so you can verify dots cover the full map
    xs = [wp.transform.location.x for wp in all_display_wps]
    ys = [wp.transform.location.y for wp in all_display_wps]
    corner_x, corner_y = min(xs), min(ys)
    overview_z = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.6
    spectator.set_transform(carla.Transform(
        carla.Location(x=corner_x, y=corner_y, z=overview_z),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
    ))
    print(f"Spectator at map corner ({corner_x:.1f}, {corner_y:.1f}) z={overview_z:.1f}m — press Enter to start following the vehicle.")
    input()

    # 4. Tracking & Measurement Loop
    try:
        print("\n=== Tracking loop active. Press Ctrl+C to stop. ===")
        while True:
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location

            # Follow the vehicle overhead
            spectator.set_transform(carla.Transform(
                carla.Location(x=vehicle_loc.x, y=vehicle_loc.y, z=40.0),
                carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0)
            ))

            curr_distance = 1000.0
            selected_wp = None

            # Find the closest waypoint out of our unique collection
            for wp in unique_waypoints:
                dist = vehicle_loc.distance(wp.transform.location)
                if dist < curr_distance:
                    curr_distance = dist
                    selected_wp = wp

            if selected_wp:
                # Calculate tracking telemetry
                distance_to_wp = selected_wp.transform.location.distance(vehicle_transform.location)
                direction_difference = (vehicle_transform.rotation.yaw - selected_wp.transform.rotation.yaw) % 180

                # Output telemetry to the console
                print(f"Deviation from waypoint: {distance_to_wp:.4f} meters | Angle discrepancy: {direction_difference:.2f} degrees")

                # Draw the target waypoint tracking indicator in the simulator window
                world.debug.draw_string(
                    selected_wp.transform.location,
                    '^',
                    draw_shadow=False,
                    color=carla.Color(r=0, g=0, b=255),
                    life_time=0.15
                )

            time.sleep(0.1) # Limits processing loop to ~10Hz

    except KeyboardInterrupt:
        print("\nTracking paused by user.")
        
    finally:
        # 5. Safe Environment Cleanup
        print("\nCleaning up spawned actors before exiting...")
        if vehicle is not None:
            vehicle.destroy()
        print("Cleanup complete. Environment safe.")

if __name__ == '__main__':
    main()
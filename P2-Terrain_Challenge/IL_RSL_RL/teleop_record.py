#!/usr/bin/env python3

import subprocess
import time
import os
from datetime import datetime

def check_topics():
    """Check if required topics are publishing"""
    required_topics = [
        "/joint_states",
        "/joint_group_effort_controller/joint_trajectory", 
        "/cmd_vel",
        "/imu/data",
        "/odom"
    ]
    
    print("\nChecking topic availability...")
    result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True)
    available_topics = result.stdout.strip().split('\n')
    
    missing = []
    for topic in required_topics:
        if topic in available_topics:
            print(f"✓ {topic}")
        else:
            print(f"✗ {topic} - MISSING")
            missing.append(topic)
    
    if missing:
        print(f"\nWarning: Missing topics: {missing}")
        response = input("Continue anyway? (y/n): ")
        return response.lower() == 'y'
    
    print("\nAll required topics available!")
    return True

def record_demo(demo_name, duration, cmd_vel_commands):
    """Record a demonstration with specific velocity commands"""
    
    # Create recording directory
    record_dir = os.path.expanduser("~/rosbag_recordings")
    os.makedirs(record_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_name = f"{demo_name}_{timestamp}"
    
    print(f"\n{'='*60}")
    print(f"Recording: {demo_name}")
    print(f"Duration: {duration}s")
    print(f"Bag location: {record_dir}/{bag_name}")
    print(f"{'='*60}\n")
    
    # Start rosbag recording with all necessary topics
    bag_process = subprocess.Popen([
        "ros2", "bag", "record", 
        "-o", os.path.join(record_dir, bag_name),
        "/joint_states",                                  # Joint positions/velocities
        "/joint_group_effort_controller/joint_trajectory", # Commanded positions (actions)
        "/cmd_vel",                                       # Velocity commands
        "/imu/data",                                      # IMU data for gravity
        "/odom",                                          # Odometry for base velocities
        "/foot_contacts"                                  # Optional: if available
    ], stderr=subprocess.DEVNULL)  # Suppress warnings about missing topics
    
    time.sleep(3)  # Give rosbag more time to start
    print("Recording started...")
    
    # Execute velocity commands
    for cmd_time, cmd in cmd_vel_commands:
        print(f"[{cmd_time}s] Sending: {cmd}")
        subprocess.run([
            "ros2", "topic", "pub", "--once",
            "/cmd_vel", "geometry_msgs/msg/Twist", cmd
        ])
        time.sleep(cmd_time)
    
    # Stop recording
    print("\nStopping recording...")
    bag_process.terminate()
    bag_process.wait()
    
    # Check bag file was created
    bag_path = os.path.join(record_dir, bag_name)
    if os.path.exists(bag_path):
        print(f"✓ Recording saved to: {bag_path}")
        
        # Show bag info
        print("\nBag info:")
        subprocess.run(["ros2", "bag", "info", bag_path])
    else:
        print(f"✗ Error: Bag file not found at {bag_path}")

# Define demonstrations with updated velocity ranges
demos = [
    # Demo 1: Standing still (baseline)
    {
        "name": "standing_still",
        "duration": 20,
        "commands": [
            (10, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 2: Forward walking (updated ranges)
    {
        "name": "forward_walk",
        "duration": 20,
        "commands": [
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: 0.4, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (3, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 3: Backward walking
    {
        "name": "backward_walk",
        "duration": 20,
        "commands": [
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: -0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: -0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (3, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 4: Sideways walking
    {
        "name": "sideways_walk", 
        "duration": 20,
        "commands": [
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: -0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 5: Turning in place (updated ranges)
    {
        "name": "turning",
        "duration": 20,
        "commands": [
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (5, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.3}}'),
            (5, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 6: Combined motion (forward + turn)
    {
        "name": "forward_and_turn",
        "duration": 20,
        "commands": [
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'),
            (8, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.2}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    }
]

if __name__ == "__main__":
    print("Mini Pupper Demonstration Recorder")
    print("==================================")
    
    # Check if topics are available
    if not check_topics():
        print("Exiting due to missing topics.")
        exit(1)
    
    # Show available demos
    print("\nAvailable demonstrations:")
    for i, demo in enumerate(demos):
        print(f"{i+1}. {demo['name']} ({demo['duration']}s)")
    
    # Get user selection
    try:
        choice = input("\nSelect demo number (or 'a' for all): ").strip().lower()
        
        if choice == 'a':
            confirm = input("This will record all demos (~2 minutes). Continue? (y/n): ")
            if confirm.lower() == 'y':
                for demo in demos:
                    record_demo(demo["name"], demo["duration"], demo["commands"])
                    time.sleep(3)  # Pause between recordings
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                demo = demos[idx]
                record_demo(demo["name"], demo["duration"], demo["commands"])
            else:
                print("Invalid selection")
                
    except KeyboardInterrupt:
        print("\nRecording cancelled")
    except Exception as e:
        print(f"Error: {e}")
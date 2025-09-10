#!/usr/bin/env python3

import subprocess
import time
import os
from datetime import datetime
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
import threading
import signal
import sys
import re
import numpy as np
from collections import deque

def check_topics():
    """Check if required topics are publishing"""
    required_topics = [
        "/joint_states",
        "/joint_group_effort_controller/joint_trajectory", 
        "/cmd_vel",
    ]

    optional_topics = [
        "/body_pose", 
        "/foot_contacts"
    ]
    
    print("\nChecking topic availability...")
    result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True)
    available_topics = result.stdout.strip().split('\n')
    
    missing_required = []
    for topic in required_topics:
        if topic in available_topics:
            print(f"✓ {topic}")
        else:
            print(f"✗ {topic} - MISSING")
            missing_required.append(topic)
    
    if missing_required:
        print(f"\nError: Missing required topics: {missing_required}")
        return False
    
    print("\nAll required topics available!")
    return True

def check_topic_rates():
    """Check publishing rates of topics"""
    print("\nChecking topic publishing rates (5 second sample)...")
    topics_to_check = [
        "/joint_states",
        "/joint_group_effort_controller/joint_trajectory",
        "/body_pose",
        "/foot_contacts"
    ]
    
    for topic in topics_to_check:
        try:
            result = subprocess.run(
                ["timeout", "5", "ros2", "topic", "hz", topic],
                capture_output=True, text=True
            )
            if result.stdout:
                # Extract average rate from output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "average rate:" in line:
                        print(f"{topic}: {line.strip()}")
                        break
            else:
                print(f"{topic}: No data received")
        except Exception as e:
            print(f"{topic}: Error checking rate - {e}")

def estimate_data_size(duration_seconds, num_demos):
    """Estimate total data size for recordings"""
    # Typical message sizes and rates
    estimates = {
        "/joint_states": {"rate": 50, "size": 500},  # bytes per message
        "/joint_group_effort_controller/joint_trajectory": {"rate": 50, "size": 400},
        "/cmd_vel": {"rate": 50, "size": 200},
        "/imu/data": {"rate": 100, "size": 300},
        "/odom": {"rate": 50, "size": 400}
    }
    
    total_bytes_per_second = 0
    print("\nEstimated data rates:")
    for topic, info in estimates.items():
        bytes_per_sec = info["rate"] * info["size"]
        total_bytes_per_second += bytes_per_sec
        print(f"{topic}: {bytes_per_sec/1024:.1f} KB/s")
    
    print(f"\nTotal rate: {total_bytes_per_second/1024:.1f} KB/s")
    
    # Add 20% overhead for bag metadata
    total_size = total_bytes_per_second * duration_seconds * 1.2
    
    print(f"\nPer demo ({duration_seconds}s): {total_size/1024/1024:.1f} MB")
    print(f"All {num_demos} demos: {total_size * num_demos/1024/1024:.1f} MB")
    print(f"Total recording time: {duration_seconds * num_demos / 60:.1f} minutes")

class ObservationLogger(Node):
    def __init__(self):
        super().__init__('observation_logger')
        
        # State storage (matching mlpcontrolnode.py)
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.prev_joint_positions = np.zeros(12)
        self.prev_joint_time = None
        
        # History buffers
        self.joint_pos_history = deque(maxlen=3)
        self.joint_vel_history = deque(maxlen=3)
        self.prev_actions = deque([np.zeros(12)] * 3, maxlen=3)
        
        # Latest sensor data
        self.gravity_vec = np.array([0., 0., -9.81])
        self.ang_vel = np.zeros(3)
        self.cmd_vel = np.zeros(3)

        self.body_position = np.zeros(3)
        self.body_orientation = np.array([1., 0., 0., 0.])  # quaternion w,x,y,z

        # Foot contacts (if available)
        self.foot_contacts = np.ones(4) 
        
        # Gait tracking
        self.gait_freq = 3.0  # Hz - should match your controller
        self.phase_start_time = time.time()
        
        # Observations list for saving
        self.observations = []
        self.observation_times = []
        
        # Subscribe to all necessary topics
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
            
    def joint_callback(self, msg):
        """Process joint states - matching mlpcontrolnode.py logic"""
        current_time = time.time()
        
        # Update positions
        self.prev_joint_positions = self.joint_positions.copy()
        self.joint_positions = np.array(msg.position[:12])
        
        # Estimate velocities using time difference
        if self.prev_joint_time is not None:
            dt = current_time - self.prev_joint_time
            if dt > 0:
                raw_vel = (self.joint_positions - self.prev_joint_positions) / dt
                # Apply same smoothing as controller
                alpha = 0.2
                self.joint_velocities = alpha * raw_vel + (1 - alpha) * self.joint_velocities
                
        self.prev_joint_time = current_time
        
        # Update history
        self.joint_pos_history.append(self.joint_positions.copy())
        self.joint_vel_history.append(self.joint_velocities.copy())
        
    def imu_callback(self, msg):
        """Extract gravity vector and angular velocity"""
        # Convert quaternion to rotation matrix
        q = msg.orientation
        R = self._quat_to_rot([q.w, q.x, q.y, q.z])
        
        # Gravity in body frame (matching controller)
        self.gravity_vec = R.T @ np.array([0., 0., -9.81])
        
        # Angular velocity
        self.ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y, 
            msg.angular_velocity.z
        ])
        
    def cmd_callback(self, msg):
        """Store command velocities"""
        self.cmd_vel = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.angular.z
        ])
        
    def get_observation(self):
        """Build 48-dim observation vector matching mlpcontrolnode.py"""
        obs_parts = []
        
        # 1. Command velocities (3)
        obs_parts.append(self.cmd_vel)
        
        # 2. Joint positions (12) 
        obs_parts.append(self.joint_positions)
        
        # 3. Joint velocities (12)
        obs_parts.append(self.joint_velocities)
        
        # 4. Previous actions (12) - use joint positions as proxy
        if len(self.joint_pos_history) >= 2:
            obs_parts.append(self.joint_pos_history[-2])
        else:
            obs_parts.append(self.joint_positions)
            
        # 5. Gravity vector (3)
        obs_parts.append(self.gravity_vec)
        
        # 6. Angular velocity (3)
        obs_parts.append(self.ang_vel)
        
        # 7. Gait phase (2)
        phase = self._compute_gait_phase()
        obs_parts.append(phase)
        
        # 8. Foot contact (1) - simplified, you may need actual contact sensors
        obs_parts.append(np.array([1.0]))  # Assume contact
        
        # Concatenate all parts
        obs = np.concatenate(obs_parts)
        assert obs.shape[0] == 48, f"Expected 48 dims, got {obs.shape[0]}"
        
        return obs
        
    def _compute_gait_phase(self):
        """Compute gait phase as [sin, cos]"""
        current_time = time.time()
        phase_time = current_time - self.phase_start_time
        phase = 2 * np.pi * self.gait_freq * phase_time
        return np.array([np.sin(phase), np.cos(phase)])
        
    def _quat_to_rot(self, q):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        
    def log_observation(self):
        """Store current observation with timestamp"""
        try:
            obs = self.get_observation()
            self.observations.append(obs)
            self.observation_times.append(time.time())
        except Exception as e:
            self.get_logger().warn(f"Failed to log observation: {e}")
        
    def save_observations(self, filepath):
        """Save observations to numpy file"""
        if self.observations:
            np.savez(filepath,
                     observations=np.array(self.observations),
                     times=np.array(self.observation_times))
            self.get_logger().info(f"Saved {len(self.observations)} observations to {filepath}")
            return True
        return False

class SynchronizedRecorder(Node):
    def __init__(self):
        super().__init__('synchronized_recorder')
        
        # Publisher for cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Create timer for consistent publishing rate (50Hz to match your controller)
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50Hz
        
        # Current command and timing
        self.current_twist = Twist()
        self.command_queue = []
        self.start_time = None
        self.recording = False
        self.bag_process = None
        self.current_demo_name = None
        self.record_dir = None
        
        # Create observation logger
        self.obs_logger = ObservationLogger()
        
        # Timer for observation logging at 50Hz
        self.obs_timer = self.create_timer(0.02, self.log_observation_callback)
        
    def timer_callback(self):
        """Publish commands at consistent 50Hz rate"""
        if not self.recording:
            return
            
        current_time = time.time() - self.start_time
        
        # Check if we need to advance to next command
        while self.command_queue and current_time >= self.command_queue[0][0]:
            _, twist = self.command_queue.pop(0)
            self.current_twist = twist
            self.get_logger().info(f"[{current_time:.2f}s] New command: x={twist.linear.x:.2f}, y={twist.linear.y:.2f}, z={twist.angular.z:.2f}")
        
        # Always publish current command at 50Hz
        self.cmd_vel_pub.publish(self.current_twist)
        
    def log_observation_callback(self):
        """Log observations at 50Hz during recording"""
        if self.recording:
            self.obs_logger.log_observation()
    
    def load_demo_commands(self, commands):
        """Convert demo commands to timed Twist messages"""
        self.command_queue = []
        cumulative_time = 0
        
        for duration, cmd_str in commands:
            # Parse command string to Twist using regex for robustness
            twist = Twist()
            
            # Extract linear x, y values
            linear_x_match = re.search(r'linear:.*?x:\s*([-\d.]+)', cmd_str)
            linear_y_match = re.search(r'linear:.*?y:\s*([-\d.]+)', cmd_str)
            angular_z_match = re.search(r'angular:.*?z:\s*([-\d.]+)', cmd_str)
            
            if linear_x_match:
                twist.linear.x = float(linear_x_match.group(1))
            if linear_y_match:
                twist.linear.y = float(linear_y_match.group(1))
            if angular_z_match:
                twist.angular.z = float(angular_z_match.group(1))
            
            self.command_queue.append((cumulative_time, twist))
            cumulative_time += duration
            
    def start_recording(self, demo_name, duration):
        """Start synchronized recording"""
        self.recording = True
        self.start_time = time.time()
        self.current_demo_name = demo_name
        
        # Start rosbag with use_sim_time for better synchronization
        self.record_dir = os.path.expanduser("~/rosbag_recordings")
        os.makedirs(self.record_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_name = f"{demo_name}_{timestamp}"
        
        # Record all available topics
        topics_to_record = [
            "/joint_states",
            "/joint_group_effort_controller/joint_trajectory",
            "/cmd_vel",
            "/body_pose",
            "/foot_contacts"
        ]
        
        # Check which topics actually exist
        result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True)
        available_topics = result.stdout.strip().split('\n')
        topics_to_record = [t for t in topics_to_record if t in available_topics]
        
        self.get_logger().info(f"Recording topics: {topics_to_record}")
        
        # Record with compression to save space
        cmd = [
            "ros2", "bag", "record",
            "-o", os.path.join(self.record_dir, bag_name),
            "--compression-mode", "message",
            "--compression-format", "zstd"
        ] + topics_to_record
        
        self.bag_process = subprocess.Popen(cmd)
        
        return bag_name, self.record_dir

    def cleanup(self):
        """Cleanup resources and save observations"""
        self.recording = False
        
        # Save observations if we have any
        if self.current_demo_name and self.record_dir and self.obs_logger.observations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            obs_filename = f"{self.current_demo_name}_{timestamp}_observations.npz"
            obs_filepath = os.path.join(self.record_dir, obs_filename)
            
            if self.obs_logger.save_observations(obs_filepath):
                self.get_logger().info(f"Observations saved to: {obs_filepath}")
                
                # Quick validation
                data = np.load(obs_filepath)
                obs = data['observations']
                times = data['times']
                self.get_logger().info(f"Observation shape: {obs.shape} ({len(obs)} samples over {times[-1]-times[0]:.1f}s)")
        
        if self.bag_process:
            self.bag_process.terminate()
            self.bag_process.wait()

def record_demo_synchronized(demo):
    """Record with proper temporal alignment"""
    
    # Initialize ROS2
    rclpy.init()
    
    # Create executor for multiple nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    
    recorder = SynchronizedRecorder()
    executor.add_node(recorder)
    executor.add_node(recorder.obs_logger)
    
    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n\nInterrupted! Cleaning up...")
        recorder.cleanup()
        recorder.destroy_node()
        recorder.obs_logger.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Load demo commands
        recorder.load_demo_commands(demo["commands"])
        
        # Start recording
        bag_name, record_dir = recorder.start_recording(demo["name"], demo["duration"])
        
        print(f"\n{'='*60}")
        print(f"Recording: {demo['name']} (synchronized)")
        print(f"Duration: {demo['duration']}s")
        print(f"Publishing at 50Hz for temporal alignment")
        print(f"Recording 48-dim observations matching MLP format")
        print(f"{'='*60}\n")
        
        # Spin in separate thread
        spin_thread = threading.Thread(target=executor.spin)
        spin_thread.start()
        
        # Wait 0.5s to ensure recording has started
        time.sleep(0.5)
        
        # Wait for demo duration
        time.sleep(demo["duration"])
        
        # Stop recording
        print("\nStopping recording...")
        recorder.cleanup()
        
        # Wait for bag process to finish
        time.sleep(1)
        
        # Cleanup
        recorder.destroy_node()
        recorder.obs_logger.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        spin_thread.join()
        
        # Verify recording
        bag_path = os.path.join(record_dir, bag_name)
        if os.path.exists(bag_path):
            print(f"✓ Recording saved to: {bag_path}")
            # Show bag info
            result = subprocess.run(["ros2", "bag", "info", bag_path], capture_output=True, text=True)
            print(result.stdout)
        
        return bag_path
        
    except Exception as e:
        print(f"Error during recording: {e}")
        recorder.cleanup()
        recorder.destroy_node()
        recorder.obs_logger.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        raise

# Define demonstrations with equal duration and balanced structure
demos = [
    # Demo 1: Standing still (baseline)
    {
        "name": "standing_still",
        "duration": 66,
        "commands": [
            (66, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 2: Forward walking
    {
        "name": "forward_walk",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.4, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 3: Backward walking
    {
        "name": "backward_walk",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: -0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: -0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: -0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 4: Sideways walking (left and right)
    {
        "name": "sideways_walk", 
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.0, y: -0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 5: Turning in place
    {
        "name": "turning",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'),
            (20, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.3}}'),
            (20, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 6: Combined motion (forward + turn)
    {
        "name": "forward_and_turn",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'),
            (20, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.2}}'),
            (20, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 7: Diagonal walking (forward-left, forward-right)
    {
        "name": "diagonal_walk",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.2, y: 0.15, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.2, y: -0.15, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.2, y: 0.15, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 8: Mixed velocities (slow to fast)
    {
        "name": "speed_variations",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (20, '{linear: {x: 0.15, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 9: Walk to turn smooth transition
    {
        "name": "walk_to_turn_smooth",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}'),
            (10, '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'),
            (10, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'),
            (10, '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}'),
            (10, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 10: Turn to walk smooth transition
    {
        "name": "turn_to_walk_smooth",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.3}}'),
            (10, '{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'),
            (10, '{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}'),
            (10, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 11: Sideways to forward transition
    {
        "name": "sideways_to_forward",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.05, y: 0.15, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.1, y: 0.1, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.15, y: 0.05, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (10, '{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 12: Complex multi-directional transition
    {
        "name": "complex_transition",
        "duration": 66,
        "commands": [
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.15, y: 0.1, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}'),
            (8, '{linear: {x: 0.1, y: 0.15, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.15}}'),
            (8, '{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'),
            (8, '{linear: {x: -0.1, y: 0.15, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.15}}'),
            (8, '{linear: {x: -0.15, y: 0.1, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}'),
            (8, '{linear: {x: -0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    },
    
    # Demo 13: Emergency stop from various motions
    {
        "name": "emergency_stops",
        "duration": 66,
        "commands": [
            (4, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.4, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.0, y: 0.3, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.4}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: -0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (8, '{linear: {x: 0.2, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (2, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'),
            (6, '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}')
        ]
    }
]

if __name__ == "__main__":
    print("Mini Pupper Synchronized Demonstration Recorder")
    print("==============================================")
    
    # Check if topics are available
    if not check_topics():
        print("Exiting due to missing topics.")
        exit(1)
    
    # Check topic publishing rates
    check_topic_rates()
    
    # Estimate data sizes
    estimate_data_size(66, len(demos))
    
    # Show available demos
    print("\nAvailable demonstrations:")
    for i, demo in enumerate(demos):
        print(f"{i+1}. {demo['name']} ({demo['duration']}s)")
    
    # Get user selection
    try:
        choice = input("\nSelect demo number (or 'a' for all, 'r' for all with repetitions): ").strip().lower()
        
        if choice == 'r':
            # Record with repetitions
            num_repetitions = 3
            total_demos = len(demos) * num_repetitions
            total_time = total_demos * 66 / 60
            total_size_mb = total_demos * 15.5  # Estimated MB per demo
            
            confirm = input(f"\nThis will record {total_demos} demos ({num_repetitions} reps × {len(demos)} types)")
            confirm += input(f"\nTotal time: ~{total_time:.1f} minutes")
            confirm += input(f"\nEstimated size: ~{total_size_mb:.0f} MB")
            confirm = input("\nContinue? (y/n): ")
            
            if confirm.lower() == 'y':
                all_recordings = []
                
                for rep in range(num_repetitions):
                    print(f"\n{'='*60}")
                    print(f"REPETITION {rep+1}/{num_repetitions}")
                    print(f"{'='*60}")
                    
                    for i, demo in enumerate(demos):
                        demo_copy = demo.copy()
                        demo_copy['name'] = f"{demo['name']}_rep{rep+1}"
                        
                        demo_num = i + 1 + (rep * len(demos))
                        print(f"\n[{demo_num}/{total_demos}] Recording {demo_copy['name']}...")
                        
                        try:
                            bag_path = record_demo_synchronized(demo_copy)
                            all_recordings.append(bag_path)
                        except Exception as e:
                            print(f"Error recording {demo_copy['name']}: {e}")
                            continue
                        
                        # Break between demos within repetition
                        if i < len(demos) - 1:
                            print("\nWaiting 3 seconds before next demo...")
                            time.sleep(3)
                    
                    # Longer break between repetitions
                    if rep < num_repetitions - 1:
                        print(f"\n{'='*60}")
                        print(f"Completed repetition {rep+1}/{num_repetitions}")
                        print("Taking 10 second break before next repetition...")
                        print(f"{'='*60}")
                        time.sleep(10)
                
                # Summary
                print(f"\n{'='*60}")
                print("RECORDING COMPLETE!")
                print(f"{'='*60}")
                print(f"Total recordings: {len(all_recordings)}")
                print(f"Total time: {total_demos * 66 / 60:.1f} minutes")
                print(f"Recordings saved to: ~/rosbag_recordings/")
                
        elif choice == 'a':
            # Record all demos once
            confirm = input(f"\nThis will record all {len(demos)} demos (~{len(demos) * 66 / 60:.1f} minutes). Continue? (y/n): ")
            if confirm.lower() == 'y':
                for i, demo in enumerate(demos):
                    print(f"\n[{i+1}/{len(demos)}] Recording {demo['name']}...")
                    record_demo_synchronized(demo)
                    if i < len(demos) - 1:
                        print("\nWaiting 3 seconds before next recording...")
                        time.sleep(3)
                        
        else:
            # Record single demo
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                demo = demos[idx]
                
                # Ask for repetitions for single demo
                reps = input("How many times to record? (default: 1): ").strip()
                num_reps = int(reps) if reps.isdigit() else 1
                
                for rep in range(num_reps):
                    demo_copy = demo.copy()
                    if num_reps > 1:
                        demo_copy['name'] = f"{demo['name']}_rep{rep+1}"
                    
                    print(f"\n[{rep+1}/{num_reps}] Recording {demo_copy['name']}...")
                    record_demo_synchronized(demo_copy)
                    
                    if rep < num_reps - 1:
                        print("\nWaiting 3 seconds before next repetition...")
                        time.sleep(3)
            else:
                print("Invalid selection")
                
    except KeyboardInterrupt:
        print("\nRecording cancelled")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# 
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import torch
import numpy as np
import threading
import time
from collections import deque

class MLPController(Node):
    def __init__(self):
        super().__init__('mlp_controller')
        
        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # State variables
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.array([0.0, 0.0, -1.0])  # Initial guess
        self.velocity_commands = np.array([0.05, 0.0, 0.0])  # Forward command
        self.last_action = np.zeros(12)
        
        # Joint velocity estimation
        self.prev_joint_positions = np.zeros(12)
        self.prev_joint_time = None
        self.estimated_joint_velocities = np.zeros(12)
        
        # Use a moving average filter for smoother velocity estimates
        self.velocity_history = deque(maxlen=3)  # Keep last 3 velocity estimates
        
        # Action history for smoothing
        self.action_history = deque(maxlen=3)
        
        # Joint mapping (Isaac -> ROS2 topic names)
        self.joint_mapping = {
            # Isaac Lab order from your config
            0: 'base_lf1',    # LF leg (front-left) hip
            1: 'lf1_lf2',     # LF thigh  
            2: 'lf2_lf3',     # LF calf
            3: 'base_rf1',    # RF leg (front-right) hip
            4: 'rf1_rf2',     # RF thigh
            5: 'rf2_rf3',     # RF calf
            6: 'base_lb1',    # LB leg (back-left) hip
            7: 'lb1_lb2',     # LB thigh
            8: 'lb2_lb3',     # LB calf
            9: 'base_rb1',    # RB leg (back-right) hip
            10: 'rb1_rb2',    # RB thigh
            11: 'rb2_rb3'     # RB calf
        }
        
        # ROS2 subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Additional sensor subscribers for real sensor data
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        
        # Store real sensor data
        self.base_lin_vel_real = np.zeros(3)
        self.base_ang_vel_real = np.zeros(3) 
        self.gravity_real = np.array([0.0, 0.0, -1.0])
        
        # ROS2 publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory,
            '/joint_group_effort_controller/joint_trajectory',
            10
        )
        
        # Control timer (5 Hz for servo compatibility)
        self.control_timer = self.create_timer(0.2, self.control_loop)
        self.command_counter = 0
        
        # Safety limits
        self.max_joint_change = 0.05  # Max change per timestep (rad)
        
        # Debug flag
        self._joint_mapping_success = False
        
        # Print expected joint mapping for verification
        self.get_logger().info('Expected joint mapping:')
        for idx, name in self.joint_mapping.items():
            self.get_logger().info(f'  {idx}: {name}')
        
        self.get_logger().info('MLP Controller initialized')
        
    def load_model(self):
        """Load the trained MLP model"""
        try:
            # Define the model architecture (matching your training)
            model = torch.nn.Sequential(
                torch.nn.Linear(76, 512),
                torch.nn.ELU(),
                torch.nn.Linear(512, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 12)
            )
            
            # Load your trained weights
            checkpoint_path = "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/sim2real/walkingmlp.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract actor weights
            state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('actor.'):
                    new_key = key.replace('actor.', '')
                    state_dict[new_key] = value
            
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self.device)
            
            # Load the std values for action scaling
            self.action_std = checkpoint['model_state_dict']['std'].cpu().numpy()
            
            self.get_logger().info('Model loaded successfully')
            return model
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None
    
    def joint_state_callback(self, msg):
        """Update robot state with estimated joint velocities"""
        try:
            current_time = time.time()
            
            # Map joint names to our internal representation
            name_to_index = {name: idx for idx, name in self.joint_mapping.items()}
            temp_positions = np.zeros(12)
            
            # Extract current positions
            for i, joint_name in enumerate(msg.name):
                if joint_name in name_to_index:
                    joint_idx = name_to_index[joint_name]
                    if i < len(msg.position):
                        temp_positions[joint_idx] = msg.position[i]
            
            # Estimate velocities if we have previous data
            if self.prev_joint_time is not None:
                dt = current_time - self.prev_joint_time
                
                if dt > 0.001:  # Avoid division by zero and too-small time steps
                    # Calculate raw velocity estimates
                    raw_velocities = (temp_positions - self.prev_joint_positions) / dt
                    
                    # Apply smoothing filter
                    self.velocity_history.append(raw_velocities)
                    
                    if len(self.velocity_history) > 0:
                        # Use moving average for smoother estimates
                        self.estimated_joint_velocities = np.mean(
                            list(self.velocity_history), axis=0
                        )
                    else:
                        self.estimated_joint_velocities = raw_velocities
            
            # Update stored values
            self.joint_positions = temp_positions.copy()
            self.joint_velocities = self.estimated_joint_velocities.copy()
            self.prev_joint_positions = temp_positions.copy()
            self.prev_joint_time = current_time
            
        except Exception as e:
            self.get_logger().error(f'Error in joint state callback: {e}')
    
    def odom_callback(self, msg):
        """Get real base velocity from odometry"""
        try:
            self.base_lin_vel_real = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])
            
            self.base_ang_vel_real = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z
            ])
        except Exception as e:
            self.get_logger().error(f'Error in odometry callback: {e}')

    def imu_callback(self, msg):
        """Get real gravity vector from IMU"""
        try:
            # Extract gravity from linear acceleration (when robot is stationary)
            # Note: This is approximate - real gravity extraction needs proper filtering
            self.gravity_real = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y, 
                msg.linear_acceleration.z
            ])
            
            # Normalize to unit vector
            gravity_norm = np.linalg.norm(self.gravity_real)
            if gravity_norm > 0.1:  # Avoid division by zero
                self.gravity_real = self.gravity_real / gravity_norm
                
        except Exception as e:
            self.get_logger().error(f'Error in IMU callback: {e}')
    
    def cmd_vel_callback(self, msg):
        """Update velocity commands from teleop or high-level planner"""
        self.velocity_commands[0] = msg.linear.x
        self.velocity_commands[1] = msg.linear.y  
        self.velocity_commands[2] = msg.angular.z
        
        self.get_logger().info(f'Velocity command: {self.velocity_commands}')
    
    def estimate_base_velocity(self):
        """Use real odometry data instead of estimates"""
        return self.base_lin_vel_real, self.base_ang_vel_real

    def get_observation(self):
        """Construct observation using real sensors + estimates"""
        
        # Get real base velocities from odometry
        lin_vel, ang_vel = self.estimate_base_velocity()
        
        # Use real gravity from IMU (or default if not available)
        projected_gravity = self.gravity_real
        
        # Core observations (48 dims) with real sensor data
        core_obs = np.concatenate([
            lin_vel,                        # Real base linear velocity (3)
            ang_vel,                        # Real base angular velocity (3)
            projected_gravity,              # Real gravity from IMU (3)
            self.velocity_commands,         # Velocity commands (3)
            self.joint_positions,           # Real joint positions (12)
            self.estimated_joint_velocities, # Estimated joint velocities (12)
            self.last_action               # Last action (12)
        ])
        
        # Add 28 dimensions to reach 76 - using minimal padding
        padding = np.zeros(28)
        full_obs = np.concatenate([core_obs, padding])
        
        # Apply minimal noise for robustness
        noise_scales = np.concatenate([
            [0.01] * 3,    # lin_vel noise  
            [0.01] * 3,    # ang_vel noise
            [0.01] * 3,    # gravity noise
            [0.0] * 3,     # no noise on commands
            [0.01] * 12,   # joint pos noise
            [0.1] * 12,    # joint vel noise  
            [0.0] * 12,    # no noise on last action
            [0.0] * 28     # no noise on padding
        ])
        
        # Add minimal noise
        full_obs += np.random.uniform(-0.001, 0.001, full_obs.shape) * noise_scales
        
        return full_obs.astype(np.float32)

    def control_loop(self):
        """Enhanced control loop with velocity estimation debugging"""
        if self.model is None:
            return
            
        try:
            self.command_counter += 1
            
            # Get observation
            obs = self.get_observation()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from MLP
            with torch.no_grad():
                raw_action = self.model(obs_tensor).cpu().numpy()[0]
            
            # Scale action by learned std and apply safety scaling
            position_action = raw_action * self.action_std * 0.02  # Increased from 0.01
            
            # Safety: limit rate of change
            if len(self.action_history) > 0:
                action_diff = position_action - self.action_history[-1]
                action_diff = np.clip(action_diff, -self.max_joint_change, self.max_joint_change)
                position_action = self.action_history[-1] + action_diff
            
            # Store for next iteration
            self.action_history.append(position_action.copy())
            self.last_action = position_action.copy()
            
            # Apply to default positions (action is relative to default)
            default_positions = np.array([
                0.0, 0.52, -1.05,  # LF leg (hip, thigh, calf)
                0.0, 0.52, -1.05,  # RF leg
                0.0, 0.52, -1.05,  # LB leg  
                0.0, 0.52, -1.05   # RB leg
            ])
            
            target_positions = default_positions + position_action
            
            # Safety limits based on servo specs and mechanical constraints
            joint_limits_low = np.array([
                -0.5, 0.0, -2.09,  # LF leg limits (hip, thigh, calf)
                -0.5, 0.0, -2.09,  # RF leg limits
                -0.5, 0.0, -2.09,  # LB leg limits
                -0.5, 0.0, -2.09   # RB leg limits
            ])
            
            joint_limits_high = np.array([
                0.5, 1.57, -0.52,  # LF leg limits (hip, thigh, calf)
                0.5, 1.57, -0.52,  # RF leg limits  
                0.5, 1.57, -0.52,  # LB leg limits
                0.5, 1.57, -0.52   # RB leg limits
            ])
            
            target_positions = np.clip(target_positions, joint_limits_low, joint_limits_high)
            
            # Create trajectory message (remove effort commands - let controller handle internally)
            trajectory_msg = JointTrajectory()
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            trajectory_msg.joint_names = [self.joint_mapping[i] for i in range(12)]
            
            point = JointTrajectoryPoint()
            point.positions = target_positions.tolist()
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 500000000  # 500ms
            
            trajectory_msg.points = [point]
            self.joint_cmd_pub.publish(trajectory_msg)
            
            # Enhanced debug info with velocity estimates
            self.get_logger().info(f'Command #{self.command_counter}')
            self.get_logger().info(f'Velocity cmd: {self.velocity_commands}')
            self.get_logger().info(f'Target pos: {target_positions[:4].round(3)}...')
            self.get_logger().info(f'Current pos: {self.joint_positions[:4].round(3)}...')
            self.get_logger().info(f'Est. velocities: {self.estimated_joint_velocities[:4].round(3)}...')
            self.get_logger().info(f'Max |velocity|: {np.max(np.abs(self.estimated_joint_velocities)):.3f} rad/s')
            self.get_logger().info(f'Raw action: {raw_action[:4].round(3)}...')
            self.get_logger().info('---')
                
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    controller = MLPController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
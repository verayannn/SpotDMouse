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
        self.model, self.obs_mean, self.obs_var = self.load_model()
        
        # State variables
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.array([0.0, 0.0, -1.0])  # Initial guess
        self.velocity_commands = np.zeros(3)  # Start with zero command
        self.last_action = np.zeros(12)
        
        # Action history for smoothing
        self.action_history = deque(maxlen=3)
        self.filtered_action = np.zeros(12)
        
        # Joint velocity estimation
        self.prev_joint_positions = None
        self.prev_joint_time = None
        self.velocity_history = deque(maxlen=5)
        
        # Default positions for Mini Pupper
        self.default_positions = np.array([
            0.0, 0.8, -1.6,  # LF leg (hip, thigh, calf)
            0.0, 0.8, -1.6,  # RF leg
            0.0, 0.8, -1.6,  # LB leg  
            0.0, 0.8, -1.6   # RB leg
        ])
        
        # Joint mapping (Isaac -> ROS2 topic names)
        self.joint_mapping = {
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
        
        # Create inverse mapping for faster lookup
        self.name_to_idx = {name: idx for idx, name in self.joint_mapping.items()}
        
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
        
        # Add IMU subscriber for better gravity vector
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Add odometry subscriber for base velocities
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # ROS2 publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory,
            '/joint_group_effort_controller/joint_trajectory',
            10
        )
        
        # Control parameters
        self.control_frequency = 50.0  # Hz
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)
        
        # Safety and tuning parameters
        self.action_scale = 0.05  # Scale factor for actions
        self.filter_alpha = 0.90   # Action smoothing (0.8 = 80% old, 20% new)
        self.max_joint_change = 0.08  # Max change per timestep (rad)
        self.cmd_vel_deadzone = 0.05  # Deadzone for velocity commands
        
        # Control step counter
        self.step_count = 0
        
        self.get_logger().info('MLP Controller initialized with normalization')
        if self.obs_mean is not None:
            self.get_logger().info(f'Obs mean range: [{self.obs_mean.min():.3f}, {self.obs_mean.max():.3f}]')
            self.get_logger().info(f'Obs var range: [{self.obs_var.min():.3f}, {self.obs_var.max():.3f}]')
        
    def load_model(self):
        """Load the trained MLP model with normalization statistics"""
        try:
            # Define the model architecture (matching your training)
            model = torch.nn.Sequential(
                torch.nn.Linear(48, 512),
                torch.nn.ELU(),
                torch.nn.Linear(512, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 12)
            )
            
            # Load your trained weights WITH STATS
            checkpoint_path = "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/sim2real/newwalkingmlp.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
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
            
            # CRITICAL: Load observation normalization statistics
            obs_mean = None
            obs_var = None
            
            if 'obs_rms_mean' in checkpoint:
                obs_mean = checkpoint['obs_rms_mean']
                obs_var = checkpoint['obs_rms_var']
                self.get_logger().info('✓ Loaded observation normalization statistics')
            else:
                self.get_logger().warn('⚠️ No normalization stats in checkpoint!')
                # Use default values as fallback
                obs_mean = np.zeros(48, dtype=np.float32)
                obs_var = np.ones(48, dtype=np.float32)
            
            self.get_logger().info('Model loaded successfully')
            return model, obs_mean, obs_var
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None, None, None
    
    def joint_state_callback(self, msg):
        """Update robot state from joint feedback with velocity estimation"""
        try:
            current_time = time.time()
            current_positions = np.zeros(12)
            
            # Map joint positions
            for joint_name, position in zip(msg.name, msg.position):
                if joint_name in self.name_to_idx:
                    idx = self.name_to_idx[joint_name]
                    current_positions[idx] = position
            
            # Update positions
            self.joint_positions = current_positions
            
            # Estimate velocities using finite differences
            if self.prev_joint_positions is not None and self.prev_joint_time is not None:
                dt = current_time - self.prev_joint_time
                if dt > 0.001:  # Avoid division by very small numbers
                    instantaneous_vel = (current_positions - self.prev_joint_positions) / dt
                    self.velocity_history.append(instantaneous_vel)
                    
                    # Use average of recent velocities for smoothing
                    if len(self.velocity_history) > 0:
                        self.joint_velocities = np.mean(self.velocity_history, axis=0)
            
            # If velocities are provided in the message, use them
            if len(msg.velocity) == len(msg.position):
                for joint_name, velocity in zip(msg.name, msg.velocity):
                    if joint_name in self.name_to_idx:
                        idx = self.name_to_idx[joint_name]
                        # Blend estimated and provided velocities
                        self.joint_velocities[idx] = 0.7 * self.joint_velocities[idx] + 0.3 * velocity
            
            # FIX: Convert from degrees/sec to radians/sec if values are too high
            # Check before clamping to detect if conversion is needed
            max_vel = np.max(np.abs(self.joint_velocities))
            if max_vel > 10:  # Values > 10 rad/s are unrealistic for Mini Pupper
                self.joint_velocities = np.deg2rad(self.joint_velocities)
                if self.step_count % 100 == 0:  # Log occasionally
                    self.get_logger().info(f'Converting joint velocities from deg/s to rad/s')
            
            # SAFETY: Clamp to reasonable range after conversion
            self.joint_velocities = np.clip(self.joint_velocities, -10.0, 10.0)
                
            self.prev_joint_positions = current_positions.copy()
            self.prev_joint_time = current_time
            
        except Exception as e:
            self.get_logger().error(f'Error in joint state callback: {e}')
            
        except Exception as e:
            self.get_logger().error(f'Error in joint state callback: {e}')

    def cmd_vel_callback(self, msg):
        """Update velocity commands with deadzone"""
        # Apply deadzone to reduce noise when standing still
        if abs(msg.linear.x) < self.cmd_vel_deadzone:
            self.velocity_commands[0] = 0.0
        else:
            self.velocity_commands[0] = np.clip(msg.linear.x, -1.0, 1.0)
        
        if abs(msg.linear.y) < self.cmd_vel_deadzone:
            self.velocity_commands[1] = 0.0
        else:
            self.velocity_commands[1] = np.clip(msg.linear.y, -0.5, 0.5)
        
        if abs(msg.angular.z) < self.cmd_vel_deadzone:
            self.velocity_commands[2] = 0.0
        else:
            self.velocity_commands[2] = np.clip(msg.angular.z, -1.0, 1.0)
        
        # Log only when commands change significantly
        if np.linalg.norm(self.velocity_commands) > 0.1:
            self.get_logger().info(f'Velocity command: {self.velocity_commands.round(2)}')
    
    # Add this debug code to your imu_callback:
    def imu_callback(self, msg):
        """Update gravity vector from IMU"""

        self.projected_gravity = np.array([0.0, 0.0, -1.0])

        # q = msg.orientation
        
        # # Your existing conversion
        # gx = 2 * (q.x * q.z - q.w * q.y)
        # gy = 2 * (q.y * q.z + q.w * q.x)
        # gz = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z
        
        # # Debug: Show raw IMU data
        # self.get_logger().info(f'IMU quaternion: [{q.w:.2f}, {q.x:.2f}, {q.y:.2f}, {q.z:.2f}]')
        # self.get_logger().info(f'Gravity vector: [{gx:.2f}, {gy:.2f}, {gz:.2f}]')
        
        # # Also check linear acceleration (might be more reliable)
        # if hasattr(msg, 'linear_acceleration'):
        #     acc_norm = np.sqrt(msg.linear_acceleration.x**2 + 
        #                     msg.linear_acceleration.y**2 + 
        #                     msg.linear_acceleration.z**2)
        #     if acc_norm > 0.1:
        #         gx_acc = msg.linear_acceleration.x / acc_norm
        #         gy_acc = msg.linear_acceleration.y / acc_norm
        #         gz_acc = msg.linear_acceleration.z / acc_norm
        #         self.get_logger().info(f'Gravity from accel: [{gx_acc:.2f}, {gy_acc:.2f}, {gz_acc:.2f}]')

        return

    def odom_callback(self, msg):
        """Update base velocities from odometry"""
        self.base_lin_vel[0] = msg.twist.twist.linear.x
        self.base_lin_vel[1] = msg.twist.twist.linear.y
        self.base_lin_vel[2] = msg.twist.twist.linear.z
        
        self.base_ang_vel[0] = msg.twist.twist.angular.x
        self.base_ang_vel[1] = msg.twist.twist.angular.y
        self.base_ang_vel[2] = msg.twist.twist.angular.z
    
    def get_observation(self):
        """Construct the 48-dimensional observation vector matching training"""
        
        # Build raw observation
        obs = np.concatenate([
            self.base_lin_vel,                              # base_lin_vel (3)
            self.base_ang_vel,                              # base_ang_vel (3) 
            self.projected_gravity,                         # projected_gravity (3)
            self.velocity_commands,                         # velocity_commands (3)
            self.joint_positions - self.default_positions,  # joint_pos relative to default (12)
            self.joint_velocities,                          # joint_vel (12)
            self.last_action                               # last_action (12)
        ])
        
        if self.step_count % 50 == 0:
            self.get_logger().info(f'RAW OBS vel_commands (9:12): {obs[9:12].round(2)}')
            self.get_logger().info(f'ACTUAL velocity_commands: {self.velocity_commands.round(2)}')
        
        # CRITICAL: Normalize observation using training statistics
        # CRITICAL: Normalize observation using training statistics
        if self.obs_mean is not None and self.obs_var is not None:
            eps = 1e-8
            obs_normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + eps)
            obs_normalized = np.clip(obs_normalized, -10.0, 10.0)
            
            # FIX: Don't normalize velocity commands - keep them raw!
            # The network expects raw commands, not normalized ones
            obs_normalized[9:12] = self.velocity_commands
            
            return obs_normalized.astype(np.float32)
        else:
            return obs.astype(np.float32)
    
    def control_loop(self):
        """Main control loop - runs at 50 Hz"""
        if self.model is None:
            return
        
        try:
            self.step_count += 1
            
            # OVERRIDE: Ensure clean sensor values
            # Since IMU is broken, force correct gravity
            self.projected_gravity = np.array([0.0, 0.0, -1.0])
            
            # Get normalized observation
            obs = self.get_observation()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Debug output showing corrected values
            if self.step_count % 50 == 0:  # Every second
                # Show normalized obs values (what the network sees)
                self.get_logger().info(
                    f'OBS (normalized): vel_cmd={obs[9:12].round(2)} | '
                    f'joint_vel_norm={np.linalg.norm(obs[24:36]):.2f} | '
                    f'gravity={obs[6:9].round(2)}'
                )
                # Also show raw sensor values for debugging
                self.get_logger().info(
                    f'RAW sensors: joint_vel_max={np.max(np.abs(self.joint_velocities)):.2f} rad/s | '
                    f'gravity={self.projected_gravity.round(2)}'
                )

            # Check if we should be standing still
            is_standing = np.allclose(self.velocity_commands, 0, atol=0.01)
            
            if is_standing:
                # STANDING MODE: Decay actions to zero when velocity command is zero
                self.filtered_action *= 0.95  # Exponential decay
                
                # If actions are very small, set to exactly zero
                if np.linalg.norm(self.filtered_action) < 0.01:
                    self.filtered_action = np.zeros(12)
                
                position_action = self.filtered_action
                
                # Debug output for standing mode
                if self.step_count % 50 == 0:  # Every second
                    with torch.no_grad():
                        raw_action = self.model(obs_tensor).cpu().numpy()[0]
                    self.get_logger().info(
                        f'STANDING MODE | filtered_action_norm={np.linalg.norm(self.filtered_action):.4f} | '
                        f'raw_network_output_max={np.abs(raw_action).max():.4f}'
                    )
            else:
                # WALKING MODE: Use network output normally
                with torch.no_grad():
                    raw_action = self.model(obs_tensor).cpu().numpy()[0]
                
                # Scale action
                scaled_action = raw_action * self.action_scale
                
                # Apply exponential smoothing filter for stability
                self.filtered_action = (self.filter_alpha * self.filtered_action + 
                                    (1 - self.filter_alpha) * scaled_action)
                
                position_action = self.filtered_action
            
            # Safety: limit rate of change (applies to both standing and walking)
            if len(self.action_history) > 0:
                action_diff = position_action - self.action_history[-1]
                action_diff = np.clip(action_diff, -self.max_joint_change, self.max_joint_change)
                position_action = self.action_history[-1] + action_diff
            
            # Store for next iteration
            self.action_history.append(position_action.copy())
            self.last_action = position_action.copy()
            
            # Apply to default positions (action is relative to default)
            target_positions = self.default_positions + position_action
            
            # Safety limits for Mini Pupper
            joint_limits_low = np.array([
                -0.5, 0.0, -2.36,  # LF leg limits
                -0.5, 0.0, -2.36,  # RF leg limits
                -0.5, 0.0, -2.36,  # LB leg limits
                -0.5, 0.0, -2.36   # RB leg limits
            ])
            
            joint_limits_high = np.array([
                0.5, 1.57, -0.7,   # LF leg limits
                0.5, 1.57, -0.7,   # RF leg limits
                0.5, 1.57, -0.7,   # LB leg limits
                0.5, 1.57, -0.7    # RB leg limits
            ])
            
            target_positions = np.clip(target_positions, joint_limits_low, joint_limits_high)
            
            # Create JointTrajectory message
            trajectory_msg = JointTrajectory()
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            trajectory_msg.joint_names = [self.joint_mapping[i] for i in range(12)]
            
            point = JointTrajectoryPoint()
            point.positions = target_positions.tolist()
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(1e9 / self.control_frequency)
            
            trajectory_msg.points = [point]
            
            self.joint_cmd_pub.publish(trajectory_msg)
            
            # Debug info (every second)
            if self.step_count % int(self.control_frequency) == 0:
                if is_standing:
                    self.get_logger().info(
                        f'STANDING: cmd_vel={self.velocity_commands.round(2)} | '
                        f'filtered_norm={np.linalg.norm(self.filtered_action):.4f} | '
                        f'joint_vel_norm={np.linalg.norm(self.joint_velocities):.3f}'
                    )
                else:
                    with torch.no_grad():
                        raw_action = self.model(obs_tensor).cpu().numpy()[0]
                    self.get_logger().info(
                        f'WALKING: cmd_vel={self.velocity_commands.round(2)} | '
                        f'raw_act[0:3]={raw_action[:3].round(3)} | '
                        f'joint_vel_norm={np.linalg.norm(self.joint_velocities):.3f}'
                    )
            
            # Sanity check
            if self.step_count > self.control_frequency:
                # Check gravity is correct
                if not np.allclose(self.projected_gravity, [0, 0, -1], atol=0.1):
                    self.get_logger().warn(f'WARNING: Gravity vector incorrect: {self.projected_gravity}')
                    self.projected_gravity = np.array([0.0, 0.0, -1.0])  # Force correct
                    
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    controller = MLPController()
    
    # Print startup info
    controller.get_logger().info('='*50)
    controller.get_logger().info('MLP Controller Started')
    controller.get_logger().info('TEST INPUT')
    controller.get_logger().info('Commands: ros2 topic pub /cmd_vel geometry_msgs/Twist')
    controller.get_logger().info('Stop: Ctrl+C')
    controller.get_logger().info('='*50)
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
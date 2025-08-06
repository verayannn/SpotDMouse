#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
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
        self.velocity_commands = np.array([0.8, 0.0, 0.0])  # Forward command
        self.last_action = np.zeros(12)
        
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
        
        # ROS2 publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory,
            '/joint_group_effort_controller/joint_trajectory', #this needs to be reconciled with "/joint_group_effort_controller/joint_trajectory"
            10
        )
        
        # Control timer (50 Hz to match training)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        # Safety limits
        self.max_joint_change = 0.05  # Max change per timestep (rad)
        self.torque_scale = 0.5      # Scale down actions for real servos
        
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
            checkpoint_path = "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/sim2real/walkingmlp.pt"  # Update this path
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
        """Update robot state from joint feedback"""
        try:
            # Map joint names to our internal representation
            for i, joint_name in enumerate(msg.name):
                if joint_name in [self.joint_mapping[j] for j in range(12)]:
                    # Find the index in our mapping
                    joint_idx = None
                    for idx, mapped_name in self.joint_mapping.items():
                        if mapped_name == joint_name:
                            joint_idx = idx
                            break
                    
                    if joint_idx is not None:
                        self.joint_positions[joint_idx] = msg.position[i]
                        if len(msg.velocity) > i:
                            self.joint_velocities[joint_idx] = msg.velocity[i]
                            
        except Exception as e:
            self.get_logger().error(f'Error in joint state callback: {e}')
    
    def cmd_vel_callback(self, msg):
        """Update velocity commands from teleop or high-level planner"""
        self.velocity_commands[0] = msg.linear.x
        self.velocity_commands[1] = msg.linear.y  
        self.velocity_commands[2] = msg.angular.z
        
        self.get_logger().info(f'Velocity command: {self.velocity_commands}')
    
    def estimate_base_velocity(self):
        """Estimate base velocity from joint states (simplified)"""
        # This is a placeholder - in reality you'd use IMU + odometry
        # For now, assume we're achieving the commanded velocity
        return self.velocity_commands[:3], np.array([0.0, 0.0, 0.0])
    
    # def get_observation(self):
    #     """Construct the 76-dimensional observation vector matching training"""
        
    #     # Get estimated base velocities (placeholder)
    #     lin_vel, ang_vel = self.estimate_base_velocity()
        
    #     # Construct observation (same order as training)
    #     obs = np.concatenate([
    #         lin_vel,                    # base_lin_vel (3)
    #         ang_vel,                    # base_ang_vel (3) 
    #         self.projected_gravity,     # projected_gravity (3)
    #         self.velocity_commands,     # velocity_commands (3)
    #         self.joint_positions,       # joint_pos (12)
    #         self.joint_velocities,      # joint_vel (12)
    #         self.last_action           # last_action (12)
    #     ])
        
    #     # Add noise (matching training config)
    #     noise_scales = np.array([0.1, 0.1, 0.1,     # lin_vel noise
    #                             0.1, 0.1, 0.1,      # ang_vel noise  
    #                             0.05, 0.05, 0.05,   # gravity noise
    #                             0.0, 0.0, 0.0,      # no noise on commands
    #                             *([0.05] * 12),     # joint pos noise
    #                             *([0.5] * 12),      # joint vel noise
    #                             *([0.0] * 12)])     # no noise on last action
        
    #     # Add small amount of noise for robustness
    #     obs += np.random.uniform(-0.01, 0.01, obs.shape) * noise_scales
        
    #     return obs.astype(np.float32)

    def get_observation(self):
        """Construct the 76-dimensional observation vector matching training"""
        
        # Get estimated base velocities
        lin_vel, ang_vel = self.estimate_base_velocity()
        
        # Build observation step by step
        obs_parts = []
        obs_parts.append(lin_vel)                    # Should be 3
        obs_parts.append(ang_vel)                    # Should be 3
        obs_parts.append(self.projected_gravity)     # Should be 3
        obs_parts.append(self.velocity_commands)     # Should be 3
        obs_parts.append(self.joint_positions)       # Should be 12
        obs_parts.append(self.joint_velocities)      # Should be 12
        obs_parts.append(self.last_action)           # Should be 12
        
        # Debug: print sizes
        current_size = sum(part.shape[0] for part in obs_parts)
        self.get_logger().info(f'Current obs size: {current_size}, need 76')
        
        # Pad with zeros to reach 76
        obs = np.concatenate(obs_parts)
        if obs.shape[0] < 76:
            padding = np.zeros(76 - obs.shape[0])
            obs = np.concatenate([obs, padding])
        
        return obs[:76].astype(np.float32)

    def control_loop(self):
        """Main control loop - runs at 50 Hz"""
        if self.model is None:
            return
            
        try:
            # Get observation
            obs = self.get_observation()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from MLP
            with torch.no_grad():
                raw_action = self.model(obs_tensor).cpu().numpy()[0]
            
            # Scale action by learned std and apply safety scaling
            position_action = raw_action * self.action_std * 0.5  # Scale down for safety
            
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
            
            # Create proper JointTrajectory message
            trajectory_msg = JointTrajectory()
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Set joint names in the correct order
            trajectory_msg.joint_names = [self.joint_mapping[i] for i in range(12)]
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = target_positions.tolist()
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = 20000000  # 20ms
            
            trajectory_msg.points = [point]
            
            self.joint_cmd_pub.publish(trajectory_msg)
            
            # Debug info (every ~1 second)
            if self.get_clock().now().nanoseconds % 1000000000 < 20000000:
                self.get_logger().info(f'Raw action: {raw_action[:4].round(3)}...')
                self.get_logger().info(f'Target pos: {target_positions[:4].round(3)}...')
                
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
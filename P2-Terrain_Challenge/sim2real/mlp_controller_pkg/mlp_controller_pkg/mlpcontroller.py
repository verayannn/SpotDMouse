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
import os
import sys
import torch.nn as nn

DEVICE = torch.device("cpu")

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(48, 512, bias=True),
                nn.ELU(),
                nn.Linear(512, 256, bias=True),
                nn.ELU(),
                nn.Linear(256, 128, bias=True),
                nn.ELU(),
                nn.Linear(128, 12, bias=True),
                )
        self.critic = nn.Sequential(
                nn.Linear(48, 512, bias=True),
                nn.ELU(),
                nn.Linear(512, 256, bias=True),
                nn.ELU(),
                nn.Linear(256, 128, bias=True),
                nn.ELU(),
                nn.Linear(128, 1, bias=True),
                )
    def forward(self, x):
        actor = self.actor(x)
        return  actor

class MLPController(Node):
    def __init__(self):
        super().__init__('mlp_controller')
        
        self.device = DEVICE
        self.model, self.action_std = self._load_model() 
        
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.array([0.0, 0.0, -9.81]) 
        self.velocity_commands = np.zeros(3) 
        self.last_action = np.zeros(12) 
        self.initialized = False
        
        self.filtered_action = np.zeros(12)
        
        self.prev_joint_positions = None
        self.prev_joint_time = None
        
        self.default_positions = np.array([
            0.0, 0.785, -1.57, 
            0.0, 0.785, -1.57,
            0.0, 0.785, -1.57, 
            0.0, 0.785, -1.57
        ])
        
        # Joint mapping (Order MUST match model's observation/action space)
        self.joint_mapping = {
            0: 'base_lf1', 1: 'lf1_lf2', 2: 'lf2_lf3',
            3: 'base_rf1', 4: 'rf1_rf2', 5: 'rf2_rf3',
            6: 'base_lb1', 7: 'lb1_lb2', 8: 'lb2_lb3',
            9: 'base_rb1', 10: 'rb1_rb2', 11: 'rb2_rb3'
        }
        self.name_to_idx = {name: idx for idx, name in self.joint_mapping.items()}
        
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joint_cmd_pub = self.create_publisher(JointTrajectory, '/joint_group_effort_controller/joint_trajectory', 10)
        
        self.control_frequency = 50.0  # Hz
        self.control_timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)
        
        self.action_scale = 7.0      
        self.filter_alpha = 0.5       
        self.cmd_vel_deadzone = 0.15
        
        self.joint_limits_low = np.array([-0.3, -0.2, -2.36] * 4)
        self.joint_limits_high = np.array([0.3, 1.2, -0.5] * 4)
        
        self.get_logger().info('MLP Controller initialized. ⚠️ NO OBSERVATION NORMALIZATION APPLIED.')
        
    def _load_model(self):
        """Load the trained MLP model and action standard deviation."""
        try:
            model = ActorCritic()
            checkpoint_path = "/home/ubuntu/rsl_rl_trainedmodels/45degree_mlp.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Extract only the actor weights
            state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('actor.'):
                    new_key = key.replace('actor.', '')
                    state_dict[new_key] = value
            
            model.actor.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)
            
            action_std = checkpoint['model_state_dict'].get('std', torch.ones(12)).cpu().numpy()
            
            self.get_logger().info(f'Model loaded. Action STD range: [{action_std.min():.3f}, {action_std.max():.3f}]')
            return model, action_std
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None, np.ones(12)

    def joint_state_callback(self, msg):
        current_time = time.time()
        current_positions = np.zeros(12)
        
        for joint_name, position in zip(msg.name, msg.position):
            if joint_name in self.name_to_idx:
                idx = self.name_to_idx[joint_name]
                current_positions[idx] = position
        
        self.joint_positions = current_positions

        if not self.initialized and np.any(self.joint_positions != 0):
            self.last_action = self.joint_positions - self.default_positions
            self.filtered_action = self.last_action.copy()
            self.initialized = True
            self.get_logger().info('Initialized action state to current joint positions.')
        
        if self.prev_joint_positions is not None and self.prev_joint_time is not None:
            dt = current_time - self.prev_joint_time
            if dt > 0.001 and dt < 0.1: 
                instantaneous_vel = (current_positions - self.prev_joint_positions) / dt
                alpha = 0.2
                self.joint_velocities = alpha * instantaneous_vel + (1 - alpha) * self.joint_velocities
                self.joint_velocities = np.clip(self.joint_velocities, -10.0, 10.0)
        
        self.prev_joint_positions = current_positions.copy()
        self.prev_joint_time = current_time      
     
    def cmd_vel_callback(self, msg):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        self.velocity_commands[0] = np.clip(vx, -0.8, 3.5) if abs(vx) >= self.cmd_vel_deadzone else 0.0
        self.velocity_commands[1] = np.clip(vy, -0.3, 0.3) if abs(vy) >= self.cmd_vel_deadzone else 0.0
        self.velocity_commands[2] = np.clip(wz, -1.0, 1.0) if abs(wz) >= self.cmd_vel_deadzone else 0.0
    
    def imu_callback(self, msg):
        pass

    def odom_callback(self, msg):
        self.base_lin_vel[0] = msg.twist.twist.linear.x
        self.base_lin_vel[1] = msg.twist.twist.linear.y
        self.base_lin_vel[2] = msg.twist.twist.linear.z
        
        self.base_ang_vel[0] = msg.twist.twist.angular.x
        self.base_ang_vel[1] = msg.twist.twist.angular.y
        self.base_ang_vel[2] = msg.twist.twist.angular.z

    def get_observation(self):
        
        raw_obs = np.concatenate([
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity, 
            self.velocity_commands, 
            self.joint_positions - self.default_positions, 
            self.joint_velocities,
            self.last_action 
        ])
        
        return raw_obs.astype(np.float32)

    def control_loop(self):
        """Main control loop - runs at 50 Hz"""
        if self.model is None or not self.initialized:
            return
        
        try:
            self.projected_gravity = np.array([0.0, 0.0, -9.81])
            
            obs = self.get_observation()

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            is_standing = np.allclose(self.velocity_commands, 0, atol=0.01)
            
            with torch.no_grad():
                raw_action = self.model(obs_tensor).cpu().numpy()[0]
            

            action_unscaled = raw_action * self.action_std
            
            scaled_action = action_unscaled * self.action_scale

            if is_standing:
                decay_factor = 0.9
                self.filtered_action *= decay_factor
                if np.linalg.norm(self.filtered_action) < 1e-3:
                    self.filtered_action = np.zeros(12)
            else:
                alpha = self.filter_alpha
                self.filtered_action = alpha * scaled_action + (1 - alpha) * self.filtered_action
            
            position_action = self.filtered_action

            self.last_action = position_action.copy()
            
            target_positions = self.default_positions + position_action
            target_positions = np.clip(target_positions, self.joint_limits_low, self.joint_limits_high)
            
            trajectory_msg = JointTrajectory()
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            trajectory_msg.joint_names = [self.joint_mapping[i] for i in range(12)]
            
            point = JointTrajectoryPoint()
            point.positions = target_positions.tolist()
            point.time_from_start.sec = 0
            point.time_from_start.nanosec = int(1e9 / self.control_frequency)
            
            trajectory_msg.points = [point]
            self.joint_cmd_pub.publish(trajectory_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

            
def main(args=None):
    rclpy.init(args=args)
    controller = MLPController()
    
    controller.get_logger().info('='*50)
    controller.get_logger().info('MLP Controller Refactored and Started (No Obs Norm)')
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
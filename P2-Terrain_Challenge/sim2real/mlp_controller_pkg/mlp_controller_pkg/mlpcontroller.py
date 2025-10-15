#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
import torch.nn as nn
import os
import sys

# --- Configuration Constants ---
# MODEL_PATH = "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"
MODEL_PATH = "/home/ubuntu/rsl_rl_trainedmodels/45degree_mlp.pt"
# MODEL_PATH = "/home/ubuntu/rsl_rl_trainedmodels/30degree_mlp.pt"
DEVICE = torch.device("cpu")
# CRITICAL: Assuming static projected gravity based on training simplification
STATIC_PROJECTED_GRAVITY = np.array([0.0, 0.0, -9.81]) 

# --- Default Pose --- 45 degree
default_pose_values = np.array([
    0.0, 0.785, -1.57,  # LF
    0.0, 0.785, -1.57,  # RF
    0.0, 0.785, -1.57,  # LB
    0.0, 0.785, -1.57,  # RB
])

# --- Default Pose --- 30 degree
# default_pose_values = np.array([
#     0.0, 0.52, -1.05,
#     0.0, 0.52, -1.05,
#     0.0, 0.52, -1.05,
#     0.0, 0.52, -1.05,
# ])

# Joint name list (for publishing order)
joint_names = [
    "base_lf1", "lf1_lf2", "lf2_lf3",
    "base_rf1", "rf1_rf2", "rf2_rf3",
    "base_lb1", "lb1_lb2", "lb2_lb3",
    "base_rb1", "rb1_rb2", "rb2_rb3"
]
joint_name_to_idx = {name: i for i, name in enumerate(joint_names)}

# --- Model Definition (No Change) ---
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(48, 512, bias=True), nn.ELU(),
                nn.Linear(512, 256, bias=True), nn.ELU(),
                nn.Linear(256, 128, bias=True), nn.ELU(),
                nn.Linear(128, 12, bias=True),
                )
        self.critic = nn.Sequential(
                nn.Linear(48, 512, bias=True), nn.ELU(),
                nn.Linear(512, 256, bias=True), nn.ELU(),
                nn.Linear(256, 128, bias=True), nn.ELU(),
                nn.Linear(128, 1, bias=True),
                )
    def forward(self, x):
        actor = self.actor(x)
        return actor

# --- MLP Controller Class ---
class MLPController(Node):
    def __init__(self):
        super().__init__('mlp_controller')
        
        # --- Initialization & Model Loading ---
        self.model = ActorCritic()
        self.model.to(DEVICE).eval()
        self.default_positions = default_pose_values
        self.load_model_and_params()

        # --- State Variables ---
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.last_actions = np.zeros(12)
        self.smoothed_actions = np.zeros(12)
        
        self.cmd_vel_linear = np.zeros(3) # [Vx, Vy, Vz]
        self.cmd_vel_angular = np.zeros(3) # [Wx, Wy, Wz] (Only Wz is used)
        self.base_lin_vel = np.zeros(3) # From Odometry
        self.base_ang_vel = np.zeros(3) # From Odometry
        
        # Static projected gravity (no IMU rotation needed)
        self.projected_gravity = STATIC_PROJECTED_GRAVITY.copy()

        # --- Control & Action Parameters ---
        self.control_frequency = 50.0 
        self.dt = 1.0 / self.control_frequency
        self.action_smoothing = 0.75 #.025
        self.cmd_timeout = 0.50
        self.last_cmd_time = self.get_clock().now()
        self.has_received_cmd = False

        self.joint_data_received = False 
        
        # --- **JOINT-SPECIFIC ACTION SCALING** ---
        # THIGH_CALF_SCALE = 1.0#0.4 
        HIP_SCALE = 1.0 #0.35#1.0#2.0
        THIGH_SCALE = 1.0 #1.5#1.8
        CALF_SCALE = 1.0 #2.0#1.0

        # self.get_logger().info(f"thigh scales: {THIGH_CALF_SCALE}, hip scales:{HIP_SCALE}")
        
        # # Scaling array for the 12 joints: [H, T, C, H, T, C, ...]
        # self.action_scale_vector = np.array([
        #     HIP_SCALE, THIGH_CALF_SCALE, THIGH_CALF_SCALE,
        #     HIP_SCALE, THIGH_CALF_SCALE, THIGH_CALF_SCALE,
        #     HIP_SCALE, THIGH_CALF_SCALE, THIGH_CALF_SCALE,
        #     HIP_SCALE, THIGH_CALF_SCALE, THIGH_CALF_SCALE,
        # ])

        self.action_scale_vector = np.array([
            HIP_SCALE, THIGH_SCALE, CALF_SCALE,
            HIP_SCALE, THIGH_SCALE, CALF_SCALE,
            HIP_SCALE, THIGH_SCALE, CALF_SCALE,
            HIP_SCALE, THIGH_SCALE, CALF_SCALE,
        ])

        self.get_logger().info(f"hip scales: {HIP_SCALE}, thigh scale: {THIGH_SCALE}, calf scale: {CALF_SCALE}")
        # --- Initialization ---
        self.initialized = False
        self.init_duration = 15.0 
        self.init_start_time = None  # Add this missing attribute
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        # --- Subscribers & Publishers ---
        self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10) 
        # IMU subscription REMOVED
        
        self.joint_pub = self.create_publisher(JointTrajectory, 'joint_group_effort_controller/joint_trajectory', 10)
        self.raw_mlp_output_pub = self.create_publisher(Float32MultiArray, 'mlp_controller/raw_output', 10)
        self.mlp_observation_pub = self.create_publisher(Float32MultiArray, 'mlp_controller/observation', 10)
        
        self.get_logger().info("MLP Controller initialized (Static Gravity Mode)!")

    def load_model_and_params(self):
        """Loads model weights and normalization parameters."""
        # (Loading logic is unchanged)
        if not os.path.exists(MODEL_PATH):
            self.get_logger().error(f"Model file not found at {MODEL_PATH}")
            sys.exit(1)
            
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('actor_critic') or checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            self.get_logger().info("Model loaded successfully!")
            
            if 'obs_rms_mean' in checkpoint and 'obs_rms_var' in checkpoint:
                self.obs_rms_mean = checkpoint['obs_rms_mean'].to(DEVICE)
                self.obs_rms_var = checkpoint['obs_rms_var'].to(DEVICE)
                self.use_obs_normalization = True
            else:
                self.use_obs_normalization = False
                
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

    def joint_state_callback(self, msg):
        """Update joint positions and velocities from sensor data."""
        for i, name in enumerate(msg.name):
            if name in joint_name_to_idx:
                idx = joint_name_to_idx[name]
                self.joint_positions[idx] = msg.position[i]
                if len(msg.velocity) > i:
                    self.joint_velocities[idx] = msg.velocity[i]
        if not self.joint_data_received:
            self.joint_data_received = True

    def cmd_vel_callback(self, msg):
        """Update command velocities from user input."""
        self.cmd_vel_linear[0] = msg.linear.x
        self.cmd_vel_linear[1] = msg.linear.y
        self.cmd_vel_angular[2] = msg.angular.z
        self.last_cmd_time = self.get_clock().now()
        
        if np.any(np.abs(self.cmd_vel_linear[:2]) > 0.01) or np.abs(self.cmd_vel_angular[2]) > 0.01:
            if not self.has_received_cmd:
                self.has_received_cmd = True

    def odom_callback(self, msg):
        """Update base linear and angular velocities from Odometry."""
        # Use only the linear/angular components required for the 48-dim vector
        self.base_lin_vel[0] = msg.twist.twist.linear.x
        self.base_lin_vel[1] = msg.twist.twist.linear.y
        self.base_lin_vel[2] = msg.twist.twist.linear.z
        
        self.base_ang_vel[0] = msg.twist.twist.angular.x
        self.base_ang_vel[1] = msg.twist.twist.angular.y
        self.base_ang_vel[2] = msg.twist.twist.angular.z

    # IMU callback is REMOVED
    
    def construct_observation(self):
        """
        CRITICAL: Construct 48-dimensional observation vector in the IL Training Order.
        IL Order: [Cmd(3) | Q_rel(12) | dQ(12) | Action_prev(12) | Grav(3) | LinVel(3) | AngVel(3)]
        RSL Order: [LinVel(3), AngVel(3), Grav(3), Cmd(3), Q_rel(12), dQ(12), Action_prev(12)]
        """
        
        # 1. Command Velocities (3): [Vx_cmd, Vy_cmd, Wz_cmd]
        cmd_vels = np.array([self.cmd_vel_linear[0], self.cmd_vel_linear[1], self.cmd_vel_angular[2]])
        
        # 2. Joint Positions (Relative) (12)
        joint_pos_rel = self.default_positions - self.joint_positions #self.joint_positions - self.default_positions
        
        # 3. Joint Velocities (12)
        joint_vels = self.joint_velocities
        
        # 4. Last Action (12)
        last_actions = self.last_actions
        
        # 5. Projected Gravity (3) - STATIC
        proj_gravity = self.projected_gravity # STATIC_PROJECTED_GRAVITY
        
        # 6. Base Linear Velocity (3)
        base_lin_vel = self.base_lin_vel
        
        # 7. Base Angular Velocity (3)
        base_ang_vel = self.base_ang_vel

        # raw_obs = np.concatenate([
        #     cmd_vels,
        #     joint_pos_rel,
        #     joint_vels,
        #     last_actions,
        #     proj_gravity, # Indices 39-41
        #     base_lin_vel, # Indices 42-44
        #     base_ang_vel  # Indices 45-47
        # ])

        # Assemble into the RSL-RL vector
        rsl_obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            proj_gravity,
            cmd_vels,
            joint_pos_rel,
            joint_vels,
            last_actions
        ], axis=-1)

        # rsl_obs = np.concatenate([
        #     base_lin_vel,
        #     base_ang_vel,
        #     proj_gravity,
        #     cmd_vels,
        #     joint_pos_rel,
        #     joint_vels,
        #     last_actions
        # ])

        # return raw_obs.astype(np.float32)
        return rsl_obs.astype(np.float32)

    # --- Control Loop and Utilities (Unchanged) ---
    def is_command_active(self):
        # (Logic is unchanged)
        if self.last_cmd_time is None: return False
        time_since_cmd = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        if time_since_cmd > self.cmd_timeout: return False
        if np.all(np.abs(self.cmd_vel_linear[:2]) < 0.01) and np.abs(self.cmd_vel_angular[2]) < 0.01:
            return False
        return True

    def move_to_default_pose(self):
        # (Logic is unchanged)
        if self.init_start_time is None:
            self.init_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.init_positions = self.joint_positions.copy()
        
        current_time = self.get_clock().now().nanoseconds * 1e-9
        elapsed = current_time - self.init_start_time
        
        if elapsed >= self.init_duration:
            self.initialized = True
            self.get_logger().info("Default stance reached. Starting MLP control...")
            return True
        
        alpha = min(elapsed / self.init_duration, 1.0)
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)

        target_positions = self.init_positions * (1 - alpha) + self.default_positions * alpha
        
        segment_time = rclpy.duration.Duration(seconds=self.dt).to_msg()
        target_velocities = np.zeros(12).tolist()

        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        
        point = JointTrajectoryPoint(positions=target_positions.tolist(), 
                                     velocities=target_velocities, # Added velocities
                                     time_from_start=segment_time)
        
        traj_msg.points = [point]
        self.joint_pub.publish(traj_msg)
        return False

    # def move_to_default_pose(self):
    #     # (Logic is unchanged)
    #     if self.init_start_time is None:
    #         self.init_start_time = self.get_clock().now().nanoseconds * 1e-9
    #         self.init_positions = self.joint_positions.copy()
        
    #     current_time = self.get_clock().now().nanoseconds * 1e-9
    #     elapsed = current_time - self.init_start_time
        
    #     if elapsed >= self.init_duration:
    #         self.initialized = True
    #         self.get_logger().info("Default stance reached. Starting MLP control...")
    #         return True
        
    #     alpha = min(elapsed / self.init_duration, 1.0)
    #     alpha = 0.5 - 0.5 * np.cos(np.pi * alpha) 
        
    #     target_positions = self.init_positions * (1 - alpha) + self.default_positions * alpha
        
    #     traj_msg = JointTrajectory()
    #     traj_msg.joint_names = joint_names
        
    #     point = JointTrajectoryPoint(positions=target_positions.tolist(), 
    #                                  time_from_start=rclpy.duration.Duration(seconds=self.dt).to_msg())
        
    #     traj_msg.points = [point]
    #     self.joint_pub.publish(traj_msg)
        
    #     return False


    def control_loop(self):

        if not self.joint_data_received:
            self.get_logger().warn("Waiting for initial joint_states...")
            return
        
        if not self.initialized:
            self.move_to_default_pose()
            return

        if not self.is_command_active():
            # Maintain default stance logic
            if self.has_received_cmd:
                self.has_received_cmd = False
                self.last_actions = np.zeros(12)
                self.smoothed_actions = np.zeros(12)
            
            traj_msg = JointTrajectory()
            traj_msg.joint_names = joint_names
            point = JointTrajectoryPoint(positions=self.default_positions.tolist(), 
                                         time_from_start=rclpy.duration.Duration(seconds=self.dt).to_msg())
            traj_msg.points = [point]
            self.joint_pub.publish(traj_msg)
            return
        
        # --- MLP Control Logic ---
        obs = self.construct_observation()
        
        obs_msg = Float32MultiArray(data=obs.tolist())
        self.mlp_observation_pub.publish(obs_msg)
        
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
        
        if self.use_obs_normalization:
            obs_mean = self.obs_rms_mean
            obs_std = torch.sqrt(self.obs_rms_var + 1e-8)
            obs_tensor = (obs_tensor - obs_mean) / obs_std
        
        with torch.no_grad():
            raw_action = self.model(obs_tensor).squeeze(0).cpu().numpy()
        
        # Debug prints instead of IPython
        # self.get_logger().info(f"Raw action range: [{raw_action.min():.4f}, {raw_action.max():.4f}]")
        # self.get_logger().info(f"Raw action mean: {raw_action.mean():.4f}, std: {raw_action.std():.4f}")
        # self.get_logger().info(f"Raw action: {raw_action}")
        # self.get_logger().info(f"Raw action shape: {raw_action.shape}")
        
        raw_output_msg = Float32MultiArray(data=raw_action.tolist())
        self.raw_mlp_output_pub.publish(raw_output_msg)
        
        # --- **CRITICAL: JOINT-SPECIFIC SCALING** ---
        # Note: Scaling is division (raw_action / scale_factor)
        # scaled_action = raw_action / self.action_scale_vector
        scaled_action = raw_action * self.action_scale_vector

        # self.get_logger().info(f"RBH SCALED ACTION (Index 9): {scaled_action[9]:.4f}") #######
        
        self.smoothed_actions = (self.action_smoothing * self.smoothed_actions + 
                                (1 - self.action_smoothing) * scaled_action)
        
        action = np.clip(self.smoothed_actions, -1.5, 1.5)
        self.last_actions = action.copy()
        
        target_positions = self.default_positions + action
        
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        
        point = JointTrajectoryPoint(positions=target_positions.tolist(), 
                                     time_from_start=rclpy.duration.Duration(seconds=self.dt).to_msg())
        
        traj_msg.points = [point]
        self.joint_pub.publish(traj_msg)


def main(args=None):
    rclpy.init(args=args)
    controller = MLPController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down MLP Controller...")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

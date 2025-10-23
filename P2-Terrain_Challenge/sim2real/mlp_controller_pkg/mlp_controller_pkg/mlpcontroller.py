#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray, Float64MultiArray
from std_srvs.srv import Trigger
import torch
import numpy as np
import torch.nn as nn
import os
import sys
import json
from threading import Lock

# Import MangDang calibration modules
try:
    from MangDang.mini_pupper.ServoCalibration import MICROS_PER_RAD
    import MangDang.mini_pupper.nvram as nvram
    NVRAM_AVAILABLE = True
except ImportError:
    NVRAM_AVAILABLE = False
    print("Warning: MangDang modules not available, using fallback calibration")

# --- Configuration Constants ---
# MODEL_PATH = "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"
MODEL_PATH = "/home/ubuntu/rsl_rl_trainedmodels/002footclearance45degree_mlp.pt"
DEVICE = torch.device("cpu")

# Joint name list (maintains consistent ordering)
JOINT_NAMES = [
    "base_lf1", "lf1_lf2", "lf2_lf3",
    "base_rf1", "rf1_rf2", "rf2_rf3",
    "base_lb1", "lb1_lb2", "lb2_lb3",
    "base_rb1", "rb1_rb2", "rb2_rb3"
]
JOINT_NAME_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}

# --- Model Definition ---
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
        return self.actor(x)

class UnifiedMLPController(Node):
    def __init__(self):
        super().__init__('unified_mlp_controller')
        
        # --- Threading Safety ---
        self.state_lock = Lock()
        
        # --- Model Loading ---
        self.model = ActorCritic()
        self.model.to(DEVICE).eval()
        self.load_model_and_params()
        
        # --- NVRAM Calibration System ---
        self.calibrated_default_positions = None
        self.calibration_loaded = False
        self.nvram_calibration_data = None
        self.servo_calibration_offsets = None
        
        # Nominal default positions (45-degree stance)
        self.nominal_default_positions = np.array([
            0.0, 0.785, -1.57,  # LF
            0.0, 0.785, -1.57,  # RF
            0.0, 0.785, -1.57,  # LB
            0.0, 0.785, -1.57,  # RB
        ])

        # self.nominal_default_positions = np.array([
        #     0.0, 0.52, -1.05,  # LF
        #     0.0, 0.52, -1.05,  # RF
        #     0.0, 0.52, -1.05,  # LB
        #     0.0, 0.52, -1.05,  # RB
        # ])
        
        # --- State Variables ---
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.joint_efforts = np.zeros(12)
        self.last_actions = np.zeros(12)
        self.smoothed_actions = np.zeros(12)  # For EMA smoothing
        self.action_buffer = []  # For interpolation at higher frequency
        self.action_smoothing = 0.65  # EMA coefficient (0.65 = 65% previous, 35% new)
        
        self.cmd_vel_linear = np.zeros(3)
        self.cmd_vel_angular = np.zeros(3)
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.projected_gravity = np.array([0.0, 0.0, -9.81])
        
        # --- Control Parameters ---
        self.mlp_frequency = 50.0  # MLP inference rate
        self.control_frequency = 250.0  # Match joint controller rate
        self.mlp_dt = 1.0 / self.mlp_frequency
        self.control_dt = 1.0 / self.control_frequency
        self.interpolation_steps = int(self.control_frequency / self.mlp_frequency)
        

        #toggle off global gains in lieu of local gains
        # arbritrary = 0.55
        # --- Action Scaling (Tunable per joint type) ---
        # self.action_scale = {
        #     'hip': 0.15,
        #     'thigh': 1.20 + arbritrary,
        #     'calf': 1.20 + arbritrary
        # }

        self.build_action_scale_vector()
        
        # --- Control Modes ---
        self.use_effort_control = True  # Set True to use torque control
        self.use_hybrid_control = False  # Combine position + feedforward effort
        self.use_interpolation = False  # Use high-freq interpolation (set False for simple direct control)
        
        # --- PID Gains (matching the yaml configuration) ---
        self.kp = {'hip': 10.0, 'thigh': 10.0, 'calf': 10.0}
        self.kd = {'hip': 0.006, 'thigh': 0.006, 'calf': 0.006}
        self.build_gain_vectors()
        
        # --- Command Handling ---
        self.cmd_timeout = 0.5
        self.last_cmd_time = self.get_clock().now()
        self.has_received_cmd = False
        self.joint_data_received = False
        
        # --- Initialization Sequence ---
        self.initialized = False
        self.init_duration = 2.0
        self.init_start_time = None
        
        # --- Publishers and Subscribers ---
        self.setup_ros_interfaces()
        
        # --- Load NVRAM Calibration FIRST ---
        self.load_nvram_calibration()
        
        # --- Timers ---
        self.mlp_timer = self.create_timer(self.mlp_dt, self.mlp_inference_loop)
        self.control_timer = self.create_timer(self.control_dt, self.high_freq_control_loop)
        
        self.get_logger().info("Unified MLP Controller initialized!")
        self.get_logger().info(f"MLP Rate: {self.mlp_frequency}Hz, Control Rate: {self.control_frequency}Hz")
        self.get_logger().info(f"Control Mode: {'Interpolation' if self.use_interpolation else 'Direct'}")
        # self.get_logger().info(f"Smoothing: {self.action_smoothing}, Action scales: H={self.action_scale['hip']}, T={self.action_scale['thigh']}, C={self.action_scale['calf']}")
        self.get_logger().info(f"NVRAM Available: {NVRAM_AVAILABLE}")
        if self.calibration_loaded:
            self.get_logger().info(f"Calibration loaded successfully from NVRAM")

    def setup_ros_interfaces(self):
        """Setup all ROS publishers and subscribers"""
        # Subscribers
        self.sub_joint_states = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.sub_cmd_vel = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.sub_odom = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        
        # Publishers for different control modes
        self.pub_joint_trajectory = self.create_publisher(
            JointTrajectory, 'joint_group_effort_controller/joint_trajectory', 10)
        self.pub_effort_commands = self.create_publisher(
            Float64MultiArray, 'joint_group_effort_controller/commands', 10)
        
        # Debug publishers
        self.pub_mlp_observation = self.create_publisher(
            Float32MultiArray, 'mlp_controller/observation', 10)
        self.pub_mlp_output = self.create_publisher(
            Float32MultiArray, 'mlp_controller/raw_output', 10)
        self.pub_calibration_status = self.create_publisher(
            Float32MultiArray, 'mlp_controller/calibration', 10)
        
        # Service for recalibration
        self.srv_recalibrate = self.create_service(
            Trigger, 'mlp_controller/recalibrate', self.recalibrate_service)

    def build_action_scale_vector(self):
        """Build joint-specific action scaling vector"""
        # self.action_scale_vector = np.array([
        #     self.action_scale['hip'], self.action_scale['thigh'], self.action_scale['calf'],
        #     self.action_scale['hip'], self.action_scale['thigh'], self.action_scale['calf'],
        #     self.action_scale['hip'], self.action_scale['thigh'], self.action_scale['calf'],
        #     self.action_scale['hip'], self.action_scale['thigh'], self.action_scale['calf'],
        # ])

        ###################################
        ###################################
        ###################################

        arbritrary = 1.0
        HIP_BASE = 0.15
        THIGH_BASE = 1.20 + arbritrary  
        CALF_BASE = 1.20 + arbritrary

        lf_arbritrary = 6.5    
        LF_THIGH_BOOST = -1.20 - 2.0
        LF_CALF_BOOST = (1.20 + lf_arbritrary)
        
        lb_arbritrary = 1.5
        LB_THIGH_BOOST = -1.2#0.8   
        LB_CALF_BOOST = -1 * (1.20 + lb_arbritrary)#0.8

        self.action_scale_vector = np.array([
            HIP_BASE, LF_THIGH_BOOST, LF_CALF_BOOST,  # LF
            HIP_BASE, THIGH_BASE, CALF_BASE,  # RF
            HIP_BASE, LB_THIGH_BOOST, LB_CALF_BOOST, # LB
            HIP_BASE, THIGH_BASE, CALF_BASE,  # RB
        ])

        ###################################
        ###################################
        ###################################

    def build_gain_vectors(self):
        """Build joint-specific PID gain vectors"""
        self.kp_vector = np.array([
            self.kp['hip'], self.kp['thigh'], self.kp['calf'],
            self.kp['hip'], self.kp['thigh'], self.kp['calf'],
            self.kp['hip'], self.kp['thigh'], self.kp['calf'],
            self.kp['hip'], self.kp['thigh'], self.kp['calf'],
        ])
        self.kd_vector = np.array([
            self.kd['hip'], self.kd['thigh'], self.kd['calf'],
            self.kd['hip'], self.kd['thigh'], self.kd['calf'],
            self.kd['hip'], self.kd['thigh'], self.kd['calf'],
            self.kd['hip'], self.kd['thigh'], self.kd['calf'],
        ])

    def load_model_and_params(self):
        """Load model weights and normalization parameters"""
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
                self.get_logger().info("Loaded observation normalization parameters")
            else:
                self.use_obs_normalization = False
                
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            sys.exit(1)

    def load_nvram_calibration(self):
        """Load calibration data from NVRAM (same as preprogrammed gait)"""
        if not NVRAM_AVAILABLE:
            self.get_logger().warn("NVRAM not available, using nominal positions")
            self.calibrated_default_positions = self.nominal_default_positions.copy()
            self.calibration_loaded = True
            return
        
        try:
            # Read NVRAM data exactly as the preprogrammed gait does
            nvram_data = nvram.read()
            
            # Get the calibration values
            self.nvram_calibration_data = nvram_data
            micros_per_rad = nvram_data.get('MICROS_PER_RAD', MICROS_PER_RAD)
            neutral_angles_deg = nvram_data.get('NEUTRAL_ANGLE_DEGREES', [])
            
            self.get_logger().info(f"NVRAM MICROS_PER_RAD: {micros_per_rad}")
            self.get_logger().info(f"NVRAM neutral angles (degrees): {neutral_angles_deg}")
            
            # Convert the 3x4 array to our 12-joint format
            # NVRAM format is [leg0, leg1, leg2, leg3] where each leg has [hip, thigh, calf, reserved]
            # Our format is [LF_hip, LF_thigh, LF_calf, RF_hip, RF_thigh, RF_calf, ...]
            
            # Map NVRAM leg indices to our leg order
            # NVRAM: leg0=LF, leg1=RF, leg2=LB, leg3=RB (this may need adjustment)
            nvram_array = np.array(neutral_angles_deg)
            
            # Extract calibration offsets for each joint (ignoring the 4th column which is reserved)
            calibration_offsets_deg = np.zeros(12)
            
            # The mapping depends on how your URDF names map to the servo hardware
            # This is a common mapping but may need adjustment:
            leg_mapping = {
                0: 0,  # NVRAM leg0 -> LF (indices 0,1,2)
                1: 3,  # NVRAM leg1 -> RF (indices 3,4,5)
                2: 6,  # NVRAM leg2 -> LB (indices 6,7,8)
                3: 9,  # NVRAM leg3 -> RB (indices 9,10,11)
            }
            
            for nvram_leg_idx, our_base_idx in leg_mapping.items():
                if nvram_leg_idx < len(nvram_array):
                    # Take first 3 values (hip, thigh, calf) from each NVRAM leg
                    for j in range(3):
                        if j < len(nvram_array[nvram_leg_idx]):
                            calibration_offsets_deg[our_base_idx + j] = nvram_array[nvram_leg_idx][j]
            
            # Convert to radians
            calibration_offsets_rad = np.deg2rad(calibration_offsets_deg)
            
            # Apply calibration offsets to nominal positions
            # The sign and application method may need adjustment based on your servo convention
            self.calibrated_default_positions = self.nominal_default_positions #+ calibration_offsets_rad
            self.servo_calibration_offsets = calibration_offsets_rad
            
            self.calibration_loaded = True
            
            self.get_logger().info(f"Calibration offsets (deg): {calibration_offsets_deg}")
            self.get_logger().info(f"Calibration offsets (rad): {calibration_offsets_rad}")
            self.get_logger().info(f"Calibrated default positions: {self.calibrated_default_positions}")
            
            # Publish calibration status
            calib_msg = Float32MultiArray(data=self.calibrated_default_positions.tolist())
            self.pub_calibration_status.publish(calib_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to load NVRAM calibration: {e}")
            self.get_logger().warn("Falling back to nominal positions")
            self.calibrated_default_positions = self.nominal_default_positions.copy()
            self.calibration_loaded = True

    def recalibrate_service(self, request, response):
        """Service to reload calibration from NVRAM"""
        self.load_nvram_calibration()
        response.success = self.calibration_loaded
        response.message = "Recalibration from NVRAM completed" if self.calibration_loaded else "Failed to load NVRAM"
        return response

    def get_default_positions(self):
        """Get calibrated default positions"""
        if self.calibration_loaded and self.calibrated_default_positions is not None:
            return self.calibrated_default_positions
        else:
            self.get_logger().warn_once("Using nominal positions - calibration not loaded")
            return self.nominal_default_positions

    def joint_state_callback(self, msg):
        """Update joint positions, velocities, and efforts"""
        with self.state_lock:
            for i, name in enumerate(msg.name):
                if name in JOINT_NAME_TO_IDX:
                    idx = JOINT_NAME_TO_IDX[name]
                    self.joint_positions[idx] = msg.position[i]
                    if len(msg.velocity) > i:
                        self.joint_velocities[idx] = msg.velocity[i]
                    if len(msg.effort) > i:
                        self.joint_efforts[idx] = msg.effort[i]
            
            if not self.joint_data_received:
                self.joint_data_received = True
                
                # Log the initial joint positions vs calibrated defaults
                if self.calibration_loaded:
                    self.get_logger().info("Initial joint positions vs calibrated defaults:")
                    for i, name in enumerate(JOINT_NAMES):
                        diff = self.joint_positions[i] - self.calibrated_default_positions[i]
                        self.get_logger().info(f"  {name}: current={self.joint_positions[i]:.3f}, "
                                             f"calibrated={self.calibrated_default_positions[i]:.3f}, "
                                             f"diff={diff:.3f} rad ({np.rad2deg(diff):.1f} deg)")

    def cmd_vel_callback(self, msg):
        """Update command velocities"""
        with self.state_lock:
            old_cmd_linear = self.cmd_vel_linear[:2].copy()  # Fixed: only compare x,y components
            self.cmd_vel_linear[0] = msg.linear.x
            self.cmd_vel_linear[1] = msg.linear.y
            self.cmd_vel_angular[2] = msg.angular.z
            self.last_cmd_time = self.get_clock().now()
            
            # Debug logging for command reception
            if np.any(np.abs(self.cmd_vel_linear[:2]) > 0.01) or np.abs(self.cmd_vel_angular[2]) > 0.01:
                if not self.has_received_cmd:
                    self.has_received_cmd = True
                    self.get_logger().info(f"First movement command received: linear=[{self.cmd_vel_linear[0]:.2f}, {self.cmd_vel_linear[1]:.2f}], angular={self.cmd_vel_angular[2]:.2f}")
                elif np.any(np.abs(old_cmd_linear - self.cmd_vel_linear[:2]) > 0.1):
                    self.get_logger().debug(f"Command updated: linear=[{self.cmd_vel_linear[0]:.2f}, {self.cmd_vel_linear[1]:.2f}], angular={self.cmd_vel_angular[2]:.2f}")

    def odom_callback(self, msg):
        """Update base velocities"""
        with self.state_lock:
            self.base_lin_vel[0] = msg.twist.twist.linear.x
            self.base_lin_vel[1] = msg.twist.twist.linear.y
            self.base_lin_vel[2] = 0.0  # Set Z to 0 for RSL compatibility
            
            self.base_ang_vel[0] = 0.0  # Set roll rate to 0
            self.base_ang_vel[1] = 0.0  # Set pitch rate to 0
            self.base_ang_vel[2] = msg.twist.twist.angular.z

    def construct_observation(self):
        """Construct 48-dimensional observation vector with calibrated positions"""
        default_pos = self.get_default_positions()
        
        # Build observation components
        cmd_vels = np.array([self.cmd_vel_linear[0], self.cmd_vel_linear[1], self.cmd_vel_angular[2]])
        
        # CRITICAL: Use calibrated default positions for relative joint positions
        joint_pos_rel = self.joint_positions - default_pos #default_pos - self.joint_positions  # IL convention self.joint_positions - default_pos # RL Convention
        
        joint_vels = self.joint_velocities
        last_actions = self.last_actions
        proj_gravity = self.projected_gravity
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        
        # Assemble observation (IL order)
        # obs = np.concatenate([
        #     cmd_vels,      # 0-2
        #     joint_pos_rel, # 3-14
        #     joint_vels,    # 15-26
        #     last_actions,  # 27-38
        #     proj_gravity,  # 39-41
        #     base_lin_vel,  # 42-44
        #     base_ang_vel   # 45-47
        # ])

        rsl_obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            proj_gravity,
            cmd_vels,
            joint_pos_rel,
            joint_vels,
            last_actions
        ], axis=-1)
        
        return rsl_obs.astype(np.float32)#obs.astype(np.float32)

    def is_command_active(self):
        """Check if we have an active movement command"""
        if self.last_cmd_time is None:
            return False
        time_since_cmd = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        if time_since_cmd > self.cmd_timeout:
            return False
        if np.all(np.abs(self.cmd_vel_linear[:2]) < 0.01) and np.abs(self.cmd_vel_angular[2]) < 0.01:
            return False
        return True

    def move_to_default_pose(self):
        """Smooth transition to calibrated default pose"""
        default_pos = self.get_default_positions()
        
        if self.init_start_time is None:
            self.init_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.init_positions = self.joint_positions.copy()
            self.get_logger().info(f"Moving to calibrated default stance...")
        
        current_time = self.get_clock().now().nanoseconds * 1e-9
        elapsed = current_time - self.init_start_time
        
        if elapsed >= self.init_duration:
            self.initialized = True
            self.get_logger().info("Default stance reached. Starting MLP control...")
            return True
        
        # Smooth interpolation with cosine profile
        alpha = min(elapsed / self.init_duration, 1.0)
        alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
        
        target_positions = self.init_positions * (1 - alpha) + default_pos * alpha
        
        # Send position command
        self.send_position_command(target_positions)
        return False

    def mlp_inference_loop(self):
        """Run MLP inference at lower frequency (50Hz)"""
        if not self.joint_data_received or not self.calibration_loaded:
            return
        
        if not self.initialized:
            self.move_to_default_pose()
            return
        
        with self.state_lock:
            # Check for active command
            if not self.is_command_active():
                # Return to default stance
                if self.has_received_cmd:
                    self.has_received_cmd = False
                    self.last_actions = np.zeros(12)
                    self.smoothed_actions = np.zeros(12)  # Reset smoothed actions
                    self.action_buffer = []
                
                default_pos = self.get_default_positions()
                self.send_position_command(default_pos)
                return
            
            # Construct observation with calibrated positions
            obs = self.construct_observation()
            
            # Publish observation for debugging
            obs_msg = Float32MultiArray(data=obs.tolist())
            self.pub_mlp_observation.publish(obs_msg)
            
            # Run MLP inference
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
            
            if self.use_obs_normalization:
                obs_mean = self.obs_rms_mean
                obs_std = torch.sqrt(self.obs_rms_var + 1e-8)
                obs_tensor = (obs_tensor - obs_mean) / obs_std
            
            with torch.no_grad():
                raw_action = self.model(obs_tensor).squeeze(0).cpu().numpy()
            
            # Publish raw output for debugging
            raw_msg = Float32MultiArray(data=raw_action.tolist())
            self.pub_mlp_output.publish(raw_msg)
            
            # Apply joint-specific scaling
            scaled_action = raw_action * self.action_scale_vector
            
            # Apply EMA smoothing (restored from original code)
            self.smoothed_actions = (self.action_smoothing * self.smoothed_actions + 
                                   (1 - self.action_smoothing) * scaled_action)
            
            # Clip actions
            action = np.clip(self.smoothed_actions, -1.5, 1.5)
            self.last_actions = action.copy()
            
            if self.use_interpolation:
                # Generate interpolation targets for high-frequency loop
                self.generate_interpolation_buffer(action)
            else:
                # Direct control at MLP frequency (like original code)
                default_pos = self.get_default_positions()
                target_positions = default_pos + action
                
                # Log the command being sent (for debugging)
                self.get_logger().debug(f"Sending joint command: action range [{action.min():.3f}, {action.max():.3f}], "
                                       f"target range [{target_positions.min():.3f}, {target_positions.max():.3f}]")
                
                # Send command directly
                traj_msg = JointTrajectory()
                traj_msg.joint_names = JOINT_NAMES
                
                point = JointTrajectoryPoint(
                    positions=target_positions.tolist(),
                    velocities=np.zeros(12).tolist(),
                    time_from_start=rclpy.duration.Duration(seconds=self.mlp_dt).to_msg()
                )
                
                traj_msg.points = [point]
                self.pub_joint_trajectory.publish(traj_msg)
                
                # Also log once per second for visibility
                current_time = int(self.get_clock().now().nanoseconds * 1e-9)
                if current_time != getattr(self, 'last_log_time', 0):
                    self.last_log_time = current_time
                    self.get_logger().info(f"Publishing joints: cmd_vel=[{self.cmd_vel_linear[0]:.2f}, {self.cmd_vel_angular[2]:.2f}], "
                                         f"action_range=[{action.min():.2f}, {action.max():.2f}]")

    def generate_interpolation_buffer(self, target_action):
        """Generate smooth interpolation targets for high-frequency control"""
        with self.state_lock:
            if len(self.action_buffer) == 0:
                # Initialize with current action
                start_action = self.last_actions
            else:
                # Continue from last buffered action
                start_action = self.action_buffer[-1]
            
            # Clear buffer and generate new interpolation points
            self.action_buffer = []
            for i in range(self.interpolation_steps):
                alpha = (i + 1) / self.interpolation_steps
                # Use cubic interpolation for smoother motion
                alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                interpolated = start_action * (1 - alpha_smooth) + target_action * alpha_smooth
                self.action_buffer.append(interpolated)

    def high_freq_control_loop(self):
        """High-frequency control loop (250Hz) - only runs if interpolation is enabled"""
        if not self.use_interpolation:
            return  # Skip if using direct control
            
        if not self.initialized or not self.calibration_loaded:
            return
        
        with self.state_lock:
            if len(self.action_buffer) == 0:
                return
            
            # Pop next interpolated action
            action = self.action_buffer.pop(0)
            
            # Get calibrated default positions
            default_pos = self.get_default_positions()
            
            # Apply action to calibrated positions
            target_positions = default_pos + action
            
            if self.use_effort_control:
                # Pure effort/torque control
                self.send_effort_command(action)
            elif self.use_hybrid_control:
                # Hybrid: PD control + feedforward
                self.send_hybrid_command(target_positions, action)
            else:
                # Pure position control
                self.send_position_command(target_positions)

    def send_position_command(self, target_positions):
        """Send position command through trajectory controller"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = JOINT_NAMES
        
        point = JointTrajectoryPoint(
            positions=target_positions.tolist(),
            velocities=np.zeros(12).tolist(),
            time_from_start=rclpy.duration.Duration(seconds=self.control_dt).to_msg()
        )
        
        traj_msg.points = [point]
        self.pub_joint_trajectory.publish(traj_msg)

    def send_effort_command(self, efforts):
        """Send direct effort/torque commands"""
        effort_msg = Float64MultiArray(data=efforts.tolist())
        self.pub_effort_commands.publish(effort_msg)

    def send_hybrid_command(self, target_positions, feedforward_efforts):
        """Send hybrid position + feedforward effort command"""
        # Calculate PD control efforts
        position_error = target_positions - self.joint_positions
        velocity_error = -self.joint_velocities  # Assuming target velocity is 0
        
        pd_efforts = (self.kp_vector * position_error + 
                     self.kd_vector * velocity_error)
        
        # Combine PD and feedforward
        total_efforts = pd_efforts + feedforward_efforts * 1000.0  # Scale feedforward contribution
        
        # Send as effort command
        self.send_effort_command(total_efforts)

def main(args=None):
    rclpy.init(args=args)
    controller = UnifiedMLPController()
    
    # Use MultiThreadedExecutor for better performance
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down Unified MLP Controller...")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
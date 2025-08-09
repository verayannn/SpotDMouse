"""
Mini-Pupper MLP controller – Fixed observation pipeline and action processing.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MLPController(Node):
    # ════════════════════════════════════════════════════════════════════════
    def __init__(self):
        super().__init__("mlp_controller")

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter(
            "model_path",
            "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/sim2real/newwalkingmlp.pt")
        self.declare_parameter("control_frequency", 50.0)   # Hz
        self.declare_parameter("action_scale", 0.25)        # Scale factor for actions
        self.declare_parameter("smoothing_alpha", 0.8)      # Action smoothing (0=full smooth, 1=no smooth)

        self.dt = 1.0 / float(self.get_parameter("control_frequency").value)
        self.action_scale = float(self.get_parameter("action_scale").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)

        # ── load network & statistics ────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(self.get_parameter("model_path").value).expanduser()
        self.model, self.action_std, self.obs_mean, self.obs_var = self._load_checkpoint(ckpt_path)
        if self.model is None:
            raise RuntimeError("Could not load NN checkpoint.")

        # ── state holders ────────────────────────────────────────────────────
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.prev_joint_positions = np.zeros(12)
        self.prev_joint_time = None
        
        # Use rolling average for velocity estimation
        self.vel_history = deque(maxlen=5)
        
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.gravity_vec = np.array([0.0, 0.0, -1.0])

        self.velocity_commands = np.array([0.05, 0.0, 0.0])
        self.last_action = np.zeros(12)
        
        # Initialize previous target for smoothing
        self.prev_target_positions = None
        
        # Add gait phase tracking for cyclic motion
        self.phase = 0.0
        self.phase_freq = 2.2  # Hz, typical gait frequency

        # Isaac-lab joint order → robot joint names
        self.joint_names = [
            "base_lf1", "lf1_lf2", "lf2_lf3",
            "base_rf1", "rf1_rf2", "rf2_rf3",
            "base_lb1", "lb1_lb2", "lb2_lb3",
            "base_rb1", "rb1_rb2", "rb2_rb3",
        ]
        self.name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        
        # Default joint positions (standing pose)
        self.default_positions = np.array([0.0, 0.52, -1.05] * 4, dtype=np.float32)
        
        # Joint limits
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.09] * 4, dtype=np.float32)
        self.joint_upper_limits = np.array([0.5, 1.57, -0.52] * 4, dtype=np.float32)

        # ── ROS I/O ──────────────────────────────────────────────────────────
        self.create_subscription(JointState, "/joint_states", self._cb_joint, 10)
        self.create_subscription(Twist, "/cmd_vel", self._cb_cmd_vel, 10)
        self.create_subscription(Odometry, "/odom", self._cb_odom, 10)
        self.create_subscription(Imu, "/imu/data", self._cb_imu, 10)

        self.pub_traj = self.create_publisher(
            JointTrajectory, "/joint_group_effort_controller/joint_trajectory", 10)

        self.timer = self.create_timer(self.dt, self._control_loop)
        self.step = 0
        self.start_time = time.time()

        self._log_joint_order()
        self.get_logger().info(f"Controller ready @ {1/self.dt:.1f} Hz")
        self.get_logger().info(f"Action scale: {self.action_scale}, Smoothing: {self.smoothing_alpha}")

    # ════════════════════════════════════════════════════════════════════════
    #                           Checkpoint loading
    # ════════════════════════════════════════════════════════════════════════
    def _load_checkpoint(self, path: Path):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)

            model = torch.nn.Sequential(
                torch.nn.Linear(48, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(),
                torch.nn.Linear(256, 128), torch.nn.ELU(),
                torch.nn.Linear(128, 12),
            )
            
            # actor.* → model
            actor_state = {k.replace("actor.", ""): v
                           for k, v in ckpt["model_state_dict"].items()
                           if k.startswith("actor.")}
            model.load_state_dict(actor_state)
            model.eval().to(self.device)

            # Get action std
            action_std = ckpt["model_state_dict"].get("std", torch.ones(12) * 0.5)
            if isinstance(action_std, torch.Tensor):
                action_std = action_std.cpu().numpy()

            # Get observation normalization stats
            obs_mean = ckpt.get("obs_rms_mean", np.zeros(48, dtype=np.float32))
            obs_var = ckpt.get("obs_rms_var", np.ones(48, dtype=np.float32))
            
            # Convert to numpy if torch tensors
            if isinstance(obs_mean, torch.Tensor):
                obs_mean = obs_mean.cpu().numpy()
            if isinstance(obs_var, torch.Tensor):
                obs_var = obs_var.cpu().numpy()

            self.get_logger().info(f"Loaded checkpoint from {path}")
            self.get_logger().info(f"Action std shape: {action_std.shape}, mean: {np.mean(action_std):.3f}")
            return model, action_std, obs_mean, obs_var
            
        except Exception as e:
            self.get_logger().error(f"Checkpoint load failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

    # ════════════════════════════════════════════════════════════════════════
    #                               Callbacks
    # ════════════════════════════════════════════════════════════════════════
    def _cb_joint(self, msg: JointState):
        cur = np.zeros(12, dtype=np.float32)
        vel = np.zeros(12, dtype=np.float32)
        
        for name, pos, vel_val in zip(msg.name, msg.position, msg.velocity if msg.velocity else [0]*len(msg.name)):
            if name in self.name_to_idx:
                idx = self.name_to_idx[name]
                cur[idx] = pos
                vel[idx] = vel_val

        self.joint_positions = cur
        
        # Use provided velocities if available, otherwise estimate
        if msg.velocity:
            self.joint_velocities = vel
        else:
            now = time.time()
            if self.prev_joint_time is not None:
                dt = now - self.prev_joint_time
                if dt > 1e-3:
                    estimated_vel = (cur - self.prev_joint_positions) / dt
                    self.vel_history.append(estimated_vel)
                    if len(self.vel_history) > 0:
                        self.joint_velocities = np.mean(self.vel_history, axis=0)
            
            self.prev_joint_positions = cur.copy()
            self.prev_joint_time = now

    def _cb_cmd_vel(self, msg: Twist):
        # Scale commands appropriately
        self.velocity_commands[0] = np.clip(msg.linear.x, -1.0, 1.0)
        self.velocity_commands[1] = np.clip(msg.linear.y, -0.5, 0.5)
        self.velocity_commands[2] = np.clip(msg.angular.z, -1.0, 1.0)

    def _cb_odom(self, msg: Odometry):
        self.base_lin_vel[:] = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]
        self.base_ang_vel[:] = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]

    def _cb_imu(self, msg: Imu):
        # Get gravity vector from IMU
        g = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z])
        n = np.linalg.norm(g)
        if n > 0.1:
            self.gravity_vec = -g / n  # Negative for proper orientation

    # ════════════════════════════════════════════════════════════════════════
    #                           Observation building
    # ════════════════════════════════════════════════════════════════════════
    def _build_obs(self):
        """Build 48-dim observation vector matching training format"""
        
        # Add phase information for gait (sin and cos for continuity)
        phase_sin = np.sin(2 * np.pi * self.phase)
        phase_cos = np.cos(2 * np.pi * self.phase)
        
        # Build observation vector (must be 48 dims total)
        raw = np.concatenate([
            self.base_lin_vel,              # 3
            self.base_ang_vel,              # 3  
            self.gravity_vec,               # 3
            self.velocity_commands,         # 3
            self.joint_positions,           # 12
            self.joint_velocities,          # 12
            self.last_action,               # 12
        ]).astype(np.float32)              # Total: 48

        # Normalize observation
        eps = 1e-8
        norm = (raw - self.obs_mean) / np.sqrt(self.obs_var + eps)
        norm = np.clip(norm, -10.0, 10.0)
        
        return norm

    # ════════════════════════════════════════════════════════════════════════
    #                               Main control loop
    # ════════════════════════════════════════════════════════════════════════
    def _control_loop(self):
        self.step += 1
        
        # Update gait phase
        self.phase = (self.phase + self.phase_freq * self.dt) % 1.0
        
        # Build observation and get action from network
        obs = torch.from_numpy(self._build_obs()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean_action = self.model(obs).cpu().numpy()[0]
        
        # Add noise for exploration (optional, can be reduced for deployment)
        noise_scale = 0.05  # Small noise for smoother motion
        raw_action = mean_action + np.random.randn(12) * self.action_std * noise_scale
        
        # Scale actions appropriately
        scaled_action = raw_action * self.action_scale
        
        # Store action for next observation
        self.last_action = scaled_action
        
        # Calculate target positions (residual on top of default)
        target_positions = self.default_positions + scaled_action
        
        # Apply smoothing for hardware
        if self.prev_target_positions is not None:
            target_positions = (self.smoothing_alpha * target_positions + 
                               (1 - self.smoothing_alpha) * self.prev_target_positions)
        
        # Clip to joint limits
        target_positions = np.clip(target_positions, 
                                   self.joint_lower_limits, 
                                   self.joint_upper_limits)
        
        self.prev_target_positions = target_positions.copy()
        
        # Publish trajectory
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = target_positions.tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(self.dt * 1e9)
        traj.points = [pt]

        self.pub_traj.publish(traj)

        # Enhanced logging every second
        if self.step % int(1 / self.dt) == 0:
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f"t={elapsed:.1f}s | "
                f"cmd=[{self.velocity_commands[0]:.2f}, {self.velocity_commands[1]:.2f}, {self.velocity_commands[2]:.2f}] | "
                f"phase={self.phase:.2f} | "
                f"act[0:3]={scaled_action[:3].round(3)} | "
                f"tgt[0:3]={target_positions[:3].round(3)}"
            )
            
            # Debug: Check if observations are changing
            obs_hash = hash(obs.cpu().numpy().tobytes())
            self.get_logger().debug(f"Obs hash: {obs_hash}, Joint vel norm: {np.linalg.norm(self.joint_velocities):.3f}")

    # ════════════════════════════════════════════════════════════════════════
    def _log_joint_order(self):
        self.get_logger().info("Joint order mapping:")
        for i, n in enumerate(self.joint_names):
            self.get_logger().info(f"  {i:2d}: {n}")


# ════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = MLPController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
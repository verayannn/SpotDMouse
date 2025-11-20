#!/usr/bin/env python3
"""
Optimized MP2 RL Controller with Isaac Sim Observation Alignment
Removes ROS2 dependencies for direct ESP32 control
"""

import numpy as np
import torch
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time
import threading
from collections import deque

class MP2IsaacRLController:
    """
    Direct control of Mini Pupper 2 using Isaac Sim trained policy.
    Properly aligns observations with training format.
    """
    
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        
        print("Initializing MP2 Isaac RL Controller...")

        # Joint order mapping
        # Isaac order: [LF, RF, LB, RB] 
        # ESP32 order: [FR, FL, BR, BL]
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        # Create inverse mapping
        self.isaac_to_esp32 = np.zeros(12, dtype=int)
        for i, esp_idx in enumerate(self.esp32_servo_order):
            self.isaac_to_esp32[esp_idx] = i
        
        # Initialize ESP32
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Test connection
        test_pos = self.esp32.servos_get_position()
        if test_pos is None or len(test_pos) != 12:
            raise RuntimeError("Cannot connect to ESP32")
        print(f"ESP32 connected. Current positions: {test_pos}")
        
        # Control parameters (tuned based on your testing)
        self.ACTION_SCALE = 0.5  # Scale from simulation to real
        self.MAX_ACTION_CHANGE = 0.15  # Smooth action changes
        self.CONTROL_FREQUENCY = 50  # Hz (matches training: decimation=10, dt=0.002)
        
        # Servo calibration
        self.servo_offset = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # CRITICAL: Use Isaac Sim default positions
        # These match your training config exactly
        self.isaac_defaults = np.array([
            -0.1, 0.8, -1.5,  # LF
             0.1, 0.8, -1.5,  # RF
            -0.1, 0.8, -1.5,  # LB
             0.1, 0.8, -1.5   # RB
        ])
        
        # Get current robot positions
        current_servos = np.array(self.esp32.servos_get_position())
        self.current_positions = (current_servos - self.servo_offset) / self.servo_scale
        
        print(f"Current positions (rad): {self.current_positions.round(3)}")
        print(f"Isaac defaults (rad): {self.isaac_defaults.round(3)}")
        print(f"Difference: {(self.current_positions - self.isaac_defaults).round(3)}")
        
        # State tracking
        self.prev_positions = self.current_positions.copy()
        self.prev_actions = np.zeros(12)
        self.prev_time = time.time()
        
        # Base velocity estimation (critical for closed-loop control)
        self.velocity_estimator = VelocityEstimator()
        
        # Load policy
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Policy loaded from {policy_path}")
        
        # Control state
        self.control_active = False
        self.velocity_command = np.zeros(3)
        self.shutdown = False
        
        # Performance monitoring
        self.loop_times = deque(maxlen=50)
        
    def calibrate_to_default(self):
        """Move robot to Isaac Sim default standing position"""
        print("\nCalibrating to Isaac Sim default position...")
        
        target_servos = self.isaac_defaults * self.servo_scale + self.servo_offset
        target_servos = np.clip(target_servos, 100, 924)
        
        # Get current position
        current_servos = np.array(self.esp32.servos_get_position())
        
        # Smooth transition over 2 seconds
        steps = 40
        for i in range(steps):
            alpha = (i + 1) / steps
            # Use cosine interpolation for smoother motion
            alpha = 0.5 * (1 - np.cos(alpha * np.pi))
            
            interpolated = current_servos + alpha * (target_servos - current_servos)
            servo_commands = [int(pos) for pos in interpolated]
            
            self.esp32.servos_set_position_torque(servo_commands, [1]*12)
            time.sleep(0.05)
        
        # Update state
        self.current_positions = self.isaac_defaults.copy()
        self.prev_positions = self.isaac_defaults.copy()
        
        print("Calibration complete!")
        
    def get_observation(self):
        """
        Build observation vector matching Isaac Sim training format.
        
        Returns:
            60-dimensional observation vector
        """
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Clamp dt to reasonable range
        dt = np.clip(dt, 0.01, 0.1)
        
        try:
            # Get sensor data
            raw_positions = np.array(self.esp32.servos_get_position())
            raw_loads = np.array(self.esp32.servos_get_load())
            imu_data = self.esp32.imu_get_data()
        except Exception as e:
            print(f"Sensor read error: {e}")
            return self._get_safe_observation()

        ###
        # Get sensor data
        raw_positions = np.array(self.esp32.servos_get_position())
        raw_loads = np.array(self.esp32.servos_get_load())

        # REORDER to Isaac format
        isaac_positions = raw_positions[self.esp32_servo_order]
        isaac_loads = raw_loads[self.esp32_servo_order]

        # Now use isaac_positions
        current_positions = (isaac_positions - self.servo_offset) / self.servo_scale
        ###
        
        # 1. Joint positions (relative to Isaac defaults)
        # current_positions = (raw_positions - self.servo_offset) / self.servo_scale
        joint_pos_rel = current_positions - self.isaac_defaults
        
        # 2. Joint velocities
        joint_velocities = (current_positions - self.prev_positions) / dt
        joint_velocities = np.clip(joint_velocities, -50, 50)
        
        # 3. Joint efforts (normalized)
        joint_efforts = np.clip(raw_loads / 500.0, -10, 10)
        
        # 4. Base angular velocity
        base_ang_vel = np.array([
            imu_data['gx'] * 0.01745,  # deg/s to rad/s
            imu_data['gy'] * 0.01745,
            imu_data['gz'] * 0.01745
        ])
        base_ang_vel = np.clip(base_ang_vel, -4, 4)
        
        # 5. Projected gravity
        acc = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 0.1:
            projected_gravity = acc / acc_norm
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        # 6. Base linear velocity (estimated)
        base_lin_vel = self.velocity_estimator.estimate(
            self.velocity_command,
            joint_velocities,
            base_ang_vel,
            dt
        )
        base_lin_vel = np.clip(base_lin_vel, -2, 2)
        
        # Build observation
        observation = np.concatenate([
            base_lin_vel,          # 3
            base_ang_vel,         # 3
            projected_gravity,    # 3
            self.velocity_command,# 3
            joint_pos_rel,        # 12
            joint_velocities,     # 12
            joint_efforts,        # 12
            self.prev_actions     # 12
        ])
        
        # Update state
        self.prev_positions = current_positions.copy()
        self.prev_time = current_time
        
        return observation
    
    def _get_safe_observation(self):
        """Return safe default observation on sensor error"""
        return np.concatenate([
            np.zeros(3),  # base_lin_vel
            np.zeros(3),  # base_ang_vel
            np.array([0, 0, -1]),  # gravity
            self.velocity_command,  # commands
            np.zeros(12),  # joint_pos
            np.zeros(12),  # joint_vel
            np.zeros(12),  # joint_effort
            self.prev_actions  # prev_actions
        ])
    
    def process_actions(self, raw_actions):
        """Process raw policy output to servo commands"""
        
        # Scale actions
        scaled = raw_actions * self.ACTION_SCALE
        
        # Apply per-joint limits based on training ranges
        limited = scaled.copy()
        for i in range(4):  # 4 legs
            base_idx = i * 3
            # Hip: smaller range
            limited[base_idx] = np.clip(limited[base_idx], -0.5, 0.5)
            # Thigh: medium range
            limited[base_idx + 1] = np.clip(limited[base_idx + 1], -1.0, 1.0)
            # Calf: medium range
            limited[base_idx + 2] = np.clip(limited[base_idx + 2], -1.0, 1.0)
        
        # Smooth actions
        action_delta = limited - self.prev_actions
        action_delta = np.clip(action_delta, -self.MAX_ACTION_CHANGE, self.MAX_ACTION_CHANGE)
        smoothed = self.prev_actions + action_delta
        
        # Update for next iteration
        self.prev_actions = smoothed.copy()
        
        # Convert to servo positions
        absolute_positions = smoothed + self.isaac_defaults
        servo_positions = absolute_positions * self.servo_scale + self.servo_offset
        servo_positions = np.clip(servo_positions, 100, 924)

        esp32_positions = servo_positions[self.isaac_to_esp32]
        
        return [int(pos) for pos in esp32_positions]
    
    def control_loop(self):
        """Main control loop running at specified frequency"""
        
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl loop started at {self.CONTROL_FREQUENCY}Hz")
        print("Waiting for velocity commands...")
        
        while not self.shutdown:
            loop_start = time.time()
            
            try:
                if self.control_active:
                    # Get observation
                    obs = self.get_observation()
                    
                    # Run policy
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        raw_actions = self.policy(obs_tensor).squeeze().numpy()
                    
                    # Process and send actions
                    servo_commands = self.process_actions(raw_actions)
                    self.esp32.servos_set_position_torque(servo_commands, [1]*12)
                    
                    # Monitor performance (every second)
                    if len(self.loop_times) == self.loop_times.maxlen:
                        avg_time = np.mean(self.loop_times)
                        if int(loop_start) % 1 == 0:
                            print(f"Loop: {1/avg_time:.1f}Hz | "
                                  f"Cmd: [{self.velocity_command[0]:.2f}, "
                                  f"{self.velocity_command[1]:.2f}, "
                                  f"{self.velocity_command[2]:.2f}] | "
                                  f"Actions: [{np.min(self.prev_actions):.2f}, "
                                  f"{np.max(self.prev_actions):.2f}]")
                
            except Exception as e:
                print(f"Control loop error: {e}")
                self.control_active = False
            
            # Maintain frequency
            elapsed = time.time() - loop_start
            self.loop_times.append(elapsed)
            
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
        
        print("Control loop stopped")
    
    def set_velocity_command(self, vx, vy, vyaw):
        """Set velocity command and activate control"""
        self.velocity_command = np.array([
            np.clip(vx, -0.35, 0.40),   # Match training ranges
            np.clip(vy, -0.35, 0.35),
            np.clip(vyaw, -0.30, 0.30)
        ])
        
        # Activate on non-zero command
        if np.any(np.abs(self.velocity_command) > 0.01):
            if not self.control_active:
                print(f"Control activated with command: {self.velocity_command}")
            self.control_active = True
        else:
            self.control_active = False
    
    def shutdown_controller(self):
        """Graceful shutdown"""
        print("\nShutting down controller...")
        self.control_active = False
        self.shutdown = True
        time.sleep(0.2)
        
        # Return to default position
        self.calibrate_to_default()
        print("Controller shutdown complete")


class VelocityEstimator:
    """Estimates base velocity from available sensors"""
    
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Exponential smoothing factor
        self.estimated = np.zeros(3)
        self.prev_command = np.zeros(3)
        
    def estimate(self, command, joint_vel, ang_vel, dt):
        """
        Estimate base velocity using multiple signals.
        
        In simulation, this comes from perfect odometry.
        On real robot, we must estimate it.
        """
        
        # Method 1: Command following with lag
        # Robot velocity follows command with delay and scaling
        command_follow = command * 0.6  # Assume 60% tracking
        
        # Method 2: Joint velocity correlation
        # Higher joint velocities indicate motion
        avg_joint_speed = np.mean(np.abs(joint_vel))
        motion_indicator = np.tanh(avg_joint_speed / 5.0)
        
        # Method 3: Gait phase detection
        # Alternating joint velocities suggest walking
        hip_vels = joint_vel[[0, 3, 6, 9]]  # Hip joints
        gait_indicator = np.std(hip_vels)
        
        # Combine estimates
        vx_est = command[0] * (0.5 + 0.5 * motion_indicator)
        vy_est = command[1] * 0.5  # Lateral is harder
        vz_est = 0.0  # Assume flat ground
        
        # Apply smoothing
        new_estimate = np.array([vx_est, vy_est, vz_est])
        self.estimated = self.alpha * new_estimate + (1 - self.alpha) * self.estimated
        
        return self.estimated


def keyboard_interface(controller):
    """Simple keyboard control for testing"""
    import sys, termios, tty, select
    
    def get_key(timeout=0.1):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return key
    
    print("\n" + "="*50)
    print("KEYBOARD CONTROL")
    print("="*50)
    print("w/s: forward/backward")
    print("a/d: strafe left/right")
    print("q/e: turn left/right")
    print("space: stop")
    print("+/-: adjust speed")
    print("c: calibrate to default")
    print("x: exit")
    print("="*50)
    
    speed = 0.3
    velocities = {'vx': 0, 'vy': 0, 'vyaw': 0}
    
    while True:
        key = get_key(0.1)
        
        if key == 'x':
            break
        elif key == 'w':
            velocities['vx'] = speed
        elif key == 's':
            velocities['vx'] = -speed
        elif key == 'a':
            velocities['vy'] = speed
        elif key == 'd':
            velocities['vy'] = -speed
        elif key == 'q':
            velocities['vyaw'] = speed
        elif key == 'e':
            velocities['vyaw'] = -speed
        elif key == ' ':
            velocities = {'vx': 0, 'vy': 0, 'vyaw': 0}
        elif key == '+' or key == '=':
            speed = min(speed + 0.05, 0.5)
            print(f"Speed: {speed:.2f}")
        elif key == '-':
            speed = max(speed - 0.05, 0.1)
            print(f"Speed: {speed:.2f}")
        elif key == 'c':
            controller.control_active = False
            controller.calibrate_to_default()
        elif key == '':
            # No key pressed - decay velocities
            for k in velocities:
                velocities[k] *= 0.9
                if abs(velocities[k]) < 0.01:
                    velocities[k] = 0
        
        # Send command
        controller.set_velocity_command(
            velocities['vx'], 
            velocities['vy'], 
            velocities['vyaw']
        )


def main():
    print("="*60)
    print("MP2 ISAAC SIM RL CONTROLLER")
    print("="*60)
    
    # Initialize controller
    controller = MP2IsaacRLController()
    
    # Calibrate to default position
    print("\nCalibrate to Isaac Sim default stance? (y/n)")
    if input().lower() == 'y':
        controller.calibrate_to_default()
    
    # Start control loop
    control_thread = threading.Thread(target=controller.control_loop)
    control_thread.daemon = True
    control_thread.start()
    
    # Run keyboard interface
    try:
        keyboard_interface(controller)
    except KeyboardInterrupt:
        pass
    
    # Shutdown
    controller.shutdown_controller()
    print("\nController stopped successfully")


if __name__ == "__main__":
    main()
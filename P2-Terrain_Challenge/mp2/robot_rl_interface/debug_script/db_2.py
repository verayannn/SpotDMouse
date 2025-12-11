import numpy as np
import pandas as pd
import time
import os
import torch
from datetime import datetime
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class RealRobotObservationLogger:
    """
    Records full 60-dim observation vector from real robot.
    Matches Isaac Lab training observation format exactly.
    
    Observation order (60 dims):
        [0:3]   base_lin_vel      - Estimated
        [3:6]   base_ang_vel      - IMU gyro
        [6:9]   projected_gravity - IMU accel normalized
        [9:12]  velocity_commands - User input
        [12:24] joint_pos_rel     - Servo position - default
        [24:36] joint_vel         - Derived from position
        [36:48] joint_effort      - Servo load
        [48:60] prev_actions      - Previous policy output
    """
    
    def __init__(self, output_dir="realbot_60dim_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Hardware
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.pwm_params = self.hardware.pwm_params
        self.servo_params = self.hardware.servo_params
        self.esp32 = self.pwm_params.esp32
        
        # Simulation defaults (must match training)
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])
        
        # Hardware mapping
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}  # FR->RF, FL->LF, BR->RB, BL->LB
        
        # Direction multipliers (tune these based on test results)
        self.effort_direction = np.array([
            1.0,  1.0,  1.0,   # LF
            1.0,  1.0,  1.0,   # RF
            1.0,  1.0,  1.0,   # LB
            1.0,  1.0,  1.0,   # RB
        ])
        
        # Calibration
        self.gyro_offset = np.zeros(3)
        self.gyro_scale = np.pi / 180.0  # deg/s -> rad/s
        self.accel_scale = 9.81  # g -> m/s²
        self.load_offset = np.zeros(12)
        self.load_scale = 5000.0
        
        # State
        self.prev_joint_angles = self.sim_default_positions.copy()
        self.prev_actions = np.zeros(12)
        self.prev_time = time.time()
        
        # Data storage
        self.observations = []
        self.actions = []
        self.raw_sensor_data = []  # For debugging
        
        # Joint names for CSV columns
        self.joint_names = [
            'base_lf1', 'lf1_lf2', 'lf2_lf3',
            'base_rf1', 'rf1_rf2', 'rf2_rf3',
            'base_lb1', 'lb1_lb2', 'lb2_lb3',
            'base_rb1', 'rb1_rb2', 'rb2_rb3'
        ]
        
        print("Calibrating sensors...")
        self._calibrate_sensors()
        print("Calibration complete.\n")
    
    def _calibrate_sensors(self, samples=50):
        """Calibrate gyro and load offsets at rest."""
        gyro_samples = []
        load_samples = []
        
        for _ in range(samples):
            imu = self.esp32.imu_get_data()
            if imu:
                gyro_samples.append([imu['gx'], imu['gy'], imu['gz']])
            
            load = self.esp32.servos_get_load()
            if load:
                load_samples.append(load)
            
            time.sleep(0.02)
        
        if gyro_samples:
            self.gyro_offset = np.mean(gyro_samples, axis=0)
            print(f"  Gyro offset: {self.gyro_offset.round(2)}")
            
            # Auto-detect units
            if np.max(np.abs(self.gyro_offset)) > 5:
                self.gyro_scale = np.pi / 180.0
                print(f"  Gyro in deg/s, converting to rad/s")
            else:
                self.gyro_scale = 1.0
                print(f"  Gyro already in rad/s")
        
        if load_samples:
            self.load_offset = np.mean(load_samples, axis=0)
            print(f"  Load offset: {self.load_offset.astype(int)}")
    
    # ==================== SENSOR READING ====================
    
    def read_joint_positions(self):
        """Read servo positions, convert to Isaac order."""
        raw_positions = self.esp32.servos_get_position()
        if raw_positions is None:
            return self.prev_joint_angles.copy(), None
        
        joint_angles_hw = np.zeros((3, 4))
        
        for leg_index in range(4):
            for axis_index in range(3):
                servo_id = self.pwm_params.servo_ids[axis_index, leg_index]
                servo_position = raw_positions[servo_id - 1]
                
                angle_deviation = (self.servo_params.neutral_position - servo_position) / self.servo_params.micros_per_rad
                angle = (angle_deviation / self.servo_params.servo_multipliers[axis_index, leg_index] 
                         + self.servo_params.neutral_angles[axis_index, leg_index])
                
                joint_angles_hw[axis_index, leg_index] = angle
        
        # Convert to Isaac order
        joint_pos_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                joint_pos_isaac[isaac_leg * 3 + axis] = joint_angles_hw[axis, hw_col]
        
        return joint_pos_isaac, raw_positions
    
    def read_joint_efforts_raw(self):
        """Read raw servo load values (before any processing)."""
        raw_load = self.esp32.servos_get_load()
        if raw_load is None:
            return np.zeros(12), None
        
        raw_load = np.array(raw_load, dtype=float)
        
        # Reorder to Isaac order
        effort_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                effort_isaac[isaac_leg * 3 + axis] = raw_load[servo_id - 1]
        
        return effort_isaac, raw_load
    
    def read_joint_efforts_normalized(self):
        """Read normalized joint efforts."""
        effort_isaac, raw_load = self.read_joint_efforts_raw()
        
        # Get offset in Isaac order
        offset_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                offset_isaac[isaac_leg * 3 + axis] = self.load_offset[servo_id - 1]
        
        # Normalize
        centered = effort_isaac - offset_isaac
        normalized = centered / self.load_scale
        
        # Apply direction
        normalized = normalized * self.effort_direction
        
        return normalized, effort_isaac, raw_load
    
    # ==================== OBSERVATION BUILDING ====================
    
    def get_observation(self, velocity_command):
        """Build full 60-dim observation vector."""
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Read all sensors
        imu_data = self.esp32.imu_get_data()
        current_angles, raw_positions = self.read_joint_positions()
        effort_normalized, effort_raw_isaac, effort_raw = self.read_joint_efforts_normalized()
        
        # 1. Base linear velocity (estimated)
        base_lin_vel = velocity_command * 0.7
        
        # 2. Base angular velocity (from IMU)
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        
        # 3. Projected gravity (from IMU)
        accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel = accel_raw * self.accel_scale
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            projected_gravity = accel / accel_norm
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        # 4. Velocity commands
        velocity_commands = velocity_command.copy()
        
        # 5. Joint positions relative
        joint_pos_rel = current_angles - self.sim_default_positions
        
        # 6. Joint velocities
        if dt > 0.001:
            joint_vel = (current_angles - self.prev_joint_angles) / dt
            joint_vel = np.clip(joint_vel, -10, 10)
        else:
            joint_vel = np.zeros(12)
        
        # 7. Joint efforts (normalized)
        joint_effort = effort_normalized
        
        # 8. Previous actions
        prev_actions = self.prev_actions.copy()
        
        # Update state
        self.prev_joint_angles = current_angles.copy()
        self.prev_time = current_time
        
        # Build observation dict for logging
        obs_dict = {
            'time_step': len(self.observations),
            'timestamp': current_time,
            'dt': dt,
            
            # Base velocities
            'base_lin_vel_x': base_lin_vel[0],
            'base_lin_vel_y': base_lin_vel[1],
            'base_lin_vel_z': base_lin_vel[2],
            'base_ang_vel_x': base_ang_vel[0],
            'base_ang_vel_y': base_ang_vel[1],
            'base_ang_vel_z': base_ang_vel[2],
            
            # Gravity
            'projected_gravity_x': projected_gravity[0],
            'projected_gravity_y': projected_gravity[1],
            'projected_gravity_z': projected_gravity[2],
            
            # Commands
            'velocity_command_x': velocity_commands[0],
            'velocity_command_y': velocity_commands[1],
            'velocity_command_yaw': velocity_commands[2],
        }
        
        # Joint positions, velocities, efforts
        for i, name in enumerate(self.joint_names):
            obs_dict[f'joint_pos_{name}'] = joint_pos_rel[i]
            obs_dict[f'joint_vel_{name}'] = joint_vel[i]
            obs_dict[f'joint_effort_{name}'] = joint_effort[i]
            obs_dict[f'prev_action_{name}'] = prev_actions[i]
        
        # Raw sensor data for debugging
        raw_dict = {
            'time_step': len(self.observations),
            'gyro_raw_x': gyro_raw[0],
            'gyro_raw_y': gyro_raw[1],
            'gyro_raw_z': gyro_raw[2],
            'accel_raw_x': accel_raw[0],
            'accel_raw_y': accel_raw[1],
            'accel_raw_z': accel_raw[2],
        }
        
        # Add raw effort values
        for i, name in enumerate(self.joint_names):
            raw_dict[f'effort_raw_isaac_{name}'] = effort_raw_isaac[i]
        
        if effort_raw is not None:
            for i in range(12):
                raw_dict[f'effort_raw_servo_{i}'] = effort_raw[i]
        
        # Build flat observation vector
        obs_vector = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            velocity_commands,
            joint_pos_rel,
            joint_vel,
            joint_effort,
            prev_actions,
        ]).astype(np.float32)
        
        return obs_vector, obs_dict, raw_dict
    
    def log_step(self, velocity_command, action):
        """Log one step of observation and action."""
        obs_vector, obs_dict, raw_dict = self.get_observation(velocity_command)
        
        # Build action dict
        action_dict = {'time_step': len(self.actions)}
        for i, name in enumerate(self.joint_names):
            action_dict[f'action_{name}'] = action[i]
        
        self.observations.append(obs_dict)
        self.actions.append(action_dict)
        self.raw_sensor_data.append(raw_dict)
        
        # Update prev_actions
        self.prev_actions = action.copy()
        
        return obs_vector
    
    def save_data(self, prefix="movement"):
        """Save logged data to CSV files."""
        if len(self.observations) == 0:
            print("No data to save")
            return
        
        df_obs = pd.DataFrame(self.observations)
        df_act = pd.DataFrame(self.actions)
        df_raw = pd.DataFrame(self.raw_sensor_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        obs_file = os.path.join(self.output_dir, f'{prefix}_observations_{timestamp}.csv')
        act_file = os.path.join(self.output_dir, f'{prefix}_actions_{timestamp}.csv')
        raw_file = os.path.join(self.output_dir, f'{prefix}_raw_sensors_{timestamp}.csv')
        
        df_obs.to_csv(obs_file, index=False)
        df_act.to_csv(act_file, index=False)
        df_raw.to_csv(raw_file, index=False)
        
        print(f"Saved {len(df_obs)} samples:")
        print(f"  Observations: {obs_file}")
        print(f"  Actions: {act_file}")
        print(f"  Raw sensors: {raw_file}")
        
        return obs_file, act_file, raw_file
    
    def clear_data(self):
        """Clear logged data."""
        self.observations = []
        self.actions = []
        self.raw_sensor_data = []
        self.prev_actions = np.zeros(12)


# ==================== EFFORT DIRECTION TEST ====================

def test_effort_directions(logger):
    """
    Interactive test to determine correct effort direction signs.
    
    The test:
    1. For each leg, push DOWN on the thigh
    2. Record if effort goes positive or negative
    3. Compare with what simulation expects
    
    In simulation, when the robot supports its weight:
    - Thigh joints should show POSITIVE effort (resisting gravity)
    - If real robot shows NEGATIVE, flip the sign
    """
    print("\n" + "="*60)
    print("EFFORT DIRECTION TEST")
    print("="*60)
    print("""
This test determines if your servo load signs match simulation.

Instructions:
1. Robot should be in standing position
2. For each leg, you'll PUSH DOWN on the thigh
3. Watch if effort goes POSITIVE or NEGATIVE
4. In simulation, pushing down should cause POSITIVE effort
   (the servo resists by pushing back up)

Press Enter to start, Ctrl+C to stop at any time.
""")
    input()
    
    legs = ['LF', 'RF', 'LB', 'RB']
    joints = ['hip', 'thigh', 'calf']
    
    results = {}
    
    try:
        for leg_idx, leg in enumerate(legs):
            print(f"\n--- Testing {leg} leg ---")
            print(f"Push DOWN firmly on the {leg} THIGH and hold...")
            print("Press Enter when ready to record, then push.")
            input()
            
            print("Recording for 3 seconds... PUSH NOW!")
            
            effort_samples = []
            for _ in range(30):  # 3 seconds at ~10Hz
                effort_norm, effort_raw, _ = logger.read_joint_efforts_normalized()
                # Get thigh effort (index 1 for each leg)
                thigh_idx = leg_idx * 3 + 1
                effort_samples.append(effort_raw[thigh_idx])  # Use raw (before direction mult)
                time.sleep(0.1)
            
            # Analyze
            effort_samples = np.array(effort_samples)
            baseline = effort_samples[:5].mean()  # First 0.5s as baseline
            pushed = effort_samples[10:].mean()   # After 1s as pushed value
            delta = pushed - baseline
            
            print(f"\n{leg} Thigh Results:")
            print(f"  Baseline (resting): {baseline:.1f}")
            print(f"  When pushed: {pushed:.1f}")
            print(f"  Delta: {delta:.1f}")
            
            if delta > 50:
                print(f"  -> POSITIVE when pushed down (matches sim, keep sign as +1)")
                results[f'{leg}_thigh'] = 1.0
            elif delta < -50:
                print(f"  -> NEGATIVE when pushed down (opposite sim, flip sign to -1)")
                results[f'{leg}_thigh'] = -1.0
            else:
                print(f"  -> Inconclusive (delta too small). Try pushing harder.")
                results[f'{leg}_thigh'] = None
            
            print("\nPress Enter to continue to next leg...")
            input()
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    # Summary
    print("\n" + "="*60)
    print("EFFORT DIRECTION SUMMARY")
    print("="*60)
    print("\nRecommended effort_direction array:")
    print("(Based on thigh test - apply same to hip and calf per leg)")
    print()
    
    new_directions = logger.effort_direction.copy()
    for leg_idx, leg in enumerate(legs):
        key = f'{leg}_thigh'
        if key in results and results[key] is not None:
            # Apply same direction to all joints on this leg
            for j in range(3):
                new_directions[leg_idx * 3 + j] = results[key]
    
    print("self.effort_direction = np.array([")
    for leg_idx, leg in enumerate(legs):
        d = new_directions[leg_idx * 3: leg_idx * 3 + 3]
        print(f"    {d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f},   # {leg}")
    print("])")
    
    return new_directions


# ==================== LIVE EFFORT MONITOR ====================

def monitor_effort_live(logger, duration=30):
    """
    Monitor effort values live to see response to pushing/pulling.
    """
    print("\n" + "="*60)
    print("LIVE EFFORT MONITOR")
    print("="*60)
    print(f"Monitoring for {duration} seconds. Push on legs to see response.")
    print("Press Ctrl+C to stop.\n")
    
    legs = ['LF', 'RF', 'LB', 'RB']
    
    try:
        start = time.time()
        while time.time() - start < duration:
            effort_norm, effort_raw, raw_load = logger.read_joint_efforts_normalized()
            
            # Print raw values (before direction)
            output = "Raw: "
            for i, leg in enumerate(legs):
                e = effort_raw[i*3:(i+1)*3]
                output += f"{leg}:[{e[0]:+6.0f},{e[1]:+6.0f},{e[2]:+6.0f}] "
            
            print(f"\r{output}", end='', flush=True)
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        pass
    
    print("\n")


# ==================== SIMPLE WALKING SEQUENCE ====================

def record_standing_and_commands(logger, duration_per_command=3.0):
    """
    Record observations while giving simple velocity commands.
    Robot stays in place but policy would try to move.
    """
    print("\n" + "="*60)
    print("RECORDING STANDING + COMMAND SEQUENCE")
    print("="*60)
    
    commands = [
        ("Standing", np.array([0.0, 0.0, 0.0])),
        ("Forward", np.array([0.2, 0.0, 0.0])),
        ("Backward", np.array([-0.2, 0.0, 0.0])),
        ("Left", np.array([0.0, 0.2, 0.0])),
        ("Right", np.array([0.0, -0.2, 0.0])),
        ("Turn Left", np.array([0.0, 0.0, 0.2])),
        ("Turn Right", np.array([0.0, 0.0, -0.2])),
    ]
    
    logger.clear_data()
    
    for name, cmd in commands:
        print(f"\nRecording: {name} (cmd={cmd})")
        print(f"  Duration: {duration_per_command}s")
        
        start = time.time()
        while time.time() - start < duration_per_command:
            # Use zero actions (robot stands still)
            action = np.zeros(12)
            obs = logger.log_step(cmd, action)
            time.sleep(0.02)  # 50Hz
        
        print(f"  Samples: {len(logger.observations)}")
    
    # Save
    logger.save_data("standing_commands")


# ==================== POLICY-CONTROLLED RECORDING ====================

def record_with_policy(logger, policy_path, duration=10.0, velocity_cmd=None):
    """
    Record observations while running the actual policy.
    """
    print("\n" + "="*60)
    print("RECORDING WITH POLICY CONTROL")
    print("="*60)
    
    # Load policy
    try:
        policy = torch.jit.load(policy_path)
        policy.eval()
        print(f"Loaded policy: {policy_path}")
    except Exception as e:
        print(f"Failed to load policy: {e}")
        return
    
    if velocity_cmd is None:
        velocity_cmd = np.array([0.2, 0.0, 0.0])  # Forward
    
    print(f"Velocity command: {velocity_cmd}")
    print(f"Duration: {duration}s")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    logger.clear_data()
    
    # Hardware offset for calf
    calf_offset = 0.785
    sim_defaults = logger.sim_default_positions.copy()
    
    prev_actions = np.zeros(12)
    
    try:
        start = time.time()
        step = 0
        
        while time.time() - start < duration:
            loop_start = time.time()
            
            # Get observation
            obs, obs_dict, raw_dict = logger.get_observation(velocity_cmd)
            
            # Policy inference
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                raw_actions = policy(obs_tensor).squeeze().numpy()
            
            # Fade-in for first 50 steps
            if step < 50:
                fade = step / 50.0
            else:
                fade = 1.0
            
            faded_actions = raw_actions * fade
            
            # Compute target
            target = sim_defaults + faded_actions * 0.5  # ACTION_SCALE = 0.5
            target[2::3] += calf_offset  # Hardware offset
            
            # Send to hardware
            target_matrix = logger._isaac_to_hardware_matrix(target)
            logger.hardware.set_actuator_postions(target_matrix)
            
            # Log
            logger.observations.append(obs_dict)
            action_dict = {'time_step': step}
            for i, name in enumerate(logger.joint_names):
                action_dict[f'action_{name}'] = faded_actions[i]
            logger.actions.append(action_dict)
            logger.raw_sensor_data.append(raw_dict)
            
            # Update state
            logger.prev_actions = faded_actions.copy()
            
            step += 1
            
            # Maintain 50Hz
            elapsed = time.time() - loop_start
            if elapsed < 0.02:
                time.sleep(0.02 - elapsed)
        
        print(f"\nRecorded {step} steps")
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save
    logger.save_data("policy_controlled")


def _isaac_to_hardware_matrix(self, flat_angles_radians):
    """Convert Isaac flat array to hardware matrix."""
    matrix = np.zeros((3, 4))
    matrix[:, 1] = flat_angles_radians[0:3]   # LF
    matrix[:, 0] = flat_angles_radians[3:6]   # RF
    matrix[:, 3] = flat_angles_radians[6:9]   # LB
    matrix[:, 2] = flat_angles_radians[9:12]  # RB
    return matrix

# Add method to logger class
RealRobotObservationLogger._isaac_to_hardware_matrix = _isaac_to_hardware_matrix


# ==================== COMPARISON WITH SIMULATION ====================

def compare_with_simulation(real_csv_path, sim_csv_path):
    """
    Compare real robot observations with simulation.
    """
    print("\n" + "="*60)
    print("REAL vs SIMULATION COMPARISON")
    print("="*60)
    
    df_real = pd.read_csv(real_csv_path)
    df_sim = pd.read_csv(sim_csv_path)
    
    print(f"Real samples: {len(df_real)}")
    print(f"Sim samples: {len(df_sim)}")
    
    # Compare key observations
    obs_to_compare = [
        'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z',
        'projected_gravity_x', 'projected_gravity_y', 'projected_gravity_z',
        'velocity_command_x', 'velocity_command_y', 'velocity_command_yaw',
    ]
    
    print("\n" + "-"*60)
    print(f"{'Observation':<25} {'Real Mean':>12} {'Real Std':>10} {'Sim Mean':>12} {'Sim Std':>10}")
    print("-"*60)
    
    for obs in obs_to_compare:
        if obs in df_real.columns and obs in df_sim.columns:
            real_mean = df_real[obs].mean()
            real_std = df_real[obs].std()
            sim_mean = df_sim[obs].mean()
            sim_std = df_sim[obs].std()
            print(f"{obs:<25} {real_mean:>12.4f} {real_std:>10.4f} {sim_mean:>12.4f} {sim_std:>10.4f}")
    
    # Joint efforts comparison
    print("\n" + "-"*60)
    print("Joint Effort Comparison:")
    print("-"*60)
    
    joint_names = [
        'base_lf1', 'lf1_lf2', 'lf2_lf3',
        'base_rf1', 'rf1_rf2', 'rf2_rf3',
        'base_lb1', 'lb1_lb2', 'lb2_lb3',
        'base_rb1', 'rb1_rb2', 'rb2_rb3'
    ]
    
    for name in joint_names:
        col = f'joint_effort_{name}'
        if col in df_real.columns and col in df_sim.columns:
            real_mean = df_real[col].mean()
            sim_mean = df_sim[col].mean()
            sign_match = "✓" if (real_mean * sim_mean > 0 or abs(real_mean) < 0.01) else "✗ SIGN MISMATCH"
            print(f"  {name:<15}: Real={real_mean:+.3f}, Sim={sim_mean:+.3f}  {sign_match}")


# ==================== MAIN ====================

if __name__ == "__main__":
    logger = RealRobotObservationLogger()
    
    print("\n" + "="*60)
    print("REAL ROBOT OBSERVATION LOGGER")
    print("="*60)
    print("""
Options:
  1. Test effort directions (interactive)
  2. Monitor effort live
  3. Record standing + commands (no movement)
  4. Record with policy control
  5. Compare with simulation data
  x. Exit
""")
    
    while True:
        cmd = input("> ").strip().lower()
        
        if cmd == '1':
            new_dirs = test_effort_directions(logger)
            print("\nApply these directions? (y/n): ", end='')
            if input().strip().lower() == 'y':
                logger.effort_direction = new_dirs
                print("Applied!")
        
        elif cmd == '2':
            duration = input("Duration (seconds, default 30): ").strip()
            duration = float(duration) if duration else 30
            monitor_effort_live(logger, duration)
        
        elif cmd == '3':
            record_standing_and_commands(logger)
        
        elif cmd == '4':
            policy_path = input("Policy path (default /home/ubuntu/mp2_mlp/policy_joyboy.pt): ").strip()
            if not policy_path:
                policy_path = "/home/ubuntu/mp2_mlp/policy_joyboy.pt"
            
            vx = input("Forward velocity (default 0.2): ").strip()
            vx = float(vx) if vx else 0.2
            
            record_with_policy(logger, policy_path, duration=10.0, 
                             velocity_cmd=np.array([vx, 0.0, 0.0]))
        
        elif cmd == '5':
            real_path = input("Real robot CSV path: ").strip()
            sim_path = input("Simulation CSV path: ").strip()
            compare_with_simulation(real_path, sim_path)
        
        elif cmd == 'x':
            break
        
        else:
            print("Unknown command")
    
    print("Done!")
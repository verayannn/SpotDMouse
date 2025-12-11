import numpy as np
import pandas as pd
import torch
import time
import os
from datetime import datetime
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration


class EffortDirectionTester:
    """
    Simple test to determine correct effort direction signs by comparing
    what happens when you push on the robot vs what simulation expects.
    """
    
    def __init__(self):
        print("Initializing hardware...")
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.pwm_params = self.hardware.pwm_params
        self.servo_params = self.hardware.servo_params
        self.esp32 = self.pwm_params.esp32
        
        # Hardware mapping: hw_col -> isaac_leg
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}  # FR->RF, FL->LF, BR->RB, BL->LB
        
        # Calibrate load offset
        print("Calibrating load sensors (keep robot still)...")
        self.load_offset = self._calibrate_load()
        print(f"Load offset: {self.load_offset.astype(int)}")
        print("Ready!\n")
    
    def _calibrate_load(self, samples=30):
        """Get resting load values."""
        load_samples = []
        for _ in range(samples):
            load = self.esp32.servos_get_load()
            if load:
                load_samples.append(load)
            time.sleep(0.02)
        
        if load_samples:
            return np.mean(load_samples, axis=0)
        return np.zeros(12)
    
    def read_raw_effort_isaac_order(self):
        """Read raw load values, reordered to Isaac format."""
        raw_load = self.esp32.servos_get_load()
        if raw_load is None:
            return np.zeros(12), None
        
        raw_load = np.array(raw_load, dtype=float)
        
        # Reorder to Isaac: [LF, RF, LB, RB]
        effort_isaac = np.zeros(12)
        offset_isaac = np.zeros(12)
        
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                effort_isaac[isaac_leg * 3 + axis] = raw_load[servo_id - 1]
                offset_isaac[isaac_leg * 3 + axis] = self.load_offset[servo_id - 1]
        
        # Remove resting offset
        centered = effort_isaac - offset_isaac
        
        return centered, raw_load
    
    def monitor_live(self, duration=60):
        """
        Monitor effort values live.
        Push on legs to see how values change.
        """
        print("="*70)
        print("LIVE EFFORT MONITOR")
        print("="*70)
        print("Push on each leg and watch the values change.")
        print("Format: LF=[hip, thigh, calf] RF=[...] LB=[...] RB=[...]")
        print("Press Ctrl+C to stop.\n")
        
        legs = ['LF', 'RF', 'LB', 'RB']
        
        try:
            start = time.time()
            while time.time() - start < duration:
                effort, _ = self.read_raw_effort_isaac_order()
                
                output = ""
                for i, leg in enumerate(legs):
                    e = effort[i*3:(i+1)*3]
                    output += f"{leg}=[{e[0]:+6.0f},{e[1]:+6.0f},{e[2]:+6.0f}] "
                
                print(f"\r{output}", end='', flush=True)
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            pass
        
        print("\n")
    
    def test_single_joint(self, leg_name, joint_name):
        """
        Test a single joint's effort direction.
        
        In simulation, when you push DOWN on a leg:
        - The servo resists by pushing UP
        - This should register as POSITIVE effort
        
        If real robot shows NEGATIVE, the sign needs to be flipped.
        """
        leg_map = {'LF': 0, 'RF': 1, 'LB': 2, 'RB': 3}
        joint_map = {'hip': 0, 'thigh': 1, 'calf': 2}
        
        if leg_name not in leg_map or joint_name not in joint_map:
            print(f"Invalid leg '{leg_name}' or joint '{joint_name}'")
            return None
        
        joint_idx = leg_map[leg_name] * 3 + joint_map[joint_name]
        
        print(f"\nTesting {leg_name} {joint_name} (index {joint_idx})")
        print("="*50)
        print("1. Robot should be standing normally")
        print("2. When prompted, PUSH DOWN firmly on the leg")
        print("3. Hold for 2 seconds")
        print()
        
        # Baseline measurement
        print("Recording baseline (don't touch)...")
        baseline_samples = []
        for _ in range(20):
            effort, _ = self.read_raw_effort_isaac_order()
            baseline_samples.append(effort[joint_idx])
            time.sleep(0.05)
        baseline = np.mean(baseline_samples)
        baseline_std = np.std(baseline_samples)
        print(f"Baseline: {baseline:.1f} ± {baseline_std:.1f}")
        
        # Wait for user
        input("\nPress Enter, then PUSH DOWN on the leg...")
        
        # Pushed measurement
        print("Recording pushed values...")
        pushed_samples = []
        for _ in range(40):  # 2 seconds
            effort, _ = self.read_raw_effort_isaac_order()
            pushed_samples.append(effort[joint_idx])
            time.sleep(0.05)
        pushed = np.mean(pushed_samples)
        pushed_std = np.std(pushed_samples)
        print(f"Pushed: {pushed:.1f} ± {pushed_std:.1f}")
        
        # Analysis
        delta = pushed - baseline
        print(f"\nDelta: {delta:+.1f}")
        
        if abs(delta) < 50:
            print("⚠️  Delta too small - try pushing harder")
            return None
        elif delta > 0:
            print("✓ POSITIVE when pushed down → Sign is CORRECT (+1)")
            return 1.0
        else:
            print("✗ NEGATIVE when pushed down → Sign needs FLIP (-1)")
            return -1.0
    
    def run_full_direction_test(self):
        """
        Test all joints and generate the effort_direction array.
        """
        print("="*70)
        print("FULL EFFORT DIRECTION TEST")
        print("="*70)
        print("""
This will test each leg to determine correct effort signs.

For each leg:
1. You'll push DOWN on the thigh
2. We measure if effort goes positive or negative
3. In simulation, pushing down should cause POSITIVE effort

The same sign typically applies to all joints on a leg.
        """)
        
        input("Press Enter to begin...")
        
        legs = ['LF', 'RF', 'LB', 'RB']
        results = {}
        
        for leg in legs:
            print(f"\n{'='*50}")
            print(f"Testing {leg} leg")
            print(f"{'='*50}")
            
            result = self.test_single_joint(leg, 'thigh')
            results[leg] = result if result else 1.0  # Default to +1 if inconclusive
            
            if leg != legs[-1]:
                input("\nPress Enter to continue to next leg...")
        
        # Generate array
        print("\n" + "="*70)
        print("RESULTS - Copy this to your controller:")
        print("="*70)
        
        print("\nself.effort_direction = np.array([")
        for leg in legs:
            sign = results[leg]
            # Apply same sign to all joints on this leg
            print(f"    {sign:+.1f}, {sign:+.1f}, {sign:+.1f},   # {leg}: hip, thigh, calf")
        print("])")
        
        return results


class PolicyDataRecorder:
    """
    Records full 60-dim observations while running the RL policy.
    """
    
    def __init__(self, output_dir="policy_obs_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Hardware
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.pwm_params = self.hardware.pwm_params
        self.servo_params = self.hardware.servo_params
        self.esp32 = self.pwm_params.esp32
        
        # Sim defaults
        self.sim_default_positions = np.array([
            0.0, 0.785, -1.57,  # LF
            0.0, 0.785, -1.57,  # RF
            0.0, 0.785, -1.57,  # LB
            0.0, 0.785, -1.57,  # RB
        ])
        
        # Hardware mapping
        self.hw_to_isaac_leg = {0: 1, 1: 0, 2: 3, 3: 2}
        
        # Calibration
        self.gyro_offset = np.zeros(3)
        self.load_offset = np.zeros(12)
        
        # Effort direction (UPDATE THIS after running direction test)
        self.effort_direction = np.array([
            1.0, 1.0, 1.0,  # LF
            1.0, 1.0, 1.0,  # RF
            1.0, 1.0, 1.0,  # LB
            1.0, 1.0, 1.0,  # RB
        ])
        
        # Scaling
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81
        self.load_scale = 5000.0
        
        # Hardware offset
        self.calf_offset = 0.785  # Difference between sim (-1.57) and real (-0.785) neutral
        
        # State
        self.prev_actions = np.zeros(12)
        self.prev_joint_angles = self.sim_default_positions.copy()
        self.prev_time = time.time()
        
        # Data storage
        self.data = []
        
        print("Calibrating...")
        self._calibrate()
        print("Ready!")
    
    def _calibrate(self, samples=30):
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
        if load_samples:
            self.load_offset = np.mean(load_samples, axis=0)
    
    def _read_joint_positions(self):
        raw = self.esp32.servos_get_position()
        if raw is None:
            return self.prev_joint_angles.copy()
        
        angles_hw = np.zeros((3, 4))
        for leg in range(4):
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, leg]
                pos = raw[servo_id - 1]
                dev = (self.servo_params.neutral_position - pos) / self.servo_params.micros_per_rad
                angle = dev / self.servo_params.servo_multipliers[axis, leg] + self.servo_params.neutral_angles[axis, leg]
                angles_hw[axis, leg] = angle
        
        angles_isaac = np.zeros(12)
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                angles_isaac[isaac_leg * 3 + axis] = angles_hw[axis, hw_col]
        
        return angles_isaac
    
    def _read_joint_efforts(self):
        raw = self.esp32.servos_get_load()
        if raw is None:
            return np.zeros(12)
        
        raw = np.array(raw, dtype=float)
        effort = np.zeros(12)
        offset = np.zeros(12)
        
        for hw_col, isaac_leg in self.hw_to_isaac_leg.items():
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col]
                effort[isaac_leg * 3 + axis] = raw[servo_id - 1]
                offset[isaac_leg * 3 + axis] = self.load_offset[servo_id - 1]
        
        centered = effort - offset
        normalized = (centered / self.load_scale) * self.effort_direction
        return normalized
    
    def _get_observation(self, velocity_cmd):
        now = time.time()
        dt = now - self.prev_time
        
        imu = self.esp32.imu_get_data()
        angles = self._read_joint_positions()
        effort = self._read_joint_efforts()
        
        # Base velocities
        base_lin_vel = velocity_cmd * 0.7
        gyro = np.array([imu['gx'], imu['gy'], imu['gz']])
        base_ang_vel = (gyro - self.gyro_offset) * self.gyro_scale
        
        # Gravity
        accel = np.array([imu['ax'], imu['ay'], imu['az']]) * self.accel_scale
        norm = np.linalg.norm(accel)
        gravity = accel / norm if norm > 0.1 else np.array([0, 0, -1])
        
        # Joint states
        joint_pos_rel = angles - self.sim_default_positions
        joint_vel = (angles - self.prev_joint_angles) / dt if dt > 0.001 else np.zeros(12)
        
        self.prev_joint_angles = angles.copy()
        self.prev_time = now
        
        obs = np.concatenate([
            base_lin_vel, base_ang_vel, gravity, velocity_cmd,
            joint_pos_rel, joint_vel, effort, self.prev_actions
        ]).astype(np.float32)
        
        return obs
    
    def _isaac_to_hw_matrix(self, angles):
        m = np.zeros((3, 4))
        m[:, 1] = angles[0:3]   # LF
        m[:, 0] = angles[3:6]   # RF
        m[:, 3] = angles[6:9]   # LB
        m[:, 2] = angles[9:12]  # RB
        return m
    
    def record_with_policy(self, policy_path, velocity_cmd, duration=10.0):
        """Run policy and record all data."""
        print(f"\nLoading policy: {policy_path}")
        policy = torch.jit.load(policy_path)
        policy.eval()
        
        print(f"Velocity command: {velocity_cmd}")
        print(f"Duration: {duration}s")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        self.data = []
        self.prev_actions = np.zeros(12)
        
        try:
            start = time.time()
            step = 0
            
            while time.time() - start < duration:
                loop_start = time.time()
                
                obs = self._get_observation(velocity_cmd)
                
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    raw_actions = policy(obs_t).squeeze().numpy()
                
                # Fade in
                fade = min(step / 50.0, 1.0)
                faded = raw_actions * fade
                
                self.prev_actions = faded.copy()
                
                # Compute target
                target = self.sim_default_positions + faded * 0.5
                target[2::3] += self.calf_offset
                
                # Send
                self.hardware.set_actuator_postions(self._isaac_to_hw_matrix(target))
                
                # Log
                self.data.append({
                    'step': step,
                    'time': time.time() - start,
                    **{f'obs_{i}': obs[i] for i in range(60)},
                    **{f'action_{i}': faded[i] for i in range(12)},
                    **{f'raw_action_{i}': raw_actions[i] for i in range(12)},
                })
                
                step += 1
                
                # 50Hz
                elapsed = time.time() - loop_start
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)
            
            print(f"\nRecorded {step} steps")
            
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        # Save
        df = pd.DataFrame(self.data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"policy_data_{timestamp}.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved to: {filepath}")
        
        return filepath


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EFFORT DIRECTION & DATA RECORDING TOOL")
    print("="*60)
    print("""
Options:
  1. Test effort directions (determine correct signs)
  2. Monitor effort values live
  3. Record data while running policy
  x. Exit
""")
    
    while True:
        cmd = input("> ").strip()
        
        if cmd == '1':
            tester = EffortDirectionTester()
            tester.run_full_direction_test()
        
        elif cmd == '2':
            tester = EffortDirectionTester()
            tester.monitor_live(duration=120)
        
        elif cmd == '3':
            recorder = PolicyDataRecorder()
            policy = input("Policy path [/home/ubuntu/mp2_mlp/policy_joyboy.pt]: ").strip()
            if not policy:
                policy = "/home/ubuntu/mp2_mlp/policy_joyboy.pt"
            
            vx = input("Forward velocity [0.2]: ").strip()
            vx = float(vx) if vx else 0.2
            
            recorder.record_with_policy(
                policy, 
                np.array([vx, 0.0, 0.0]),
                duration=10.0
            )
        
        elif cmd.lower() == 'x':
            break
        
        else:
            print("Unknown command")
    
    print("Done!")
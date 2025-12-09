import numpy as np
import torch
import time
import threading
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

class BlindController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 60)
        print("Initializing Blind Controller (With Calf Correction)")
        print("   - Writing: HardwareInterface")
        print("   - Reading: BLIND (Open Loop)")
        print("   - Fixes: Offsets Calf by +0.77 rads to match Sim")
        print("=" * 60)
        
        # 1. Hardware
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.esp32 = self.hardware.pwm_params.esp32
        time.sleep(0.5)
        
        # 2. Policy
        try:
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            print(f"[OK] Loaded policy from {policy_path}")
        except Exception as e:
            print(f"[ERROR] Could not load policy: {e}")
            self.shutdown = True
            return
        
        # 3. Simulation Defaults
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])
        
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # 4. Tuning
        self.ACTION_SCALE = 0.50
        self.action_smoothing = 0.5   
        self.target_smoothing = 0.3   
        
        # --- THE CALF FIX ---
        # Your sensor test showed a difference of ~0.77 rads 
        # between Sim (-1.57) and Real Standing (-0.80).
        # We apply this offset so the robot handles Sim commands while standing tall.
        self.calf_offset = 0.77
        
        # IMU Config
        self.gyro_z_sign = -1.0 
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81

        # 5. INITIALIZATION
        # We calculate the "Real Standing Pose" by applying the offset to the Sim Default.
        # This ensures we start exactly where the robot is standing.
        
        start_pose = self.sim_default_positions.copy()
        # Add offset to Calves (Indices 2, 5, 8, 11)
        start_pose[2::3] += self.calf_offset
        
        print(f"Sim Default Calf: {self.sim_default_positions[2]:.2f}")
        print(f"Corrected Start Calf: {start_pose[2]:.2f} (Should be ~ -0.80)")
        
        self.prev_target_positions = start_pose.copy()
        self.prev_joint_angles = start_pose.copy()
        self.prev_actions = np.zeros(12)
        self.prev_time = time.time()
        
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        self.debug_counter = 0
        self.startup_steps = 0
        self.startup_duration = 50 
        self.CONTROL_FREQUENCY = 50

        print("[CALIBRATION] Calibrating IMU...")
        self._calibrate_imu()

    def _calibrate_imu(self, samples=50):
        gyro_samples = []
        for _ in range(samples):
            data = self.esp32.imu_get_data()
            if data:
                gyro_samples.append([data['gx'], data['gy'], data['gz']])
            time.sleep(0.02)
        if len(gyro_samples) > 0:
            self.gyro_offset = np.mean(gyro_samples, axis=0)
        else:
            self.gyro_offset = np.zeros(3)

    def _isaac_to_hardware_matrix(self, flat_angles_radians):
        matrix = np.zeros((3, 4))
        matrix[:, 1] = flat_angles_radians[0:3] # LF
        matrix[:, 0] = flat_angles_radians[3:6] # RF
        matrix[:, 3] = flat_angles_radians[6:9] # LB
        matrix[:, 2] = flat_angles_radians[9:12] # RB
        return matrix

    def get_observation(self):
        current_time = time.time()
        
        # 1. READ SENSORS (IMU ONLY)
        imu_data = self.esp32.imu_get_data()
        
        # 2. JOINT POSITIONS = TARGET (BLIND MODE)
        # IMPORTANT: The policy expects the joints to be relative to SIM DEFAULT (-1.57).
        # But our physical legs are at REAL STANDING (-0.80).
        # We must subtract the offset before showing it to the policy, 
        # otherwise the policy thinks the legs are hyper-extended.
        
        real_current_angles = self.prev_target_positions.copy()
        sim_compatible_angles = real_current_angles.copy()
        sim_compatible_angles[2::3] -= self.calf_offset
        
        # 3. Build Obs
        base_lin_vel = self.velocity_command * 0.7
        
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = (gyro_raw - self.gyro_offset) * self.gyro_scale
        base_ang_vel[2] *= self.gyro_z_sign
        
        accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel = accel_raw * self.accel_scale
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1: projected_gravity = accel / accel_norm
        else: projected_gravity = np.array([0.0, 0.0, -1.0])
        
        velocity_commands = self.velocity_command.copy()
        
        # Rel positions are calculated against SIM DEFAULTS
        joint_pos_rel = sim_compatible_angles - self.sim_default_positions
        
        joint_vel = np.zeros(12, dtype=np.float32)
        joint_effort = np.zeros(12, dtype=np.float32)
        prev_actions = self.prev_actions.copy()
        
        self.prev_joint_angles = sim_compatible_angles.copy()
        self.prev_time = current_time
        
        obs = np.concatenate([
            base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
            joint_pos_rel, joint_vel, joint_effort, prev_actions
        ]).astype(np.float32)
        
        return obs

    def control_step(self):
        obs = self.get_observation()
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
            
        clipped_actions = np.clip(raw_actions, -1.0, 1.0)
        
        if self.startup_steps < self.startup_duration:
            fade_in = self.startup_steps / self.startup_duration
            self.startup_steps += 1
        else:
            fade_in = 1.0
        faded_actions = clipped_actions * fade_in
        
        # 1. Action Smoothing
        alpha = self.action_smoothing
        smoothed_actions = (alpha * faded_actions) + ((1.0 - alpha) * self.prev_actions)
        self.prev_actions = smoothed_actions.copy()
        
        # 2. Compute Target (SIM FRAME)
        policy_target_sim = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        
        # 3. Apply CALF OFFSET (Convert Sim -> Real)
        policy_target_real = policy_target_sim.copy()
        policy_target_real[2::3] += self.calf_offset
        
        # Clip (be careful with limits, we shifted the range)
        # We relax the lower limit for calves because we shifted them up
        policy_target_real = np.clip(policy_target_real, -2.0, 2.0)
        
        # 4. Target Smoothing (Slew Rate)
        beta = self.target_smoothing
        final_target = (beta * policy_target_real) + ((1.0 - beta) * self.prev_target_positions)
        
        # STORE TARGET FOR NEXT OBS
        self.prev_target_positions = final_target.copy()
        
        # 5. Write
        target_matrix = self._isaac_to_hardware_matrix(final_target)
        self.hardware.set_actuator_postions(target_matrix)
        
        self.debug_counter += 1
        if self.debug_counter % 25 == 0:
            print(f"Act: [{smoothed_actions.min():+.2f}, {smoothed_actions.max():+.2f}] | "
                  f"Calf Tgt: {final_target[2]:.2f}")

    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        print(f"\nRunning Blind Controller at {self.CONTROL_FREQUENCY}Hz")
        
        while not self.shutdown:
            loop_start = time.time()
            if self.control_active:
                try:
                    self.control_step()
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    self.control_active = False
            
            elapsed = time.time() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)

    def set_velocity_command(self, vx, vy, vyaw):
        self.velocity_command = np.array([vx, vy, vyaw])
        if np.any(np.abs(self.velocity_command) > 0.01):
            self.control_active = True
            print(f"CMD: {self.velocity_command}")
        else:
            self.control_active = False
            print("STOPPED")

    def stop(self):
        self.control_active = False
        self.shutdown = True

if __name__ == "__main__":
    controller = BlindController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    thread = threading.Thread(target=controller.control_loop)
    thread.start()
    
    print("Commands: w/s/a/d/q/e, space=stop, x=exit")
    cur_speed = 0.2
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'w': controller.set_velocity_command(cur_speed, 0, 0)
            elif cmd == 's': controller.set_velocity_command(-cur_speed, 0, 0)
            elif cmd == 'a': controller.set_velocity_command(0, cur_speed, 0)
            elif cmd == 'd': controller.set_velocity_command(0, -cur_speed, 0)
            elif cmd == 'q': controller.set_velocity_command(0, 0, cur_speed)
            elif cmd == 'e': controller.set_velocity_command(0, 0, -cur_speed)
            elif cmd in [' ','']: controller.set_velocity_command(0, 0, 0)
            elif cmd == 'x': break
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    thread.join()
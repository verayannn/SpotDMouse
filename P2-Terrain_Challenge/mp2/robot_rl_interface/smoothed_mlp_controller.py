import numpy as np
import torch
import time
import threading
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

class SimFrameController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        print("=" * 60)
        print("Initializing Sim-Frame Blind Controller")
        print("   - Brain: Lives in Matrix (Think's it is squatting)")
        print("   - Body:  Lives in Reality (Offset applied at output)")
        print("   - Sensors: IGNORED (Prevents freezing)")
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
        
        # 3. Simulation Defaults (The "Squat" Pose)
        self.sim_default_positions = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57,  # RB
        ])
        
        # Limits (Sim Frame)
        self.joint_lower_limits = np.array([-0.5, 0.0, -2.5] * 4)
        self.joint_upper_limits = np.array([0.5, 1.5, -0.5] * 4)
        
        # 4. Tuning
        # We increase scale slightly to break static friction (stiction)
        self.ACTION_SCALE = 0.60
        self.action_smoothing = 0.5   
        self.target_smoothing = 0.2   
        
        # --- THE MAGIC OFFSET ---
        # We apply this ONLY when writing to hardware.
        # The brain never knows this exists.
        self.calf_hardware_offset = 0.60
        
        # IMU Config
        self.gyro_z_sign = -1.0 
        self.gyro_scale = np.pi / 180.0
        self.accel_scale = 9.81

        # 5. INITIALIZATION
        # Start in Sim Frame (Squat)
        self.prev_target_positions = self.sim_default_positions.copy()
        self.prev_joint_angles = self.sim_default_positions.copy()
        self.prev_actions = np.zeros(12)
        self.prev_joint_vel = np.zeros(12) # Blind velocity
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
        
        # 2. JOINT POSITIONS = PREVIOUS TARGET (PURE BLIND)
        # We feed back exactly what the brain commanded last time.
        # The brain thinks it is controlling a perfect simulation.
        current_angles = self.prev_target_positions.copy()
        
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
        joint_pos_rel = current_angles - self.sim_default_positions
        
        # Zero Velocity (Perfect Sim assumption)
        joint_vel = np.zeros(12, dtype=np.float32)
        joint_effort = np.zeros(12, dtype=np.float32)
        prev_actions = self.prev_actions.copy()
        
        self.prev_joint_angles = current_angles.copy()
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
        
        # 2. Compute Target (SIM FRAME - SQUAT)
        sim_target = self.sim_default_positions + smoothed_actions * self.ACTION_SCALE
        sim_target = np.clip(sim_target, self.joint_lower_limits, self.joint_upper_limits)
        
        # 3. Target Smoothing (Slew Rate)
        beta = self.target_smoothing
        final_sim_target = (beta * sim_target) + ((1.0 - beta) * self.prev_target_positions)
        
        # STORE SIM TARGET (This is what we feed back to the brain)
        self.prev_target_positions = final_sim_target.copy()
        
        # 4. HARDWARE OFFSET (The Secret Sauce)
        # We take the Sim (Squat) target and add the offset just before sending.
        hardware_target = final_sim_target.copy()
        hardware_target[2::3] += self.calf_hardware_offset
        
        # 5. Write to Hardware
        target_matrix = self._isaac_to_hardware_matrix(hardware_target)
        self.hardware.set_actuator_postions(target_matrix)
        
        self.debug_counter += 1
        if self.debug_counter % 25 == 0:
            print(f"Act: [{smoothed_actions.min():+.2f}, {smoothed_actions.max():+.2f}]")

    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        print(f"\nRunning Sim-Frame Blind Controller at {self.CONTROL_FREQUENCY}Hz")
        
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
    controller = SimFrameController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
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
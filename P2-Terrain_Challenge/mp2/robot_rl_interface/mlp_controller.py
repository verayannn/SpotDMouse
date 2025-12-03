import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class MatchedMLPController:
    """
    Controller tuned to match simulation dynamics more closely.
    """
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/latest_mlp_network/policy_only.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")
        
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        self.direction_multipliers = np.array([
            -1.0, -1.0, -1.0,
            -1.0,  1.0,  1.0,
             1.0, -1.0, -1.0,
             1.0,  1.0,  1.0,
        ])
        
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Match training: action_scale = 0.5
        self.ACTION_SCALE = 0.5
        
        print("Recording standing pose...")
        raw = np.array(self.esp32.servos_get_position())
        self.standing_servos = raw[self.esp32_servo_order].astype(float)
        print(f"Standing servos: {self.standing_servos}")
        
        # State
        self.prev_actions = np.zeros(12)
        self.prev_joint_pos_rel = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        
        print("Calibrating IMU...")
        self._calibrate_imu()
        
        # MATCHED TO SIMULATION RANGES
        self.CONTROL_FREQUENCY = 50  # Match training (decimation=10, dt=0.002 → 50Hz)
        
        # Based on sim data: actions range ±1.5, joint_pos_rel ranges ±0.25
        self.action_clip = 1.5           # Match sim action range
        self.obs_joint_pos_clip = 0.8#0.4    # Slightly wider than sim's ±0.25 for safety
        self.obs_joint_vel_clip = 1.5    # Match sim's joint velocity range
        self.obs_ang_vel_clip = 2.0      # Match sim's angular velocity range
        
        # Less aggressive filtering to allow gait to develop
        self.action_smoothing = 0.6      # More responsive (let more of raw action through)
        self.max_action_delta = 0.25     # Allow faster changes to match gait frequency
        
        # Startup ramp
        self.startup_steps = 0
        self.startup_duration = 25       # 0.5 second ramp at 50Hz
        
        self.debug_counter = 0

        print("Measuring policy rest bias...")
        self._measure_rest_bias()
    
    def _measure_rest_bias(self):
        """Measure what the policy outputs for a perfect standing observation."""
        # Create perfect standing observation
        rest_obs = np.zeros(60, dtype=np.float32)
        rest_obs[6:9] = [0, 0, -1]  # projected_gravity
        
        with torch.no_grad():
            self.rest_bias = self.policy(torch.tensor(rest_obs).unsqueeze(0)).squeeze().numpy()
        
        print(f"Rest bias: [{self.rest_bias.min():.3f}, {self.rest_bias.max():.3f}]")
        print(f"  LF: [{self.rest_bias[0]:+.2f}, {self.rest_bias[1]:+.2f}, {self.rest_bias[2]:+.2f}]")
        print(f"  RF: [{self.rest_bias[3]:+.2f}, {self.rest_bias[4]:+.2f}, {self.rest_bias[5]:+.2f}]")
        print(f"  LB: [{self.rest_bias[6]:+.2f}, {self.rest_bias[7]:+.2f}, {self.rest_bias[8]:+.2f}]")
        print(f"  RB: [{self.rest_bias[9]:+.2f}, {self.rest_bias[10]:+.2f}, {self.rest_bias[11]:+.2f}]")

        
    def _calibrate_imu(self, samples=50):
        print("Keep robot still...")
        gyro_sum = np.zeros(3)
        for _ in range(samples):
            data = self.esp32.imu_get_data()
            gyro_sum += np.array([data['gx'], data['gy'], data['gz']])
            time.sleep(0.02)
        self.gyro_offset = gyro_sum / samples
        print(f"Gyro offset: {self.gyro_offset}")
    
    def read_joint_positions(self):
        raw = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw[self.esp32_servo_order].astype(float)
        servo_delta = isaac_ordered - self.standing_servos
        delta_radians = servo_delta / self.servo_scale
        joint_pos_rel = delta_radians * self.direction_multipliers
        return joint_pos_rel
    
    def write_joint_positions(self, actions):
        target_pos_rel = actions * self.ACTION_SCALE
        delta_radians = target_pos_rel * self.direction_multipliers
        servo_delta = delta_radians * self.servo_scale
        target_servos = self.standing_servos + servo_delta
        target_servos = np.clip(target_servos, 180, 844)
        
        esp32_out = np.zeros(12)
        esp32_out[self.esp32_servo_order] = target_servos
        self.esp32.servos_set_position([int(p) for p in esp32_out])
    
    def get_observation(self):
        imu_data = self.esp32.imu_get_data()
        
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = gyro_raw - self.gyro_offset
        ####
        base_ang_vel = np.zeros(3)
        base_ang_vel = np.clip(base_ang_vel, -self.obs_ang_vel_clip, self.obs_ang_vel_clip)
        
        accel = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            projected_gravity = accel / accel_norm
        else:
            projected_gravity = np.array([0, 0, -1])

        ###    
        projected_gravity = np.array([0.0, 0.0, -1.0])
        
        base_lin_vel = np.zeros(3)
        
        joint_pos_rel_raw = self.read_joint_positions()
        
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0.001:
            joint_vel_raw = (joint_pos_rel_raw - self.prev_joint_pos_rel) / dt
            alpha = 0.5
            joint_vel = alpha * joint_vel_raw + (1 - alpha) * self.prev_joint_vel
            joint_vel = np.clip(joint_vel, -self.obs_joint_vel_clip, self.obs_joint_vel_clip)
        else:
            joint_vel = self.prev_joint_vel
        
        self.prev_joint_pos_rel = joint_pos_rel_raw.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        joint_pos_rel_clipped = np.clip(joint_pos_rel_raw, -self.obs_joint_pos_clip, self.obs_joint_pos_clip)
        
        joint_effort = np.zeros(12)
        
        obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            self.velocity_command,
            joint_pos_rel_clipped,
            joint_vel,
            joint_effort,
            self.prev_actions
        ]).astype(np.float32)
        
        return obs, joint_pos_rel_raw, joint_vel
    
    def control_step(self):
        if hasattr(self, 'test_mode') and self.test_mode:
            # Bypass policy entirely - just send zeros
            actions = np.zeros(12)
            self.write_joint_positions(actions)
            self.debug_counter += 1
            if self.debug_counter % 100 == 0:
                pos = self.read_joint_positions()
                print(f"Test mode - Pos: [{pos.min():.3f}, {pos.max():.3f}]")
            return actions

        obs, joint_pos_rel, joint_vel = self.get_observation()

        # Print on FIRST step
        if self.debug_counter == 0:
            print("\n=== FIRST STEP OBSERVATION ===")
            print(f"joint_pos_rel: {joint_pos_rel}")
            print(f"prev_actions:  {self.prev_actions}")
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
        if self.debug_counter == 0:
            print(f"raw_actions:   {raw_actions}")
            print("===============================\n")

        raw_actions = raw_actions - self.rest_bias

        # Clip to simulation range
        raw_actions = np.clip(raw_actions, -self.action_clip, self.action_clip)
        
        # Startup ramp
        if self.startup_steps < self.startup_duration:
            ramp = self.startup_steps / self.startup_duration
            raw_actions = raw_actions * ramp
            self.startup_steps += 1
        
        # Smoothing (but less aggressive than before)
        smoothed = self.action_smoothing * raw_actions + (1 - self.action_smoothing) * self.prev_actions
        
        # Rate limiting (but allowing faster changes)
        delta = smoothed - self.prev_actions
        delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
        actions = self.prev_actions + delta
        
        actions = np.clip(actions, -self.action_clip, self.action_clip)
        
        self.prev_actions = actions.copy()
        self.write_joint_positions(actions)
        
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            print(f"\n--- Step {self.debug_counter} ---")
            print(f"Cmd: {self.velocity_command}")
            print(f"Pos: [{joint_pos_rel.min():.3f}, {joint_pos_rel.max():.3f}]")
            print(f"Raw: [{raw_actions.min():.3f}, {raw_actions.max():.3f}]")
            print(f"Out: [{actions.min():.3f}, {actions.max():.3f}]")
        
        if self.debug_counter % 100 == 0:  # Every 2 seconds
            print(f"\n--- Step {self.debug_counter} | Cmd: {self.velocity_command} ---")
            print(f"Joint Pos Rel (per leg):")
            print(f"  LF: [{joint_pos_rel[0]:+.2f}, {joint_pos_rel[1]:+.2f}, {joint_pos_rel[2]:+.2f}]")
            print(f"  RF: [{joint_pos_rel[3]:+.2f}, {joint_pos_rel[4]:+.2f}, {joint_pos_rel[5]:+.2f}]")
            print(f"  LB: [{joint_pos_rel[6]:+.2f}, {joint_pos_rel[7]:+.2f}, {joint_pos_rel[8]:+.2f}]")
            print(f"  RB: [{joint_pos_rel[9]:+.2f}, {joint_pos_rel[10]:+.2f}, {joint_pos_rel[11]:+.2f}]")
            print(f"Actions (per leg):")
            print(f"  LF: [{actions[0]:+.2f}, {actions[1]:+.2f}, {actions[2]:+.2f}]")
            print(f"  RF: [{actions[3]:+.2f}, {actions[4]:+.2f}, {actions[5]:+.2f}]")
            print(f"  LB: [{actions[6]:+.2f}, {actions[7]:+.2f}, {actions[8]:+.2f}]")
            print(f"  RB: [{actions[9]:+.2f}, {actions[10]:+.2f}, {actions[11]:+.2f}]")
        
        if self.debug_counter == 49:
            print("\n=== COMPLETE OBSERVATION VECTOR ===")
            print(f"base_lin_vel:      {obs[0:3]}")
            print(f"base_ang_vel:      {obs[3:6]}")
            print(f"projected_gravity: {obs[6:9]}")
            print(f"velocity_cmd:      {obs[9:12]}")
            print(f"joint_pos_rel:     {obs[12:24]}")
            print(f"joint_vel:         {obs[24:36]}")
            print(f"joint_effort:      {obs[36:48]}")
            print(f"prev_actions:      {obs[48:60]}")
            print("===================================\n")
        
        if self.debug_counter % 100 == 0:
            print(f"\n--- Step {self.debug_counter} | Cmd: {self.velocity_command} ---")
            print(f"Pos: [{joint_pos_rel.min():.3f}, {joint_pos_rel.max():.3f}]")
            print(f"Raw-bias: [{raw_actions.min():.3f}, {raw_actions.max():.3f}]")
            print(f"Out: [{actions.min():.3f}, {actions.max():.3f}]")
        
        return actions
    
    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl @ {self.CONTROL_FREQUENCY}Hz | smooth={self.action_smoothing} | delta={self.max_action_delta}")
        
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
        self.velocity_command = np.array([
            np.clip(vx, -0.35, 0.40),
            np.clip(vy, -0.35, 0.35),
            np.clip(vyaw, -0.30, 0.30)
        ])
        if np.any(np.abs(self.velocity_command) > 0.01):
            if not self.control_active:
                self.prev_actions = np.zeros(12)
                self.prev_joint_pos_rel = np.zeros(12)  # Force to zero
                self.prev_joint_vel = np.zeros(12)
                self.prev_time = time.time()
                self.startup_steps = 0
                
                # TEST: What does policy output for perfect observation?
                test_obs = np.zeros(60, dtype=np.float32)
                test_obs[6:9] = [0, 0, -1]  # projected_gravity
                with torch.no_grad():
                    test_actions = self.policy(torch.tensor(test_obs).unsqueeze(0)).squeeze().numpy()
                print(f"\n=== Policy output for PERFECT zero observation ===")
                print(f"Actions: [{test_actions.min():.3f}, {test_actions.max():.3f}]")
                print(f"Per leg:")
                print(f"  LF: [{test_actions[0]:+.2f}, {test_actions[1]:+.2f}, {test_actions[2]:+.2f}]")
                print(f"  RF: [{test_actions[3]:+.2f}, {test_actions[4]:+.2f}, {test_actions[5]:+.2f}]")
                print(f"  LB: [{test_actions[6]:+.2f}, {test_actions[7]:+.2f}, {test_actions[8]:+.2f}]")
                print(f"  RB: [{test_actions[9]:+.2f}, {test_actions[10]:+.2f}, {test_actions[11]:+.2f}]")
                print("=================================================\n")
                
            self.control_active = True
            print(f"Active: {self.velocity_command}")
        else:
            self.control_active = False
            self.startup_steps = 0
            print("Stopped")
    
    def stop(self):
        self.control_active = False
        self.shutdown = True


if __name__ == "__main__":
    import threading
    
    ctrl = MatchedMLPController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    print("\n--- Sanity Check ---")
    pos = ctrl.read_joint_positions()
    print(f"Standing pos: {pos}")
    
    thread = threading.Thread(target=ctrl.control_loop)
    thread.start()
    
    print("\nw/s/a/d/q/e/z/t = move | space = stop | x = exit")
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'w':
                ctrl.set_velocity_command(0.15, 0, 0)
            elif cmd == 's':
                ctrl.set_velocity_command(-0.15, 0, 0)
            elif cmd == 'a':
                ctrl.set_velocity_command(0, 0.15, 0)
            elif cmd == 'd':
                ctrl.set_velocity_command(0, -0.15, 0)
            elif cmd == 'q':
                ctrl.set_velocity_command(0, 0, 0.15)
            elif cmd == 'e':
                ctrl.set_velocity_command(0, 0, -0.15)
            elif cmd == ' ' or cmd == '':
                ctrl.set_velocity_command(0, 0, 0)
            elif cmd == 'z':
                ctrl.velocity_command = np.array([0.0, 0.0, 0.0])
                ctrl.control_active = True
                ctrl.startup_steps = 0
                print("Zero velocity, active control")
            elif cmd == 't':
                if hasattr(ctrl, 'test_mode') and ctrl.test_mode:
                    ctrl.test_mode = False
                    ctrl.control_active = False
                    print("Test mode OFF")
                else:
                    print("Test mode: Sending zero actions continuously")
                    ctrl.prev_actions = np.zeros(12)
                    ctrl.test_mode = True
                    ctrl.control_active = True
            elif cmd == 'x':
                break
    except KeyboardInterrupt:
        pass
    
    ctrl.stop()
    thread.join()

#############################################
#############################################
#############################################
# import numpy as np
# import torch
# import time
# from MangDang.mini_pupper.ESP32Interface import ESP32Interface

# class RobustMLPController:
#     def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
#         self.esp32 = ESP32Interface()
#         time.sleep(0.5)
        
#         self.policy = torch.jit.load(policy_path)
#         self.policy.eval()
#         print(f"Loaded policy from {policy_path}")
        
#         self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
#         self.direction_multipliers = np.array([
#             -1.0, -1.0, -1.0,
#             -1.0,  1.0,  1.0,
#              1.0, -1.0, -1.0,
#              1.0,  1.0,  1.0,
#         ])
        
#         self.servo_center = 512
#         self.servo_scale = 1024 / (2 * np.pi)
        
#         self.ACTION_SCALE = 0.5
        
#         print("Recording standing pose...")
#         raw = np.array(self.esp32.servos_get_position())
#         self.standing_servos = raw[self.esp32_servo_order].astype(float)
#         print(f"Standing servos: {self.standing_servos}")
        
#         # State
#         self.prev_actions = np.zeros(12)
#         self.prev_joint_pos_rel = np.zeros(12)
#         self.prev_joint_vel = np.zeros(12)
#         self.prev_time = time.time()
#         self.velocity_command = np.zeros(3)
#         self.control_active = False
#         self.shutdown = False
        
#         print("Calibrating IMU...")
#         self._calibrate_imu()
        
#         # Control parameters - TUNED FOR STABILITY
#         self.CONTROL_FREQUENCY = 50      # Match training
#         self.action_smoothing = 0.3      # Conservative smoothing
#         self.max_action_delta = 0.08     # Very gentle rate limit
        
#         # Observation clipping bounds (based on what policy saw in training)
#         self.obs_joint_pos_clip = 0.4    # Policy rarely saw beyond this
#         self.obs_joint_vel_clip = 1.5
#         self.obs_ang_vel_clip = 2.0
        
#         # Action clipping
#         self.action_clip = 1.5           # Policy outputs rarely exceeded this
        
#         self.debug_counter = 0
#         self.startup_steps = 0
#         self.startup_duration = 50       # Ramp up over 1 second at 50Hz
        
#     def _calibrate_imu(self, samples=50):
#         print("Keep robot still...")
#         gyro_sum = np.zeros(3)
#         for _ in range(samples):
#             data = self.esp32.imu_get_data()
#             gyro_sum += np.array([data['gx'], data['gy'], data['gz']])
#             time.sleep(0.02)
#         self.gyro_offset = gyro_sum / samples
#         print(f"Gyro offset: {self.gyro_offset}")
    
#     def read_joint_positions(self):
#         raw = np.array(self.esp32.servos_get_position())
#         isaac_ordered = raw[self.esp32_servo_order].astype(float)
#         servo_delta = isaac_ordered - self.standing_servos
#         delta_radians = servo_delta / self.servo_scale
#         joint_pos_rel = delta_radians * self.direction_multipliers
#         return joint_pos_rel
    
#     def write_joint_positions(self, actions):
#         target_pos_rel = actions * self.ACTION_SCALE
#         delta_radians = target_pos_rel * self.direction_multipliers
#         servo_delta = delta_radians * self.servo_scale
#         target_servos = self.standing_servos + servo_delta
#         target_servos = np.clip(target_servos, 200, 824)  # Tighter safe range
        
#         esp32_out = np.zeros(12)
#         esp32_out[self.esp32_servo_order] = target_servos
#         self.esp32.servos_set_position([int(p) for p in esp32_out])
    
#     def get_observation(self):
#         imu_data = self.esp32.imu_get_data()
        
#         # Angular velocity - clipped
#         gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
#         base_ang_vel = gyro_raw - self.gyro_offset
#         base_ang_vel = np.clip(base_ang_vel, -self.obs_ang_vel_clip, self.obs_ang_vel_clip)
        
#         # Projected gravity
#         accel = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
#         accel_norm = np.linalg.norm(accel)
#         if accel_norm > 0.1:
#             projected_gravity = accel / accel_norm
#         else:
#             projected_gravity = np.array([0, 0, -1])
        
#         base_lin_vel = np.zeros(3)
        
#         # Joint positions - RAW (unclipped) for velocity calculation
#         joint_pos_rel_raw = self.read_joint_positions()
        
#         # Joint velocities from raw positions
#         current_time = time.time()
#         dt = current_time - self.prev_time
#         if dt > 0.001:
#             joint_vel_raw = (joint_pos_rel_raw - self.prev_joint_pos_rel) / dt
#             alpha = 0.4
#             joint_vel = alpha * joint_vel_raw + (1 - alpha) * self.prev_joint_vel
#             joint_vel = np.clip(joint_vel, -self.obs_joint_vel_clip, self.obs_joint_vel_clip)
#         else:
#             joint_vel = self.prev_joint_vel
        
#         self.prev_joint_pos_rel = joint_pos_rel_raw.copy()  # Store raw for velocity calc
#         self.prev_joint_vel = joint_vel.copy()
#         self.prev_time = current_time
        
#         # CLIP joint positions for the observation (what policy sees)
#         joint_pos_rel_clipped = np.clip(joint_pos_rel_raw, -self.obs_joint_pos_clip, self.obs_joint_pos_clip)
        
#         joint_effort = np.zeros(12)
        
#         obs = np.concatenate([
#             base_lin_vel,
#             base_ang_vel,
#             projected_gravity,
#             self.velocity_command,
#             joint_pos_rel_clipped,  # Clipped for policy
#             joint_vel,
#             joint_effort,
#             self.prev_actions
#         ]).astype(np.float32)
        
#         return obs, joint_pos_rel_raw, joint_vel  # Return raw for debugging
    
#     def control_step(self):
#         obs, joint_pos_rel, joint_vel = self.get_observation()
        
#         with torch.no_grad():
#             obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#             raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
#         # Clip raw actions
#         raw_actions = np.clip(raw_actions, -self.action_clip, self.action_clip)
        
#         # Startup ramp - gradually increase authority
#         if self.startup_steps < self.startup_duration:
#             ramp = self.startup_steps / self.startup_duration
#             raw_actions = raw_actions * ramp
#             self.startup_steps += 1
        
#         # Exponential smoothing
#         smoothed = self.action_smoothing * raw_actions + (1 - self.action_smoothing) * self.prev_actions
        
#         # Rate limiting
#         delta = smoothed - self.prev_actions
#         delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
#         actions = self.prev_actions + delta
        
#         # Final clip
#         actions = np.clip(actions, -self.action_clip, self.action_clip)
        
#         self.prev_actions = actions.copy()
#         self.write_joint_positions(actions)
        
#         self.debug_counter += 1
#         if self.debug_counter % 50 == 0:
#             print(f"\n--- Step {self.debug_counter} ---")
#             print(f"Velocity cmd: {self.velocity_command}")
#             print(f"Joint pos rel (raw): [{joint_pos_rel.min():.3f}, {joint_pos_rel.max():.3f}]")
#             print(f"Raw actions (clipped): [{raw_actions.min():.3f}, {raw_actions.max():.3f}]")
#             print(f"Final actions: [{actions.min():.3f}, {actions.max():.3f}]")
#             if self.startup_steps < self.startup_duration:
#                 print(f"Startup ramp: {self.startup_steps}/{self.startup_duration}")
        
#         return actions
    
#     def control_loop(self):
#         dt_target = 1.0 / self.CONTROL_FREQUENCY
        
#         print(f"\nControl loop at {self.CONTROL_FREQUENCY}Hz")
#         print(f"Smoothing={self.action_smoothing}, MaxDelta={self.max_action_delta}")
#         print("Commands: w/s/a/d/q/e = move, space = stop, x = exit")
        
#         while not self.shutdown:
#             loop_start = time.time()
            
#             if self.control_active:
#                 try:
#                     self.control_step()
#                 except Exception as e:
#                     print(f"Error: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     self.control_active = False
            
#             elapsed = time.time() - loop_start
#             if elapsed < dt_target:
#                 time.sleep(dt_target - elapsed)
    
#     def set_velocity_command(self, vx, vy, vyaw):
#         self.velocity_command = np.array([
#             np.clip(vx, -0.35, 0.40),
#             np.clip(vy, -0.35, 0.35),
#             np.clip(vyaw, -0.30, 0.30)
#         ])
#         if np.any(np.abs(self.velocity_command) > 0.01):
#             if not self.control_active:
#                 # Reset everything on activation
#                 self.prev_actions = np.zeros(12)
#                 self.prev_joint_pos_rel = self.read_joint_positions()
#                 self.prev_joint_vel = np.zeros(12)
#                 self.prev_time = time.time()
#                 self.startup_steps = 0  # Reset startup ramp
#             self.control_active = True
#             print(f"Active: cmd={self.velocity_command}")
#         else:
#             self.control_active = False
#             self.startup_steps = 0
#             print("Stopped")
    
#     def stop(self):
#         self.control_active = False
#         self.shutdown = True


# if __name__ == "__main__":
#     import threading
    
#     controller = RobustMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
#     print("\n--- Sanity Check ---")
#     pos = controller.read_joint_positions()
#     print(f"Joint pos rel at standing: {pos}")
#     print(f"(Should be close to zeros)")
    
#     thread = threading.Thread(target=controller.control_loop)
#     thread.start()
    
#     try:
#         while True:
#             cmd = input("> ").strip().lower()
#             if cmd == 'w':
#                 controller.set_velocity_command(0.15, 0, 0)
#             elif cmd == 's':
#                 controller.set_velocity_command(-0.15, 0, 0)
#             elif cmd == 'a':
#                 controller.set_velocity_command(0, 0.15, 0)
#             elif cmd == 'd':
#                 controller.set_velocity_command(0, -0.15, 0)
#             elif cmd == 'q':
#                 controller.set_velocity_command(0, 0, 0.15)
#             elif cmd == 'e':
#                 controller.set_velocity_command(0, 0, -0.15)
#             elif cmd == ' ' or cmd == '':
#                 controller.set_velocity_command(0, 0, 0)
#             elif cmd == 'x':
#                 break
#     except KeyboardInterrupt:
#         pass
    
#     controller.stop()
#     thread.join()
#     print("Done")
#############################################
#############################################
#############################################


#####
#####
#####    
# import numpy as np
# import torch
# import time
# from MangDang.mini_pupper.ESP32Interface import ESP32Interface

# class SimplifiedMLPController:
#     def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
#         self.esp32 = ESP32Interface()
#         time.sleep(0.5)
        
#         self.policy = torch.jit.load(policy_path)
#         self.policy.eval()
#         print(f"Loaded policy from {policy_path}")
        
#         # Servo order: Isaac index -> ESP32 servo index
#         self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
#         # Direction multipliers to convert hardware motion to sim motion
#         self.direction_multipliers = np.array([
#             -1.0, -1.0, -1.0,  # LF
#             -1.0,  1.0,  1.0,  # RF
#              1.0, -1.0, -1.0,  # LB
#              1.0,  1.0,  1.0,  # RB
#         ])
        
#         # Servo constants
#         self.servo_center = 512
#         self.servo_scale = 1024 / (2 * np.pi)  # ~163 counts/radian
        
#         # Action scale from training
#         self.ACTION_SCALE = 0.5  # Use the actual training value!
        
#         # Record the standing pose servo positions
#         print("Recording standing pose...")
#         raw = np.array(self.esp32.servos_get_position())
#         self.standing_servos = raw[self.esp32_servo_order].astype(float)
#         print(f"Standing servo positions (Isaac order): {self.standing_servos}")
        
#         # State
#         self.prev_actions = np.zeros(12)
#         self.prev_joint_pos_rel = np.zeros(12)
#         self.prev_joint_vel = np.zeros(12)
#         self.prev_time = time.time()
#         self.velocity_command = np.zeros(3)
#         self.control_active = False
#         self.shutdown = False
        
#         # IMU calibration
#         print("Calibrating IMU...")
#         self._calibrate_imu()
        
#         self.CONTROL_FREQUENCY = 30
#         self.debug_counter = 0

#         #Action Smoothing
#         self.action_smoothing = 0.5      # 0.3-0.6 typical range
#         self.max_action_delta = 0.2      # Max change per step
        
#     def _calibrate_imu(self, samples=50):
#         print("Keep robot still...")
#         gyro_sum = np.zeros(3)
#         for _ in range(samples):
#             data = self.esp32.imu_get_data()
#             gyro_sum += np.array([data['gx'], data['gy'], data['gz']])
#             time.sleep(0.02)
#         self.gyro_offset = gyro_sum / samples
#         print(f"Gyro offset: {self.gyro_offset}")
    
#     def read_joint_positions(self):
#         """
#         Read servos and return joint_pos_rel (deviation from standing pose, in sim frame).
#         At standing pose, this returns zeros.
#         """
#         raw = np.array(self.esp32.servos_get_position())
#         isaac_ordered = raw[self.esp32_servo_order].astype(float)
        
#         # Deviation from standing pose in servo units
#         servo_delta = isaac_ordered - self.standing_servos
        
#         # Convert to radians
#         delta_radians = servo_delta / self.servo_scale
        
#         # Convert to sim frame (apply direction multipliers)
#         joint_pos_rel = delta_radians * self.direction_multipliers
        
#         return joint_pos_rel
    
#     def write_joint_positions(self, actions):
#         """
#         Convert policy actions to servo commands.
#         Actions are desired joint_pos_rel values (deviations from standing, in sim frame).
#         """
#         # Scale actions
#         target_pos_rel = actions * self.ACTION_SCALE
        
#         # Convert from sim frame to hardware frame
#         delta_radians = target_pos_rel * self.direction_multipliers
        
#         # Convert to servo units
#         servo_delta = delta_radians * self.servo_scale
        
#         # Add to standing pose
#         target_servos = self.standing_servos + servo_delta
#         target_servos = np.clip(target_servos, 150, 874)
        
#         # Reorder for ESP32
#         esp32_out = np.zeros(12)
#         esp32_out[self.esp32_servo_order] = target_servos
        
#         self.esp32.servos_set_position([int(p) for p in esp32_out])
    
#     def get_observation(self):
#         """Build observation vector for policy."""
#         imu_data = self.esp32.imu_get_data()
        
#         # Angular velocity (calibrated gyro)
#         gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
#         base_ang_vel = gyro_raw - self.gyro_offset
        
#         # Projected gravity (accelerometer already in correct convention)
#         accel = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
#         accel_norm = np.linalg.norm(accel)
#         if accel_norm > 0.1:
#             projected_gravity = accel / accel_norm
#         else:
#             projected_gravity = np.array([0, 0, -1])
        
#         # Base linear velocity - zeros
#         base_lin_vel = np.zeros(3)
        
#         # Joint positions (deviation from standing)
#         joint_pos_rel = self.read_joint_positions()
        
#         # Joint velocities
#         current_time = time.time()
#         dt = current_time - self.prev_time
#         if dt > 0.001:
#             joint_vel_raw = (joint_pos_rel - self.prev_joint_pos_rel) / dt
#             alpha = 0.4
#             joint_vel = alpha * joint_vel_raw + (1 - alpha) * self.prev_joint_vel
#             joint_vel = np.clip(joint_vel, -1.5, 1.5)
#         else:
#             joint_vel = self.prev_joint_vel
        
#         self.prev_joint_pos_rel = joint_pos_rel.copy()
#         self.prev_joint_vel = joint_vel.copy()
#         self.prev_time = current_time
        
#         # Joint efforts - zeros
#         joint_effort = np.zeros(12)
        
#         # Build observation
#         obs = np.concatenate([
#             base_lin_vel,           # 3
#             base_ang_vel,           # 3
#             projected_gravity,      # 3
#             self.velocity_command,  # 3
#             joint_pos_rel,          # 12
#             joint_vel,              # 12
#             joint_effort,           # 12
#             self.prev_actions       # 12
#         ]).astype(np.float32)
        
#         return obs, joint_pos_rel, joint_vel
    
#     def control_step(self):
#         """Single control loop iteration."""
#         obs, joint_pos_rel, joint_vel = self.get_observation()
        
#         # with torch.no_grad():
#         #     obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#         #     actions = self.policy(obs_tensor).squeeze().numpy()
        
#         # self.prev_actions = actions.copy()
#         # self.write_joint_positions(actions)
        
#         # self.debug_counter += 1
#         # if self.debug_counter % 50 == 0:
#         #     print(f"\n--- Step {self.debug_counter} ---")
#         #     print(f"Velocity cmd: {self.velocity_command}")
#         #     print(f"Joint pos rel: [{joint_pos_rel.min():.4f}, {joint_pos_rel.max():.4f}]")
#         #     print(f"Joint vel:     [{joint_vel.min():.3f}, {joint_vel.max():.3f}]")
#         #     print(f"Actions:       [{actions.min():.3f}, {actions.max():.3f}]")
#         #     print(f"Actions * scale: [{(actions*self.ACTION_SCALE).min():.3f}, {(actions*self.ACTION_SCALE).max():.3f}]")
        
#         # return actions
#         with torch.no_grad():
#             obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#             raw_actions = self.policy(obs_tensor).squeeze().numpy()
    
#         # Step 1: Exponential smoothing
#         smoothed = self.action_smoothing * raw_actions + (1 - self.action_smoothing) * self.prev_actions
        
#         # Step 2: Rate limiting
#         delta = smoothed - self.prev_actions
#         delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
#         actions = self.prev_actions + delta
        
#         self.prev_actions = actions.copy()
#         self.write_joint_positions(actions)
        
#         self.debug_counter += 1
#         if self.debug_counter % 50 == 0:
#             print(f"\n--- Step {self.debug_counter} ---")
#             print(f"Velocity cmd: {self.velocity_command}")
#             print(f"Joint pos rel: [{joint_pos_rel.min():.4f}, {joint_pos_rel.max():.4f}]")
#             print(f"Raw actions:   [{raw_actions.min():.3f}, {raw_actions.max():.3f}]")
#             print(f"Smooth actions:[{actions.min():.3f}, {actions.max():.3f}]")
        
#         return actions
    
#     def control_loop(self):
#         dt_target = 1.0 / self.CONTROL_FREQUENCY
        
#         print(f"\nControl loop started at {self.CONTROL_FREQUENCY}Hz")
#         print("Commands: w/s/a/d/q/e = move, space = stop, x = exit")
        
#         while not self.shutdown:
#             loop_start = time.time()
            
#             if self.control_active:
#                 try:
#                     self.control_step()
#                 except Exception as e:
#                     print(f"Error: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     self.control_active = False
            
#             elapsed = time.time() - loop_start
#             if elapsed < dt_target:
#                 time.sleep(dt_target - elapsed)
    
#     def set_velocity_command(self, vx, vy, vyaw):
#         self.velocity_command = np.array([
#             np.clip(vx, -0.35, 0.40),
#             np.clip(vy, -0.35, 0.35),
#             np.clip(vyaw, -0.30, 0.30)
#         ])
#         if np.any(np.abs(self.velocity_command) > 0.01):
#             if not self.control_active:
#                 self.prev_actions = np.zeros(12)
#                 self.prev_joint_pos_rel = self.read_joint_positions()
#                 self.prev_joint_vel = np.zeros(12)
#                 self.prev_time = time.time()
#             self.control_active = True
#             print(f"Active: cmd={self.velocity_command}")
#         else:
#             self.control_active = False
#             print("Stopped")
    
#     def stop(self):
#         self.control_active = False
#         self.shutdown = True


# if __name__ == "__main__":
#     import threading
    
#     controller = SimplifiedMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
#     # Quick sanity check
#     print("\n--- Sanity Check ---")
#     pos = controller.read_joint_positions()
#     print(f"Joint pos rel at standing: {pos}")
#     print(f"(Should be very close to zeros)")
    
#     thread = threading.Thread(target=controller.control_loop)
#     thread.start()
    
#     try:
#         while True:
#             cmd = input("> ").strip().lower()
#             if cmd == 'w':
#                 controller.set_velocity_command(0.15, 0, 0)
#             elif cmd == 's':
#                 controller.set_velocity_command(-0.15, 0, 0)
#             elif cmd == 'a':
#                 controller.set_velocity_command(0, 0.15, 0)
#             elif cmd == 'd':
#                 controller.set_velocity_command(0, -0.15, 0)
#             elif cmd == 'q':
#                 controller.set_velocity_command(0, 0, 0.15)
#             elif cmd == 'e':
#                 controller.set_velocity_command(0, 0, -0.15)
#             elif cmd == ' ' or cmd == '':
#                 controller.set_velocity_command(0, 0, 0)
#             elif cmd == 'x':
#                 break
#     except KeyboardInterrupt:
#         pass
    
#     controller.stop()
#     thread.join()
#     print("Done")
#####
#####
#####    

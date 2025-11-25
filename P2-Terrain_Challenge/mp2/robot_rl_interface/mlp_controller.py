import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class SimplifiedMLPController:
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")
        
        # Servo order: Isaac index -> ESP32 servo index
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # Direction multipliers to convert hardware motion to sim motion
        self.direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF
            -1.0,  1.0,  1.0,  # RF
             1.0, -1.0, -1.0,  # LB
             1.0,  1.0,  1.0,  # RB
        ])
        
        # Servo constants
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)  # ~163 counts/radian
        
        # Action scale from training
        self.ACTION_SCALE = 0.15  # Use the actual training value!
        
        # Record the standing pose servo positions
        print("Recording standing pose...")
        raw = np.array(self.esp32.servos_get_position())
        self.standing_servos = raw[self.esp32_servo_order].astype(float)
        print(f"Standing servo positions (Isaac order): {self.standing_servos}")
        
        # State
        self.prev_actions = np.zeros(12)
        self.prev_joint_pos_rel = np.zeros(12)
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        
        # IMU calibration
        print("Calibrating IMU...")
        self._calibrate_imu()
        
        self.CONTROL_FREQUENCY = 50
        self.debug_counter = 0
        
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
        """
        Read servos and return joint_pos_rel (deviation from standing pose, in sim frame).
        At standing pose, this returns zeros.
        """
        raw = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw[self.esp32_servo_order].astype(float)
        
        # Deviation from standing pose in servo units
        servo_delta = isaac_ordered - self.standing_servos
        
        # Convert to radians
        delta_radians = servo_delta / self.servo_scale
        
        # Convert to sim frame (apply direction multipliers)
        joint_pos_rel = delta_radians * self.direction_multipliers
        
        return joint_pos_rel
    
    def write_joint_positions(self, actions):
        """
        Convert policy actions to servo commands.
        Actions are desired joint_pos_rel values (deviations from standing, in sim frame).
        """
        # Scale actions
        target_pos_rel = actions * self.ACTION_SCALE
        
        # Convert from sim frame to hardware frame
        delta_radians = target_pos_rel * self.direction_multipliers
        
        # Convert to servo units
        servo_delta = delta_radians * self.servo_scale
        
        # Add to standing pose
        target_servos = self.standing_servos + servo_delta
        target_servos = np.clip(target_servos, 150, 874)
        
        # Reorder for ESP32
        esp32_out = np.zeros(12)
        esp32_out[self.esp32_servo_order] = target_servos
        
        self.esp32.servos_set_position([int(p) for p in esp32_out])
    
    def get_observation(self):
        """Build observation vector for policy."""
        imu_data = self.esp32.imu_get_data()
        
        # Angular velocity (calibrated gyro)
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = gyro_raw - self.gyro_offset
        
        # Projected gravity (accelerometer already in correct convention)
        accel = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            projected_gravity = accel / accel_norm
        else:
            projected_gravity = np.array([0, 0, -1])
        
        # Base linear velocity - zeros
        base_lin_vel = np.zeros(3)
        
        # Joint positions (deviation from standing)
        joint_pos_rel = self.read_joint_positions()
        
        # Joint velocities
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0.001:
            joint_vel_raw = (joint_pos_rel - self.prev_joint_pos_rel) / dt
            alpha = 0.4
            joint_vel = alpha * joint_vel_raw + (1 - alpha) * self.prev_joint_vel
            joint_vel = np.clip(joint_vel, -1.5, 1.5)
        else:
            joint_vel = self.prev_joint_vel
        
        self.prev_joint_pos_rel = joint_pos_rel.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        # Joint efforts - zeros
        joint_effort = np.zeros(12)
        
        # Build observation
        obs = np.concatenate([
            base_lin_vel,           # 3
            base_ang_vel,           # 3
            projected_gravity,      # 3
            self.velocity_command,  # 3
            joint_pos_rel,          # 12
            joint_vel,              # 12
            joint_effort,           # 12
            self.prev_actions       # 12
        ]).astype(np.float32)
        
        return obs, joint_pos_rel, joint_vel
    
    def control_step(self):
        """Single control loop iteration."""
        obs, joint_pos_rel, joint_vel = self.get_observation()
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            actions = self.policy(obs_tensor).squeeze().numpy()
        
        self.prev_actions = actions.copy()
        self.write_joint_positions(actions)
        
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            print(f"\n--- Step {self.debug_counter} ---")
            print(f"Velocity cmd: {self.velocity_command}")
            print(f"Joint pos rel: [{joint_pos_rel.min():.4f}, {joint_pos_rel.max():.4f}]")
            print(f"Joint vel:     [{joint_vel.min():.3f}, {joint_vel.max():.3f}]")
            print(f"Actions:       [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"Actions * scale: [{(actions*self.ACTION_SCALE).min():.3f}, {(actions*self.ACTION_SCALE).max():.3f}]")
        
        return actions
    
    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl loop started at {self.CONTROL_FREQUENCY}Hz")
        print("Commands: w/s/a/d/q/e = move, space = stop, x = exit")
        
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
                self.prev_joint_pos_rel = self.read_joint_positions()
                self.prev_joint_vel = np.zeros(12)
                self.prev_time = time.time()
            self.control_active = True
            print(f"Active: cmd={self.velocity_command}")
        else:
            self.control_active = False
            print("Stopped")
    
    def stop(self):
        self.control_active = False
        self.shutdown = True


if __name__ == "__main__":
    import threading
    
    controller = SimplifiedMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    
    # Quick sanity check
    print("\n--- Sanity Check ---")
    pos = controller.read_joint_positions()
    print(f"Joint pos rel at standing: {pos}")
    print(f"(Should be very close to zeros)")
    
    thread = threading.Thread(target=controller.control_loop)
    thread.start()
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'w':
                controller.set_velocity_command(0.15, 0, 0)
            elif cmd == 's':
                controller.set_velocity_command(-0.15, 0, 0)
            elif cmd == 'a':
                controller.set_velocity_command(0, 0.15, 0)
            elif cmd == 'd':
                controller.set_velocity_command(0, -0.15, 0)
            elif cmd == 'q':
                controller.set_velocity_command(0, 0, 0.15)
            elif cmd == 'e':
                controller.set_velocity_command(0, 0, -0.15)
            elif cmd == ' ' or cmd == '':
                controller.set_velocity_command(0, 0, 0)
            elif cmd == 'x':
                break
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    thread.join()
    print("Done")
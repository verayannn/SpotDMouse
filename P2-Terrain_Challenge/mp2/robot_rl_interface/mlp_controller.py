import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class SimMatchedMLPController:
    """
    Controller that exactly matches simulation's JointPositionAction processing.
    Key insight: Actions are added to simulation default positions, NOT current positions.
    """
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_joyboy.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        print(f"Loaded policy from {policy_path}")
        
        # Servo order: Isaac index -> ESP32 servo index
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # Direction multipliers to convert hardware motion to sim motion # We may need to verify this again in a separate script.
        self.direction_multipliers = np.array([
            -1.0, -1.0, -1.0,  # LF
            -1.0,  1.0,  1.0,  # RF
             1.0, -1.0, -1.0,  # LB
             1.0,  1.0,  1.0,  # RB
        ])
        
        # CRITICAL: Define simulation's default joint positions (in radians)
        # These are the joint angles when the robot is standing in simulation
        # If your sim uses different defaults, update these!
        self.sim_default_positions = np.array([ 
            # LF leg (base_lf1, lf1_lf2, lf2_lf3) 
            0.0, 0.785, -1.57,  # hip=0°, thigh=-45°, calf=90° # I think we may hvve found the issue.  #That may be th only fix we need.
            # RF leg
            0.0, 0.785, -1.57,
            # LB leg  
            0.0, 0.785, -1.57,
            # RB leg
            0.0, 0.785, -1.57
        ])
        
        # Servo constants
        self.servo_center = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Action scale from training config
        self.ACTION_SCALE = 0.25  # Matches scale=0.5 in SpotActionsCfg
        
        # Record current robot pose as "hardware standing"
        print("Recording hardware standing pose...")
        raw = np.array(self.esp32.servos_get_position())
        self.standing_servos = raw[self.esp32_servo_order].astype(float)
        print(f"Hardware standing servos: {self.standing_servos}")
        
        # Convert hardware standing to joint angles
        self.hw_standing_angles = self._servos_to_angles(self.standing_servos)
        print(f"Hardware standing angles: {self.hw_standing_angles}")
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Potential fix for differnces in refernce maps.
        # The offset maps hardware's zero to sim's standing pose
        # self.hw_to_sim_offset = self.sim_default_positions - self.hw_standing_angles
        # print(f"Hardware to sim offset: {self.hw_to_sim_offset}")
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++#

        # Add detailed debug here
        print("\n--- Hardware vs Simulation Comparison ---")
        print(f"Hardware angles: {self.hw_standing_angles}")
        print(f"Sim defaults:    {self.sim_default_positions}")
        print(f"Differences:     {self.hw_standing_angles - self.sim_default_positions}")
        print("\nPer-leg breakdown:")
        for i, leg in enumerate(['LF', 'RF', 'LB', 'RB']):
            base_idx = i * 3
            print(f"{leg}: HW=[{self.hw_standing_angles[base_idx]:.3f}, {self.hw_standing_angles[base_idx+1]:.3f}, {self.hw_standing_angles[base_idx+2]:.3f}] "
                f"Sim=[{self.sim_default_positions[base_idx]:.3f}, {self.sim_default_positions[base_idx+1]:.3f}, {self.sim_default_positions[base_idx+2]:.3f}]")

        #Results
        # Hardware standing servos: [512. 510. 514. 514. 515. 508. 516. 509. 514. 513. 518. 508.]
        # Hardware standing angles: [-0.          0.01227185 -0.01227185 -0.01227185  0.01840777 -0.02454369
        # 0.02454369  0.01840777 -0.01227185  0.00613592  0.03681554 -0.02454369]
        #Question: If the servos are reporting back numbers around the 512 average... then if we were to add/subtract from these servo positions, we would move a given peice by a determined command
        # What is the relatationship between the output of the servo, changing these relative postitons wrt to the output from the mlp? How are we doing that now?  Looks like _servos_to_angles
    
        # State
        self.prev_actions = np.zeros(12)
        self.prev_joint_angles = self.hw_standing_angles.copy()
        self.prev_joint_vel = np.zeros(12)
        self.prev_time = time.time()
        self.velocity_command = np.zeros(3)
        self.control_active = False
        self.shutdown = False
        
        # IMU calibration
        print("Calibrating IMU...")
        self._calibrate_imu()
        
        # Control parameters
        self.CONTROL_FREQUENCY = 50  # Match simulation
        self.action_clip = True      # Match simulation clipping
        self.action_smoothing = 0.6  # Moderate smoothing
        self.max_action_delta = 0.25  # Allow reasonably fast changes
        
        # Observation clipping to match training ranges
        self.obs_joint_pos_clip = 0.8
        self.obs_joint_vel_clip = 1.5
        self.obs_ang_vel_clip = 2.0
        
        self.debug_counter = 0
        self.startup_steps = 0
        self.startup_duration = 25  # 0.5 second ramp at 50Hz
        
    def _calibrate_imu(self, samples=50):
        print("Keep robot still...")
        gyro_sum = np.zeros(3)
        for _ in range(samples):
            data = self.esp32.imu_get_data()
            gyro_sum += np.array([data['gx'], data['gy'], data['gz']])
            time.sleep(0.02)
        self.gyro_offset = gyro_sum / samples
        print(f"Gyro offset: {self.gyro_offset}")
    
    def _servos_to_angles(self, servo_positions):
        """Convert servo positions to joint angles in radians."""
        servo_delta = servo_positions - self.servo_center
        angles_raw = servo_delta / self.servo_scale
        angles = angles_raw * self.direction_multipliers
        return angles

    #Request: I want to know what this output is. How it compares to the sim expected default stance
    
    def _angles_to_servos(self, angles):
        """Convert joint angles in radians to servo positions."""
        angles_hw = angles * self.direction_multipliers
        servo_delta = angles_hw * self.servo_scale
        servos = self.servo_center + servo_delta
        return servos

    #Potential fixes for the offset between references
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # def _servos_to_angles(self, servo_positions):
    #     """Convert servo positions to joint angles in radians."""
    #     servo_delta = servo_positions - self.servo_center
    #     angles_raw = servo_delta / self.servo_scale
    #     angles_hw = angles_raw * self.direction_multipliers
    #     # Add offset to convert hardware frame to sim frame
    #     angles_sim = angles_hw + self.hw_to_sim_offset
    #     return angles_sim

    # def _angles_to_servos(self, angles_sim):
    #     """Convert joint angles in radians to servo positions."""
    #     # Remove offset to convert sim frame to hardware frame
    #     angles_hw = angles_sim - self.hw_to_sim_offset
    #     angles_raw = angles_hw * self.direction_multipliers
    #     servo_delta = angles_raw * self.servo_scale
    #     servos = self.servo_center + servo_delta
    #     return servos
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def read_joint_positions(self):
        """Read current joint angles in simulation frame."""
        raw = np.array(self.esp32.servos_get_position())
        isaac_ordered = raw[self.esp32_servo_order].astype(float)
        current_angles = self._servos_to_angles(isaac_ordered)
        return current_angles
    
    def write_joint_positions_absolute(self, target_angles):
        """
        Write absolute joint positions in simulation frame.
        """
        # Convert to servo positions
        target_servos = self._angles_to_servos(target_angles)
        target_servos = np.clip(target_servos, 180, 844)
        
        # Reorder for ESP32
        esp32_out = np.zeros(12)
        esp32_out[self.esp32_servo_order] = target_servos
        
        self.esp32.servos_set_position([int(p) for p in esp32_out])
    
    def get_observation(self):
        """Build 60-dim observation vector matching simulation."""
        imu_data = self.esp32.imu_get_data()
        
        # 1. Base linear velocity (3 dims) - zeros on real robot
        base_lin_vel = np.zeros(3)
        
        # 2. Base angular velocity (3 dims)
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        base_ang_vel = gyro_raw - self.gyro_offset
        base_ang_vel = np.clip(base_ang_vel, -self.obs_ang_vel_clip, self.obs_ang_vel_clip)
        
        # 3. Projected gravity (3 dims)
        accel = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            projected_gravity = accel / accel_norm
        else:
            projected_gravity = np.array([0, 0, -1])
        
        # 4. Velocity commands (3 dims)
        velocity_command = self.velocity_command
        
        # 5. Joint positions relative to sim default (12 dims)
        current_angles = self.read_joint_positions()
        # CRITICAL: Compute relative to SIMULATION default, not hardware standing
        joint_pos_rel = current_angles - self.sim_default_positions
        joint_pos_rel_clipped = np.clip(joint_pos_rel, -self.obs_joint_pos_clip, self.obs_joint_pos_clip)
        
        # 6. Joint velocities (12 dims)
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0.001:
            joint_vel_raw = (current_angles - self.prev_joint_angles) / dt
            alpha = 0.4
            joint_vel = alpha * joint_vel_raw + (1 - alpha) * self.prev_joint_vel
            joint_vel = np.clip(joint_vel, -self.obs_joint_vel_clip, self.obs_joint_vel_clip)
        else:
            joint_vel = self.prev_joint_vel
        
        self.prev_joint_angles = current_angles.copy()
        self.prev_joint_vel = joint_vel.copy()
        self.prev_time = current_time
        
        # 7. Joint efforts (12 dims) - zeros
        joint_effort = np.zeros(12)
        
        # 8. Previous actions (12 dims)
        prev_actions = self.prev_actions
        
        # Concatenate into 60-dim observation
        obs = np.concatenate([
            base_lin_vel,           # 0:3
            base_ang_vel,           # 3:6
            projected_gravity,      # 6:9
            velocity_command,       # 9:12
            joint_pos_rel_clipped,  # 12:24
            joint_vel,              # 24:36
            joint_effort,           # 36:48
            prev_actions            # 48:60
        ]).astype(np.float32)
        
        return obs, joint_pos_rel, joint_vel
    
    def process_actions(self, raw_actions):
        """
        Process actions exactly as simulation's JointPositionAction.
        
        From simulation:
        1. Clip to [-1, 1] if configured
        2. Scale by action_scale (0.5)
        3. Add to default positions (NOT current positions!)
        """
        # Step 1: Clip if configured
        if self.action_clip:
            actions = np.clip(raw_actions, -1.0, 1.0)
        else:
            actions = raw_actions
        
        # Step 2: Scale by action_scale
        scaled_actions = actions * self.ACTION_SCALE
        
        # Step 3: Add to simulation default positions
        # This is the key difference - we add to default, not current!
        target_positions = self.sim_default_positions + scaled_actions
        
        return target_positions, actions
    
    def control_step(self):
        """Single control loop iteration."""
        # Get observation
        obs, joint_pos_rel, joint_vel = self.get_observation()
        
        # Policy inference
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            raw_actions = self.policy(obs_tensor).squeeze().numpy()
        
        # Process actions as in simulation
        target_positions, clipped_actions = self.process_actions(raw_actions)
        
        # Startup ramp
        if self.startup_steps < self.startup_duration:
            ramp = self.startup_steps / self.startup_duration
            target_positions = self.sim_default_positions + (target_positions - self.sim_default_positions) * ramp
            clipped_actions = clipped_actions * ramp
            self.startup_steps += 1
        
        # Optional smoothing on the target positions
        if self.action_smoothing > 0 and self.debug_counter > 0:
            # Smooth the change in target positions
            if hasattr(self, 'prev_target_positions'):
                alpha = self.action_smoothing
                smoothed_targets = alpha * target_positions + (1 - alpha) * self.prev_target_positions
                
                # Rate limit the change
                delta = smoothed_targets - self.prev_target_positions
                delta = np.clip(delta, -self.max_action_delta, self.max_action_delta)
                target_positions = self.prev_target_positions + delta
        
        self.prev_target_positions = target_positions.copy()
        
        # Send absolute positions to robot
        self.write_joint_positions_absolute(target_positions)
        # Question: What is the MLP outputting to the simulated robots? Torques? Joint positons? Radians?
        
        # Update state
        self.prev_actions = clipped_actions.copy()
        
        # Debug output
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            print(f"\n--- Step {self.debug_counter} ---")
            print(f"Velocity cmd: {self.velocity_command}")
            print(f"Joint pos rel: [{joint_pos_rel.min():.3f}, {joint_pos_rel.max():.3f}]")
            print(f"Raw actions: [{raw_actions.min():.3f}, {raw_actions.max():.3f}]")
            print(f"Target pos (abs): [{target_positions.min():.3f}, {target_positions.max():.3f}]")
            if self.startup_steps < self.startup_duration:
                print(f"Startup: {self.startup_steps}/{self.startup_duration}")
        
        #Result
#         --- Detailed Step 400 ---
        # Target positions (absolute, sim frame):
        #   LF: [+0.12, -0.86, +1.40]
        #   RF: [-0.03, -0.68, +1.56]
        #   LB: [+0.03, -0.85, +1.75]
        #   RB: [+0.02, -0.97, +1.42]
        # Sim defaults:
        #   LF: [+0.00, -0.79, +1.57]

        # --- Step 450 ---
        # Velocity cmd: [0.3 0.  0. ]
        # Joint pos rel: [-0.226, 0.251]
        # Raw actions: [-0.508, 0.621]
        # Target pos (abs): [-0.853, 1.687]
        # Question: Why are the Target Pos so large? Based on the per-limb rotation tests, what is the rellationship between radians and set target postiions and how can we use that to inform how this
        # robot executes commands? 
        # Question: How does the IsaacSim Send commands (target postion? Radians?) How can we comunicate with the ESP32 interface the same way that the Isaacsimn does with whatever function controls the
        #joints?
            
        if self.debug_counter % 200 == 0:
            print(f"\n--- Detailed Step {self.debug_counter} ---")
            print("Target positions (absolute, sim frame):")
            print(f"  LF: [{target_positions[0]:+.2f}, {target_positions[1]:+.2f}, {target_positions[2]:+.2f}]")
            print(f"  RF: [{target_positions[3]:+.2f}, {target_positions[4]:+.2f}, {target_positions[5]:+.2f}]")
            print(f"  LB: [{target_positions[6]:+.2f}, {target_positions[7]:+.2f}, {target_positions[8]:+.2f}]")
            print(f"  RB: [{target_positions[9]:+.2f}, {target_positions[10]:+.2f}, {target_positions[11]:+.2f}]")
            print("Sim defaults:")
            print(f"  LF: [{self.sim_default_positions[0]:+.2f}, {self.sim_default_positions[1]:+.2f}, {self.sim_default_positions[2]:+.2f}]")
        
        return clipped_actions
    
    def control_loop(self):
        dt_target = 1.0 / self.CONTROL_FREQUENCY
        
        print(f"\nControl loop at {self.CONTROL_FREQUENCY}Hz")
        print(f"Action scale: {self.ACTION_SCALE}")
        print(f"Smoothing: {self.action_smoothing}")
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
                # Reset on activation
                current_angles = self.read_joint_positions()
                self.prev_joint_angles = current_angles.copy()
                self.prev_actions = np.zeros(12)
                self.prev_joint_vel = np.zeros(12)
                self.prev_time = time.time()
                self.startup_steps = 0
                self.debug_counter = 0
                if hasattr(self, 'prev_target_positions'):
                    del self.prev_target_positions
            self.control_active = True
            print(f"Active: cmd={self.velocity_command}")
        else:
            self.control_active = False
            self.startup_steps = 0
            print("Stopped")
    
    def stop(self):
        self.control_active = False
        self.shutdown = True


if __name__ == "__main__":
    import threading
    
    # Update path as needed
    controller = SimMatchedMLPController("/home/ubuntu/mp2_mlp/policy_joyboy.pt")
    
    # Test observation
    print("\n--- Test Observation ---")
    obs, pos_rel, vel = controller.get_observation()
    print(f"Observation shape: {obs.shape}")
    print(f"Joint pos rel to sim default: [{pos_rel.min():.3f}, {pos_rel.max():.3f}]")
    
    # Test action processing
    print("\n--- Test Action Processing ---")
    test_actions = np.zeros(12)
    target_pos, _ = controller.process_actions(test_actions)
    print(f"Zero actions -> target positions: {target_pos}")
    print(f"Should equal sim defaults: {controller.sim_default_positions}")
    print(f"Match? {np.allclose(target_pos, controller.sim_default_positions)}")
    
    # Test with small actions
    test_actions = np.ones(12) * 0.1
    target_pos, _ = controller.process_actions(test_actions)
    expected = controller.sim_default_positions + 0.1 * controller.ACTION_SCALE
    print(f"\n0.1 actions -> target positions: {target_pos}")
    print(f"Expected: {expected}")
    print(f"Match? {np.allclose(target_pos, expected)}")
    
    thread = threading.Thread(target=controller.control_loop)
    thread.start()
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == 'w':
                controller.set_velocity_command(0.3, 0, 0)
            elif cmd == 's':
                controller.set_velocity_command(-0.3, 0, 0)
            elif cmd == 'a':
                controller.set_velocity_command(0, 0.3, 0)
            elif cmd == 'd':
                controller.set_velocity_command(0, -0.3, 0)
            elif cmd == 'q':
                controller.set_velocity_command(0, 0, 0.3)
            elif cmd == 'e':
                controller.set_velocity_command(0, 0, -0.3)
            elif cmd == ' ' or cmd == '':
                controller.set_velocity_command(0, 0, 0)
            elif cmd == 'x':
                break
    except KeyboardInterrupt:
        pass
    
    controller.stop()
    thread.join()
    print("Done")
import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class FinalMLPController:
    def __init__(self):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Load policy
        self.policy = self.load_policy()
        
        # === Hardware Mapping Layer ===
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.isaac_to_esp32 = np.zeros(12, dtype=int)
        for i, esp_idx in enumerate(self.esp32_servo_order):
            self.isaac_to_esp32[esp_idx] = i
        
        self.joint_direction_multipliers = np.array([
            -1.0,  1.0,  1.0,  # LF
             1.0, -1.0, -1.0,  # RF
            -1.0,  1.0,  1.0,  # LB
            -1.0, -1.0, -1.0,  # RB
        ])
        
        # === Reference Frame Translation ===
        
        # What Isaac Sim was trained with
        self.isaac_training_defaults = np.array([
            -0.1, 0.8, -1.5,  # LF
             0.1, 0.8, -1.5,  # RF
            -0.1, 0.8, -1.5,  # LB
             0.1, 0.8, -1.5   # RB
        ])
        
        # Hardware's true standing angles (from your test)
        self.hardware_standing_angles = np.array([
            -0.018, -0.043, 0.074,  # LF
             0.037, -0.049, 0.049,  # RF
             0.000, -0.055, 0.055,  # LB
            -0.006, -0.037, 0.074   # RB
        ])
        
        # Servo calibration
        self.servo_offset = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Action processing
        self.ACTION_SCALE = 0.25
        self.MAX_ACTION_CHANGE = 0.1
        self.prev_actions = np.zeros(12)
        
    def get_observation(self):
        """Get observation in Isaac Sim's training reference frame"""
        # Read hardware
        raw_positions = np.array(self.esp32.servos_get_position())
        
        # Reorder to Isaac format
        isaac_ordered = raw_positions[self.esp32_servo_order]
        
        # Convert to radians
        hardware_radians = (isaac_ordered - self.servo_offset) / self.servo_scale
        
        # Apply direction corrections to get true angles
        true_angles = hardware_radians / self.joint_direction_multipliers
        
        # CRITICAL: Transform to Isaac's training frame
        # The MLP expects positions relative to its training defaults
        isaac_relative_positions = true_angles - self.hardware_standing_angles + self.isaac_training_defaults
        
        # Build observation
        obs = np.concatenate([
            np.zeros(3),                    # base_lin_vel
            np.zeros(3),                    # base_ang_vel  
            np.array([0, 0, -1]),          # projected_gravity
            self.velocity_command,          # commands
            isaac_relative_positions,       # joint positions (in Isaac frame!)
            np.zeros(12),                   # joint velocities
            np.zeros(12),                   # joint efforts
            self.prev_actions              # previous actions
        ])
        
        return obs
    
    def process_actions(self, mlp_actions):
        """Convert MLP actions to hardware commands"""
        # Scale and smooth
        scaled = mlp_actions * self.ACTION_SCALE
        
        # Limit changes
        delta = np.clip(scaled - self.prev_actions, -self.MAX_ACTION_CHANGE, self.MAX_ACTION_CHANGE)
        smoothed = self.prev_actions + delta
        self.prev_actions = smoothed
        
        # MLP outputs are relative to Isaac training defaults
        isaac_absolute = smoothed + self.isaac_training_defaults
        
        # Transform to hardware frame
        # 1. Remove Isaac's expected standing pose
        # 2. Add hardware's actual standing pose
        hardware_angles = isaac_absolute - self.isaac_training_defaults + self.hardware_standing_angles
        
        # Apply direction corrections
        hardware_corrected = hardware_angles * self.joint_direction_multipliers
        
        # Convert to servo positions
        servo_positions = hardware_corrected * self.servo_scale + self.servo_offset
        servo_positions = np.clip(servo_positions, 100, 924)
        
        # Reorder for ESP32
        esp32_positions = servo_positions[self.isaac_to_esp32]
        
        return [int(pos) for pos in esp32_positions]

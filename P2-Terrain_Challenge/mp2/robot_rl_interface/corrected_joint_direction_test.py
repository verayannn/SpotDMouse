import numpy as np
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time

class CorrectedJointDirections:
    def __init__(self):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Servo calibration
        self.servo_offset = 512 
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Mapping arrays
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.isaac_to_esp32 = np.zeros(12, dtype=int)
        for i, esp_idx in enumerate(self.esp32_servo_order):
            self.isaac_to_esp32[esp_idx] = i
        
        # Direction multipliers for each joint
        # Based on your observations:
        # - Left legs (LF, LB): positive = forward for thigh/calf
        # - Right legs (RF, RB): negative = forward for thigh/calf
        # - Hip joints need individual handling
        
        self.joint_direction_multipliers = np.array([
            # LF leg (positive moves forward)
            -1.0,  # LF hip: positive rotates INTO robot (we may want outward for abduction)
             1.0,  # LF thigh: positive moves forward
             1.0,  # LF calf: positive moves forward
            
            # RF leg (negative moves forward)
             1.0,  # RF hip: positive rotates INTO robot (we may want outward for abduction)
            -1.0,  # RF thigh: negative moves forward
            -1.0,  # RF calf: negative moves forward
            
            # LB leg (positive moves forward)
            -1.0,  # LB hip: positive rotates INTO robot (we may want outward for abduction)
             1.0,  # LB thigh: positive moves forward
             1.0,  # LB calf: positive moves forward
            
            # RB leg (negative moves forward)
            -1.0,  # RB hip: positive rotates AWAY from robot (already correct for abduction)
            -1.0,  # RB thigh: negative moves forward
            -1.0,  # RB calf: negative moves forward
        ])
        
        # Original Isaac defaults
        self.isaac_defaults_original = np.array([
            -0.1, 0.8, -1.5,  # LF
             0.1, 0.8, -1.5,  # RF
            -0.1, 0.8, -1.5,  # LB
             0.1, 0.8, -1.5   # RB
        ])
        
        # Apply direction corrections
        self.isaac_defaults_corrected = self.isaac_defaults_original * self.joint_direction_multipliers
        
        # Store initial positions
        self.initial_positions = np.array(self.esp32.servos_get_position())
    
    def radians_to_servo(self, radians, joint_idx):
        """Convert radians to servo position with direction correction"""
        corrected_radians = radians * self.joint_direction_multipliers[joint_idx]
        return int(corrected_radians * self.servo_scale + self.servo_offset)
    
    def test_corrected_standing_position(self):
        """Test the corrected standing position"""
        print("\n" + "="*60)
        print("TESTING CORRECTED STANDING POSITION")
        print("="*60)
        
        print("\nOriginal Isaac defaults (rad):", self.isaac_defaults_original)
        print("Corrected with direction multipliers:", self.isaac_defaults_corrected)
        
        input("\nPress Enter to move to CORRECTED standing position...")
        
        # Convert to servo positions with direction correction
        servo_positions = []
        for i in range(12):
            servo_pos = self.radians_to_servo(self.isaac_defaults_original[i], i)
            servo_positions.append(servo_pos)
        
        servo_positions = np.clip(servo_positions, 100, 924)
        
        # Reorder for ESP32
        esp32_positions = np.array(servo_positions)[self.isaac_to_esp32]
        
        print("ESP32 servo positions:", [int(pos) for pos in esp32_positions])
        
        self.esp32.servos_set_position([int(pos) for pos in esp32_positions])
        
        input("\nObserve the standing position. Press Enter to continue...")
        
        # Return to initial
        self.esp32.servos_set_position(self.initial_positions)
    
    def test_symmetric_movements(self):
        """Test symmetric movements to verify corrections"""
        print("\n" + "="*60)
        print("TESTING SYMMETRIC MOVEMENTS")
        print("="*60)
        
        # Test moving all thighs forward
        input("\nPress Enter to move all THIGHS forward...")
        
        thigh_indices = [1, 4, 7, 10]  # Isaac indices for thighs
        test_positions = self.initial_positions.copy()
        
        for isaac_idx in thigh_indices:
            esp32_idx = self.isaac_to_esp32[isaac_idx]
            # Move forward by 0.3 rad in Isaac space
            servo_pos = self.radians_to_servo(0.3, isaac_idx)
            test_positions[esp32_idx] = np.clip(servo_pos, 100, 924)
        
        self.esp32.servos_set_position(test_positions)
        time.sleep(2.0)
        
        # Test moving all calfs
        input("\nPress Enter to move all CALFS to bent position...")
        
        calf_indices = [2, 5, 8, 11]  # Isaac indices for calfs
        test_positions = self.initial_positions.copy()
        
        for isaac_idx in calf_indices:
            esp32_idx = self.isaac_to_esp32[isaac_idx]
            # Bend by -0.3 rad in Isaac space
            servo_pos = self.radians_to_servo(-1.8, isaac_idx)
            test_positions[esp32_idx] = np.clip(servo_pos, 100, 924)
        
        self.esp32.servos_set_position(test_positions)
        time.sleep(2.0)
        
        # Return to initial
        print("Returning to initial position...")
        self.esp32.servos_set_position(self.initial_positions)

# Also update your controller with these corrections
def get_corrected_controller_process_actions(self):
    """Updated process_actions method for the controller"""
    def process_actions(raw_actions):
        # Scale actions
        scaled = raw_actions * self.ACTION_SCALE
        
        # Apply per-joint limits
        limited = scaled.copy()
        for i in range(4):
            base_idx = i * 3
            limited[base_idx] = np.clip(limited[base_idx], -0.5, 0.5)
            limited[base_idx + 1] = np.clip(limited[base_idx + 1], -1.0, 1.0)
            limited[base_idx + 2] = np.clip(limited[base_idx + 2], -1.0, 1.0)
        
        # Smooth actions
        action_delta = limited - self.prev_actions
        action_delta = np.clip(action_delta, -self.MAX_ACTION_CHANGE, self.MAX_ACTION_CHANGE)
        smoothed = self.prev_actions + action_delta
        
        # Update for next iteration
        self.prev_actions = smoothed.copy()
        
        # Apply direction corrections before converting to absolute positions
        corrected_actions = smoothed * self.joint_direction_multipliers
        
        # Convert to servo positions
        absolute_positions = corrected_actions + self.isaac_defaults_corrected
        servo_positions = absolute_positions * self.servo_scale + self.servo_offset
        servo_positions = np.clip(servo_positions, 100, 924)
        
        # Reorder for ESP32
        esp32_positions = servo_positions[self.isaac_to_esp32]
        
        return [int(pos) for pos in esp32_positions]
    
    return process_actions

def main():
    tester = CorrectedJointDirections()
    
    while True:
        print("\n" + "="*60)
        print("CORRECTED JOINT DIRECTION MENU")
        print("="*60)
        print("1. Test corrected standing position")
        print("2. Test symmetric movements")
        print("3. Return to home position")
        print("q. Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == '1':
            tester.test_corrected_standing_position()
        elif choice == '2':
            tester.test_symmetric_movements()
        elif choice == '3':
            print("Returning to initial position...")
            tester.esp32.servos_set_position(tester.initial_positions)
        elif choice == 'q':
            break
    
    print("\nReturning to initial position...")
    tester.esp32.servos_set_position(tester.initial_positions)
    print("Done!")

if __name__ == "__main__":
    main()
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time
import numpy as np

#The order correction between IsaacSim to the ESP32Interface leg order.
#Now we need to make sure that the direction for each limb makes sense given how the MLP was trained

class ServoMappingTester:
    def __init__(self):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Servo calibration
        self.servo_offset = 512 
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Isaac order: [LF, RF, LB, RB] with 3 joints each
        self.isaac_leg_names = ["LF", "RF", "LB", "RB"]
        self.isaac_joint_names = ["hip", "thigh", "calf"]
        
        # ESP32 order: [FR, FL, BR, BL] with 3 joints each
        self.esp32_leg_names = ["FR", "FL", "BR", "BL"]
        self.esp32_joint_names = ["abduction", "hip", "knee"]
        
        # Mapping from Isaac to ESP32
        # Isaac order: [LF, RF, LB, RB]
        # ESP32 order: [FR, FL, BR, BL]
        self.esp32_servo_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
        # Create inverse mapping
        self.isaac_to_esp32 = np.zeros(12, dtype=int)
        for i, esp_idx in enumerate(self.esp32_servo_order):
            self.isaac_to_esp32[esp_idx] = i
        
        # Store initial positions
        self.initial_positions = np.array(self.esp32.servos_get_position())
        print(f"Initial servo positions: {self.initial_positions}")
        
    def test_individual_servos_esp32_order(self):
        """Test each servo in ESP32 order to verify physical mapping"""
        print("\n" + "="*60)
        print("TESTING SERVOS IN ESP32 ORDER")
        print("="*60)
        print("Watch the robot - each joint will move slightly")
        print("ESP32 order: [FR, FL, BR, BL]")
        
        for esp32_idx in range(12):
            leg_idx = esp32_idx // 3
            joint_idx = esp32_idx % 3
            leg_name = self.esp32_leg_names[leg_idx]
            joint_name = self.esp32_joint_names[joint_idx]
            
            input(f"\nPress Enter to test ESP32 servo {esp32_idx} ({leg_name} {joint_name})...")
            
            # Move only this servo
            test_positions = self.initial_positions.copy()
            test_positions[esp32_idx] += 100  # Small movement
            
            print(f"Moving servo {esp32_idx}: {self.initial_positions[esp32_idx]} -> {test_positions[esp32_idx]}")
            self.esp32.servos_set_position(test_positions)
            time.sleep(0.5)
            
            # Return to initial
            self.esp32.servos_set_position(self.initial_positions)
            time.sleep(0.2)
    
    def test_individual_servos_isaac_order(self):
        """Test each servo in Isaac order to verify mapping"""
        print("\n" + "="*60)
        print("TESTING SERVOS IN ISAAC SIM ORDER")
        print("="*60)
        print("Watch the robot - each joint will move slightly")
        print("Isaac order: [LF, RF, LB, RB]")
        
        for isaac_idx in range(12):
            leg_idx = isaac_idx // 3
            joint_idx = isaac_idx % 3
            leg_name = self.isaac_leg_names[leg_idx]
            joint_name = self.isaac_joint_names[joint_idx]
            
            # Get corresponding ESP32 index
            esp32_idx = self.isaac_to_esp32[isaac_idx]
            
            input(f"\nPress Enter to test Isaac joint {isaac_idx} ({leg_name} {joint_name}) -> ESP32 servo {esp32_idx}...")
            
            # Move only this servo
            test_positions = self.initial_positions.copy()
            test_positions[esp32_idx] += 100  # Small movement
            
            print(f"Moving ESP32 servo {esp32_idx}: {self.initial_positions[esp32_idx]} -> {test_positions[esp32_idx]}")
            self.esp32.servos_set_position(test_positions)
            time.sleep(0.5)
            
            # Return to initial
            self.esp32.servos_set_position(self.initial_positions)
            time.sleep(0.2)
    
    def test_legs_isaac_order(self):
        """Test each leg in Isaac order"""
        print("\n" + "="*60)
        print("TESTING FULL LEGS IN ISAAC ORDER")
        print("="*60)
        
        for leg_idx, leg_name in enumerate(self.isaac_leg_names):
            input(f"\nPress Enter to test {leg_name} leg (all 3 joints)...")
            
            # Get all 3 joints for this leg
            isaac_indices = [leg_idx * 3, leg_idx * 3 + 1, leg_idx * 3 + 2]
            esp32_indices = [self.isaac_to_esp32[i] for i in isaac_indices]
            
            print(f"Isaac indices {isaac_indices} -> ESP32 indices {esp32_indices}")
            
            # Move all 3 joints
            test_positions = self.initial_positions.copy()
            for esp32_idx in esp32_indices:
                test_positions[esp32_idx] += 100
            
            self.esp32.servos_set_position(test_positions)
            time.sleep(0.8)
            
            # Return to initial
            self.esp32.servos_set_position(self.initial_positions)
            time.sleep(0.3)
    
    def verify_mapping_table(self):
        """Print mapping table for verification"""
        print("\n" + "="*60)
        print("SERVO MAPPING TABLE")
        print("="*60)
        print("Isaac Index -> ESP32 Index")
        print("-" * 60)
        
        for isaac_idx in range(12):
            leg_idx = isaac_idx // 3
            joint_idx = isaac_idx % 3
            isaac_name = f"{self.isaac_leg_names[leg_idx]} {self.isaac_joint_names[joint_idx]}"
            
            esp32_idx = self.isaac_to_esp32[isaac_idx]
            esp32_leg_idx = esp32_idx // 3
            esp32_joint_idx = esp32_idx % 3
            esp32_name = f"{self.esp32_leg_names[esp32_leg_idx]} {self.esp32_joint_names[esp32_joint_idx]}"
            
            print(f"[{isaac_idx:2d}] {isaac_name:12s} -> [{esp32_idx:2d}] {esp32_name:12s}")
        
        print("\nMapping array for code:")
        print(f"esp32_servo_order = {self.esp32_servo_order}")
        print(f"isaac_to_esp32 = {list(self.isaac_to_esp32)}")
    
    def test_sensor_feedback(self):
        """Test sensor feedback and reordering"""
        print("\n" + "="*60)
        print("TESTING SENSOR FEEDBACK")
        print("="*60)
        
        print("\nRaw ESP32 order:")
        raw_positions = np.array(self.esp32.servos_get_position())
        raw_loads = np.array(self.esp32.servos_get_load())
        
        for i in range(12):
            leg_idx = i // 3
            joint_idx = i % 3
            print(f"[{i:2d}] {self.esp32_leg_names[leg_idx]} {self.esp32_joint_names[joint_idx]}: "
                  f"pos={raw_positions[i]:4d}, load={raw_loads[i]:4d}")
        
        print("\nReordered to Isaac format:")
        isaac_positions = raw_positions[self.esp32_servo_order]
        isaac_loads = raw_loads[self.esp32_servo_order]
        
        for i in range(12):
            leg_idx = i // 3
            joint_idx = i % 3
            print(f"[{i:2d}] {self.isaac_leg_names[leg_idx]} {self.isaac_joint_names[joint_idx]}: "
                  f"pos={isaac_positions[i]:4d}, load={isaac_loads[i]:4d}")

def main():
    tester = ServoMappingTester()
    
    while True:
        print("\n" + "="*60)
        print("SERVO MAPPING TEST MENU")
        print("="*60)
        print("1. Test servos in ESP32 order (hardware order)")
        print("2. Test servos in Isaac Sim order (training order)")
        print("3. Test full legs in Isaac order")
        print("4. Show mapping table")
        print("5. Test sensor feedback")
        print("6. Return to home position")
        print("q. Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == '1':
            tester.test_individual_servos_esp32_order()
        elif choice == '2':
            tester.test_individual_servos_isaac_order()
        elif choice == '3':
            tester.test_legs_isaac_order()
        elif choice == '4':
            tester.verify_mapping_table()
        elif choice == '5':
            tester.test_sensor_feedback()
        elif choice == '6':
            print("Returning to initial position...")
            tester.esp32.servos_set_position(tester.initial_positions)
        elif choice == 'q':
            break
        else:
            print("Invalid choice")
    
    print("\nReturning to initial position...")
    tester.esp32.servos_set_position(tester.initial_positions)
    print("Done!")

if __name__ == "__main__":
    main()
    


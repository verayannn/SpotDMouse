import numpy as np
import time
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

class SensorDiagnostic:
    def __init__(self):
        print("="*60)
        print("SENSOR DIAGNOSTIC TOOL")
        print("="*60)
        
        self.config = Configuration()
        self.hardware = HardwareInterface()
        self.esp32 = self.hardware.pwm_params.esp32
        self.servo_params = self.hardware.servo_params
        self.pwm_params = self.hardware.pwm_params
        
        # Isaac Sim Standing Pose (The Target)
        self.sim_default = np.array([
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57   # RB
        ])

    def _hardware_read_to_isaac(self):
        raw_positions = self.esp32.servos_get_position()
        current_angles_isaac = np.zeros(12)
        
        # Mapping: Isaac Leg Index -> Hardware Matrix Col Index
        # LF(0)->1, RF(1)->0, LB(2)->3, RB(3)->2
        legs_map = [(0, 1), (1, 0), (2, 3), (3, 2)]
        
        for isaac_leg_idx, hw_col_idx in legs_map:
            for axis in range(3):
                servo_id = self.pwm_params.servo_ids[axis, hw_col_idx]
                raw_val = raw_positions[servo_id - 1]
                
                neutral_pos = self.servo_params.neutral_position
                micros_per_rad = self.servo_params.micros_per_rad
                neutral_angle = self.servo_params.neutral_angles[axis, hw_col_idx]
                multiplier = self.servo_params.servo_multipliers[axis, hw_col_idx]
                
                if raw_val == 0 or raw_val > 1024: 
                    angle = 0.0
                else:
                    # The Reverse Math
                    deviation = (neutral_pos - raw_val) / micros_per_rad
                    angle_dev = deviation / multiplier
                    angle = angle_dev + neutral_angle
                
                current_angles_isaac[isaac_leg_idx*3 + axis] = angle
        return current_angles_isaac

    def run(self):
        print("\nINSTRUCTIONS:")
        print("1. Manually move the legs.")
        print("2. Verify the correct leg updates on screen.")
        print("3. Verify values match the 'Target' when standing.")
        print("-" * 60)
        
        try:
            while True:
                angles = self._hardware_read_to_isaac()
                
                # Pretty Print
                # We show LF and RF side-by-side
                print(f"\033[H\033[J") # Clear screen
                print("       LEFT FRONT (LF)      |      RIGHT FRONT (RF)")
                print(f"Hip:   {angles[0]:+6.2f} (Tgt 0.00)  |  {angles[3]:+6.2f} (Tgt 0.00)")
                print(f"Thigh: {angles[1]:+6.2f} (Tgt 0.79)  |  {angles[4]:+6.2f} (Tgt 0.79)")
                print(f"Calf:  {angles[2]:+6.2f} (Tgt -1.57) |  {angles[5]:+6.2f} (Tgt -1.57)")
                print("-" * 55)
                print("       LEFT BACK (LB)       |      RIGHT BACK (RB)")
                print(f"Hip:   {angles[6]:+6.2f} (Tgt 0.00)  |  {angles[9]:+6.2f} (Tgt 0.00)")
                print(f"Thigh: {angles[7]:+6.2f} (Tgt 0.79)  |  {angles[10]:+6.2f} (Tgt 0.79)")
                print(f"Calf:  {angles[8]:+6.2f} (Tgt -1.57) |  {angles[11]:+6.2f} (Tgt -1.57)")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nDone.")

if __name__ == "__main__":
    diag = SensorDiagnostic()
    diag.run()
    
# import numpy as np
# import time
# from MangDang.mini_pupper.HardwareInterface import HardwareInterface
# from MangDang.mini_pupper.Config import Configuration

# def test_hardware_matrix():
#     print("=" * 60)
#     print("HARDWARE INTERFACE MATRIX TEST (SUPER TALL)")
#     print("=" * 60)
#     print("Commanding a HIGH standing pose to prevent the 'crouch'.")
    
#     config = Configuration()
#     hardware = HardwareInterface()
    
#     # --- FIX: COMMAND "SUPER TALL" ANGLES ---
#     # Thigh: 0.5 rad (approx 28 degrees) - Much straighter
#     # Calf: -1.0 rad (approx 57 degrees) - Much straighter
#     neutral_matrix = np.zeros((3, 4)) 
#     neutral_matrix[1, :] = 0.5    # Thighs
#     neutral_matrix[2, :] = -1.0   # Calves
    
#     print("\n[1/5] Moving to SUPER TALL Standing Pose...")
#     hardware.set_actuator_postions(neutral_matrix)
#     time.sleep(2)
    
#     # --- TEST 1: COLUMN 1 (Should be LF) ---
#     print("\n[2/5] Wiggling LEFT FRONT (Column 1)...")
#     test_mat = neutral_matrix.copy()
    
#     for _ in range(3):
#         test_mat[1, 1] = 0.5 + 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)
#         test_mat[1, 1] = 0.5 - 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)
        
#     hardware.set_actuator_postions(neutral_matrix)
#     print(">> CHECK: Did the LEFT FRONT leg wiggle?")
#     input("Press Enter to continue...")

#     # --- TEST 2: COLUMN 0 (Should be RF) ---
#     print("\n[3/5] Wiggling RIGHT FRONT (Column 0)...")
#     test_mat = neutral_matrix.copy()
    
#     for _ in range(3):
#         test_mat[1, 0] = 0.5 + 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)
#         test_mat[1, 0] = 0.5 - 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)

#     hardware.set_actuator_postions(neutral_matrix)
#     print(">> CHECK: Did the RIGHT FRONT leg wiggle?")
#     input("Press Enter to continue...")

#     # --- TEST 3: COLUMN 3 (Should be LB) ---
#     print("\n[4/5] Wiggling LEFT BACK (Column 3)...")
#     test_mat = neutral_matrix.copy()
    
#     for _ in range(3):
#         test_mat[1, 3] = 0.5 + 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)
#         test_mat[1, 3] = 0.5 - 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)
        
#     hardware.set_actuator_postions(neutral_matrix)
#     print(">> CHECK: Did the LEFT BACK leg wiggle?")
#     input("Press Enter to continue...")
    
#     # --- TEST 4: COLUMN 2 (Should be RB) ---
#     print("\n[5/5] Wiggling RIGHT BACK (Column 2)...")
#     test_mat = neutral_matrix.copy()
    
#     for _ in range(3):
#         test_mat[1, 2] = 0.5 + 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)
#         test_mat[1, 2] = 0.5 - 0.3
#         hardware.set_actuator_postions(test_mat)
#         time.sleep(0.15)

#     hardware.set_actuator_postions(neutral_matrix)
#     print(">> CHECK: Did the RIGHT BACK leg wiggle?")
#     print("\nTest Complete.")

# if __name__ == "__main__":
#     test_hardware_matrix()


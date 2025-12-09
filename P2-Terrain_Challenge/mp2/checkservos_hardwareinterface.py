import numpy as np
import time
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

def test_hardware_matrix():
    print("=" * 60)
    print("HARDWARE INTERFACE MATRIX TEST")
    print("=" * 60)
    print("We are testing if the Hybrid Controller's Matrix logic matches your robot.")
    print("WARNING: The robot will move! Put it on a stand.")
    
    # Initialize the same way the Hybrid Controller does
    config = Configuration()
    hardware = HardwareInterface()
    
    # MAPPING LOGIC TO TEST
    # We assume MangDang Config uses: 
    # Col 0 = Right Front (RF)
    # Col 1 = Left Front  (LF)
    # Col 2 = Right Back  (RB)
    # Col 3 = Left Back   (LB)
    
    # We will wiggle each column and ask you to confirm which leg moved.
    
    # Default neutral position (approximate standing) in Radians
    # This matrix represents neutral for all 12 servos
    neutral_matrix = np.zeros((3, 4)) 
    neutral_matrix[1, :] = 0.7   # Thighs down
    neutral_matrix[2, :] = -1.4  # Calves back
    
    hardware.set_actuator_postions(neutral_matrix)
    print("\nMoved to neutral. Starting individual leg test in 2 seconds...")
    time.sleep(2)
    
    # --- TEST 1: COLUMN 1 (Should be LF) ---
    print("\n[TEST 1] Moving Matrix Column 1 (We expect LEFT FRONT)...")
    test_mat = neutral_matrix.copy()
    
    # Wiggle Thigh (Row 1, Col 1)
    for _ in range(3):
        test_mat[1, 1] = 0.7 + 0.3 # Flex
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 1] = 0.7 - 0.3 # Extend
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        
    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the LEFT FRONT leg wiggle?")
    input("Press Enter to continue...")

    # --- TEST 2: COLUMN 0 (Should be RF) ---
    print("\n[TEST 2] Moving Matrix Column 0 (We expect RIGHT FRONT)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 0] = 0.7 + 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 0] = 0.7 - 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)

    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the RIGHT FRONT leg wiggle?")
    input("Press Enter to continue...")

    # --- TEST 3: COLUMN 3 (Should be LB) ---
    print("\n[TEST 3] Moving Matrix Column 3 (We expect LEFT BACK)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 3] = 0.7 + 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 3] = 0.7 - 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        
    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the LEFT BACK leg wiggle?")
    input("Press Enter to continue...")
    
    # --- TEST 4: COLUMN 2 (Should be RB) ---
    print("\n[TEST 4] Moving Matrix Column 2 (We expect RIGHT BACK)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 2] = 0.7 + 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 2] = 0.7 - 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)

    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the RIGHT BACK leg wiggle?")
    print("\nTest Complete.")

if __name__ == "__main__":
    test_hardware_matrix()
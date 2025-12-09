import numpy as np
import time
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

def test_hardware_matrix():
    print("=" * 60)
    print("HARDWARE INTERFACE MATRIX TEST (CORRECTED HEIGHT)")
    print("=" * 60)
    print("We are testing if the Hybrid Controller's Matrix logic matches your robot.")
    print("The robot should maintain its standing height.")
    
    # Initialize the same way the Hybrid Controller does
    config = Configuration()
    hardware = HardwareInterface()
    
    # MAPPING LOGIC TO TEST
    # Col 0 = Right Front (RF)
    # Col 1 = Left Front  (LF)
    # Col 2 = Right Back  (RB)
    # Col 3 = Left Back   (LB)
    
    # --- FIX: Match Simulation Default Pose Exactly ---
    # Previously I used 0.7/-1.4 which caused the "dip".
    # Now using 0.785 (45 deg) and -1.57 (90 deg) to match Isaac Sim.
    neutral_matrix = np.zeros((3, 4)) 
    neutral_matrix[1, :] = 0.785   # Thighs (45 degrees)
    neutral_matrix[2, :] = -1.57   # Calves (90 degrees)
    
    print("\nSending Neutral Command (Should match standing pose)...")
    hardware.set_actuator_postions(neutral_matrix)
    time.sleep(2)
    
    # --- TEST 1: COLUMN 1 (Should be LF) ---
    print("\n[TEST 1] Moving Matrix Column 1 (We expect LEFT FRONT)...")
    test_mat = neutral_matrix.copy()
    
    # Wiggle Thigh (Row 1, Col 1)
    for _ in range(3):
        # Wiggle by +/- 0.2 rads so it's visible
        test_mat[1, 1] = 0.785 + 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 1] = 0.785 - 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        
    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the LEFT FRONT leg wiggle?")
    input("Press Enter to continue...")

    # --- TEST 2: COLUMN 0 (Should be RF) ---
    print("\n[TEST 2] Moving Matrix Column 0 (We expect RIGHT FRONT)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 0] = 0.785 + 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 0] = 0.785 - 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)

    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the RIGHT FRONT leg wiggle?")
    input("Press Enter to continue...")

    # --- TEST 3: COLUMN 3 (Should be LB) ---
    print("\n[TEST 3] Moving Matrix Column 3 (We expect LEFT BACK)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 3] = 0.785 + 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 3] = 0.785 - 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        
    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the LEFT BACK leg wiggle?")
    input("Press Enter to continue...")
    
    # --- TEST 4: COLUMN 2 (Should be RB) ---
    print("\n[TEST 4] Moving Matrix Column 2 (We expect RIGHT BACK)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 2] = 0.785 + 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)
        test_mat[1, 2] = 0.785 - 0.2
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.2)

    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the RIGHT BACK leg wiggle?")
    print("\nTest Complete.")

if __name__ == "__main__":
    test_hardware_matrix()
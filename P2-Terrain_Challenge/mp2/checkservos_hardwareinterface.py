import numpy as np
import time
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

def test_hardware_matrix():
    print("=" * 60)
    print("HARDWARE INTERFACE MATRIX TEST (TALL STANCE)")
    print("=" * 60)
    print("Testing mapping with a TALLER neutral pose to prevent crouching.")
    
    config = Configuration()
    hardware = HardwareInterface()
    
    # --- FIX: COMMAND A TALLER STANCE ---
    # Sim Default was: Thigh=0.785, Calf=-1.57 (90 degree bend)
    # New "Tall" Pose: Thigh=0.785, Calf=-1.20 (Straighter leg)
    neutral_matrix = np.zeros((3, 4)) 
    neutral_matrix[1, :] = 0.785   # Thighs (Unchanged)
    neutral_matrix[2, :] = -1.20   # Calves (More positive = Straighter)
    
    print("\n[1/5] Moving to TALL Standing Pose...")
    hardware.set_actuator_postions(neutral_matrix)
    time.sleep(2)
    
    # --- TEST 1: COLUMN 1 (Should be LF) ---
    print("\n[2/5] Wiggling LEFT FRONT (Column 1)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 1] = 0.785 + 0.3 # Wiggle Thigh
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)
        test_mat[1, 1] = 0.785 - 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)
        
    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the LEFT FRONT leg wiggle?")
    input("Press Enter to continue...")

    # --- TEST 2: COLUMN 0 (Should be RF) ---
    print("\n[3/5] Wiggling RIGHT FRONT (Column 0)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 0] = 0.785 + 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)
        test_mat[1, 0] = 0.785 - 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)

    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the RIGHT FRONT leg wiggle?")
    input("Press Enter to continue...")

    # --- TEST 3: COLUMN 3 (Should be LB) ---
    print("\n[4/5] Wiggling LEFT BACK (Column 3)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 3] = 0.785 + 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)
        test_mat[1, 3] = 0.785 - 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)
        
    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the LEFT BACK leg wiggle?")
    input("Press Enter to continue...")
    
    # --- TEST 4: COLUMN 2 (Should be RB) ---
    print("\n[5/5] Wiggling RIGHT BACK (Column 2)...")
    test_mat = neutral_matrix.copy()
    
    for _ in range(3):
        test_mat[1, 2] = 0.785 + 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)
        test_mat[1, 2] = 0.785 - 0.3
        hardware.set_actuator_postions(test_mat)
        time.sleep(0.15)

    hardware.set_actuator_postions(neutral_matrix)
    print(">> CHECK: Did the RIGHT BACK leg wiggle?")
    print("\nTest Complete.")

if __name__ == "__main__":
    test_hardware_matrix()
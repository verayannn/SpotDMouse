import numpy as np
import time
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import ServoParams, PWMParams

class JointDirectionTestHardware:
    def __init__(self):
        self.hardware = HardwareInterface()
        self.servo_params = self.hardware.servo_params
        self.pwm_params = self.hardware.pwm_params
        
        # Joint angles are organized as [axis, leg]
        # axis: 0=hip(abduction/adduction), 1=thigh(flexion/extension), 2=calf(flexion/extension)
        # leg: 0=FR, 1=FL, 2=BR, 3=BL
        
        # Get neutral angles from servo params
        self.neutral_angles = self.servo_params.neutral_angles.copy()
        
        # Map from test order to hardware indices
        # We want to test in order: LF, RF, LB, RB
        # Hardware leg indices: 0=FR, 1=FL, 2=BR, 3=BL
        self.leg_mapping = {
            'LF': 1,  # FL in hardware
            'RF': 0,  # FR in hardware
            'LB': 3,  # BL in hardware
            'RB': 2   # BR in hardware
        }
        
        self.joint_names = [
            "LF hip (base_lf1) - should: +angle = OUTWARD (abduction)",
            "LF thigh (lf1_lf2) - should: +angle = FORWARD (flexion)", 
            "LF calf (lf2_lf3) - should: +angle = FLEXES (bends)",
            "RF hip (base_rf1) - should: +angle = INWARD (adduction)",
            "RF thigh (rf1_rf2) - should: +angle = BACKWARD (extension)",
            "RF calf (rf2_rf3) - should: +angle = EXTENDS (straightens)",
            "LB hip (base_lb1) - should: +angle = INWARD (adduction)",
            "LB thigh (lb1_lb2) - should: +angle = FORWARD (flexion)",
            "LB calf (lb2_lb3) - should: +angle = FLEXES (bends)",
            "RB hip (base_rb1) - should: +angle = OUTWARD (abduction)",
            "RB thigh (rb1_rb2) - should: +angle = BACKWARD (extension)",
            "RB calf (rb2_rb3) - should: +angle = EXTENDS (straightens)",
        ]
        
        # Direction based on the Real mapping from your file
        self.expected_directions = {
            'LF': {'hip': 'OUTWARD', 'thigh': 'FORWARD', 'calf': 'FLEXES'},
            'RF': {'hip': 'INWARD', 'thigh': 'BACKWARD', 'calf': 'EXTENDS'},
            'LB': {'hip': 'INWARD', 'thigh': 'FORWARD', 'calf': 'FLEXES'},
            'RB': {'hip': 'OUTWARD', 'thigh': 'BACKWARD', 'calf': 'EXTENDS'}
        }
        
    def test_single_joint(self, leg_name, axis_index, amplitude=0.3):
        """Test a single joint to verify direction."""
        leg_index = self.leg_mapping[leg_name]
        axis_names = ['hip', 'thigh', 'calf']
        
        print(f"\nTesting {leg_name} {axis_names[axis_index]} (leg_idx={leg_index}, axis_idx={axis_index})")
        print(f"Expected: +angle = {self.expected_directions[leg_name][axis_names[axis_index]]}")
        print("Watch the robot and verify the motion direction...")
        
        # Create joint angle array starting at neutral
        joint_angles = self.neutral_angles.copy()
        
        for t in range(100):
            # Sine wave on one joint
            angle_offset = amplitude * np.sin(t * 0.1)
            joint_angles[axis_index, leg_index] = self.neutral_angles[axis_index, leg_index] + angle_offset
            
            self.hardware.set_actuator_postions(joint_angles)
            time.sleep(0.02)
            
            if t % 25 == 0:
                print(f"  t={t}: angle offset = {angle_offset:.3f} rad")
        
        # Return to neutral
        self.hardware.set_actuator_postions(self.neutral_angles)
        print("Done. Did the joint move in the expected direction?")
        
    def run_full_test(self):
        """Run the complete joint direction test."""
        print("\n" + "="*60)
        print("JOINT DIRECTION VERIFICATION - Hardware Interface")
        print("="*60)
        print("Testing each joint with the REAL robot mapping.")
        print("Verify movements match the expected directions.")
        print("="*60)
        
        # Return to neutral position first
        print("\nSetting to neutral position...")
        self.hardware.set_actuator_postions(self.neutral_angles)
        time.sleep(1)
        
        test_sequence = [
            ('LF', 0), ('LF', 1), ('LF', 2),  # LF hip, thigh, calf
            ('RF', 0), ('RF', 1), ('RF', 2),  # RF hip, thigh, calf
            ('LB', 0), ('LB', 1), ('LB', 2),  # LB hip, thigh, calf
            ('RB', 0), ('RB', 1), ('RB', 2),  # RB hip, thigh, calf
        ]
        
        results = []
        
        for idx, (leg_name, axis_index) in enumerate(test_sequence):
            print(f"\n--- Test {idx}: {self.joint_names[idx]} ---")
            input("Press Enter to test this joint...")
            
            self.test_single_joint(leg_name, axis_index, amplitude=0.3)
            
            result = input("Did it move correctly? (y/n): ").strip().lower()
            results.append((self.joint_names[idx], result == 'y'))
            
            if result != 'y':
                print(f"  *** FAILED - servo multiplier may need adjustment ***")
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"{status}: {name}")
        
        passed_count = sum(1 for _, p in results if p)
        print(f"\nTotal: {passed_count}/12 joints passed")
        print("="*60)


if __name__ == "__main__":
    tester = JointDirectionTestHardware()
    tester.run_full_test()
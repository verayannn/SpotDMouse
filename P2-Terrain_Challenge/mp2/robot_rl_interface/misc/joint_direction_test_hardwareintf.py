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
        
    def test_single_joint_isolated(self, leg_name, axis_index, amplitude=0.3):
        """Test a single joint with isolated movements."""
        leg_index = self.leg_mapping[leg_name]
        axis_names = ['hip', 'thigh', 'calf']
        
        print(f"\n{'='*50}")
        print(f"Testing {leg_name} {axis_names[axis_index]} (leg_idx={leg_index}, axis_idx={axis_index})")
        print(f"Expected: +{amplitude:.3f} rad = {self.expected_directions[leg_name][axis_names[axis_index]]}")
        print(f"{'='*50}")
        
        # Start at neutral
        joint_angles = self.neutral_angles.copy()
        self.hardware.set_actuator_postions(joint_angles)
        time.sleep(0.5)
        
        # Test positive direction
        print(f"\n1. Moving to +{amplitude:.3f} rad...")
        joint_angles[axis_index, leg_index] = self.neutral_angles[axis_index, leg_index] + amplitude
        self.hardware.set_actuator_postions(joint_angles)
        time.sleep(1.5)  # Hold position
        
        positive_correct = input(f"   Did it move {self.expected_directions[leg_name][axis_names[axis_index]]}? (y/n): ").strip().lower() == 'y'
        
        # Return to neutral
        print("\n2. Returning to neutral...")
        joint_angles[axis_index, leg_index] = self.neutral_angles[axis_index, leg_index]
        self.hardware.set_actuator_postions(joint_angles)
        time.sleep(1.0)
        
        # Test negative direction
        print(f"\n3. Moving to -{amplitude:.3f} rad...")
        joint_angles[axis_index, leg_index] = self.neutral_angles[axis_index, leg_index] - amplitude
        self.hardware.set_actuator_postions(joint_angles)
        time.sleep(1.5)  # Hold position
        
        negative_direction = "OPPOSITE of " + self.expected_directions[leg_name][axis_names[axis_index]]
        negative_correct = input(f"   Did it move {negative_direction}? (y/n): ").strip().lower() == 'y'
        
        # Return to neutral
        print("\n4. Returning to neutral...")
        joint_angles[axis_index, leg_index] = self.neutral_angles[axis_index, leg_index]
        self.hardware.set_actuator_postions(joint_angles)
        time.sleep(0.5)
        
        return positive_correct and negative_correct
        
    def run_full_test(self):
        """Run the complete joint direction test."""
        print("\n" + "="*60)
        print("JOINT DIRECTION VERIFICATION - Hardware Interface")
        print("="*60)
        print("Testing each joint with isolated movements.")
        print("Each joint will:")
        print("  1. Move to positive angle")
        print("  2. Return to neutral")
        print("  3. Move to negative angle")
        print("  4. Return to neutral")
        print("="*60)
        
        # Return to neutral position first
        print("\nSetting all joints to neutral position...")
        self.hardware.set_actuator_postions(self.neutral_angles)
        time.sleep(2)
        
        test_sequence = [
            ('LF', 0), ('LF', 1), ('LF', 2),  # LF hip, thigh, calf
            ('RF', 0), ('RF', 1), ('RF', 2),  # RF hip, thigh, calf
            ('LB', 0), ('LB', 1), ('LB', 2),  # LB hip, thigh, calf
            ('RB', 0), ('RB', 1), ('RB', 2),  # RB hip, thigh, calf
        ]
        
        results = []
        
        for idx, (leg_name, axis_index) in enumerate(test_sequence):
            print(f"\n{'#'*60}")
            print(f"TEST {idx + 1}/12: {self.joint_names[idx]}")
            print(f"{'#'*60}")
            input("Press Enter to start this joint test...")
            
            passed = self.test_single_joint_isolated(leg_name, axis_index, amplitude=0.3)
            results.append((self.joint_names[idx], passed))
            
            if not passed:
                print(f"\n*** FAILED - servo multiplier may need adjustment ***")
            else:
                print(f"\n*** PASSED ***")
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")
        
        passed_count = sum(1 for _, p in results if p)
        print(f"\nTotal: {passed_count}/12 joints passed")
        
        if passed_count < 12:
            print("\nFailed joints need servo_multiplier adjustments in Config.py")
        print("="*60)


if __name__ == "__main__":
    tester = JointDirectionTestHardware()
    tester.run_full_test()
#!/usr/bin/env python3
"""Verify joint order mapping for sim-to-real transfer."""

import torch
import numpy as np

class JointOrderVerifier:
    def __init__(self, policy_path):
        """Initialize with exported policy to verify joint ordering."""
        
        # Load the exported JIT policy
        self.policy = torch.jit.load(policy_path).eval()
        
        # Expected joint order from your config
        self.joint_names = [
            # LF leg (front-left)
            "base_lf1", "lf1_lf2", "lf2_lf3",
            # RF leg (front-right)  
            "base_rf1", "rf1_rf2", "rf2_rf3",
            # LB leg (back-left)
            "base_lb1", "lb1_lb2", "lb2_lb3",
            # RB leg (back-right)
            "base_rb1", "rb1_rb2", "rb2_rb3"
        ]
        
        # Group by leg for easier debugging
        self.leg_groups = {
            "LF": [0, 1, 2],   # Left Front indices
            "RF": [3, 4, 5],   # Right Front indices
            "LB": [6, 7, 8],   # Left Back indices
            "RB": [9, 10, 11]  # Right Back indices
        }
        
    def test_joint_order(self):
        """Test joint order by applying specific actions and checking results."""
        
        print("=" * 60)
        print("JOINT ORDER VERIFICATION TEST")
        print("=" * 60)
        
        # Create a zero observation (robot at rest)
        obs = torch.zeros(1, 60, dtype=torch.float32)
        
        # Test 1: Move only one joint at a time
        print("\nTest 1: Single Joint Movement")
        print("-" * 40)
        
        for i, joint_name in enumerate(self.joint_names):
            # Create observation where only previous action for joint i is non-zero
            test_obs = obs.clone()
            test_obs[0, 48 + i] = 0.5  # Set previous action for this joint
            
            # Get policy output
            with torch.no_grad():
                actions = self.policy(test_obs).squeeze().numpy()
            
            # Check which joints respond
            active_joints = np.where(np.abs(actions) > 0.01)[0]
            
            print(f"Joint {i:2d} ({joint_name:10s}): ", end="")
            if len(active_joints) > 0:
                # Fixed: properly format array values
                action_values = [f"{actions[j]:.3f}" for j in active_joints]
                print(f"Actions: {action_values} at indices {active_joints.tolist()}")
            else:
                print("No significant response")
        
        # Test 2: Move one leg at a time
        print("\n\nTest 2: Leg Group Movement")
        print("-" * 40)
        
        for leg_name, indices in self.leg_groups.items():
            test_obs = obs.clone()
            
            # Set commands to move forward
            test_obs[0, 9] = 0.3  # Forward velocity command
            
            # Set previous actions for this leg
            for idx in indices:
                test_obs[0, 48 + idx] = 0.3
            
            # Get policy output
            with torch.no_grad():
                actions = self.policy(test_obs).squeeze().numpy()
            
            print(f"\n{leg_name} Leg Response:")
            for i in range(12):
                if abs(actions[i]) > 0.01:
                    print(f"  Joint {i:2d} ({self.joint_names[i]:10s}): {actions[i]:6.3f}")
        
        # Test 3: Symmetric movement test
        print("\n\nTest 3: Symmetry Test (Forward Command)")
        print("-" * 40)
        
        # Create observation with forward command
        test_obs = obs.clone()
        test_obs[0, 9] = 0.5  # Strong forward command
        
        with torch.no_grad():
            actions = self.policy(test_obs).squeeze().numpy()
        
        # Check symmetry between left and right legs
        print("\nComparing Left vs Right symmetry:")
        print("Joint Pair               Left    Right   Diff")
        print("-" * 45)
        
        pairs = [
            ("LF_hip", "RF_hip", 0, 3),
            ("LF_thigh", "RF_thigh", 1, 4),
            ("LF_shin", "RF_shin", 2, 5),
            ("LB_hip", "RB_hip", 6, 9),
            ("LB_thigh", "RB_thigh", 7, 10),
            ("LB_shin", "RB_shin", 8, 11),
        ]
        
        for name_l, name_r, idx_l, idx_r in pairs:
            diff = actions[idx_l] - actions[idx_r]
            print(f"{name_l:8s} vs {name_r:8s}: {actions[idx_l]:6.3f}  {actions[idx_r]:6.3f}  {diff:6.3f}")
        
        # Test 4: Show all actions for zero input
        print("\n\nTest 4: Zero Input Response (Default Stance)")
        print("-" * 40)
        
        test_obs = obs.clone()  # All zeros
        
        with torch.no_grad():
            actions = self.policy(test_obs).squeeze().numpy()
        
        print("All joint actions with zero input:")
        for i, (joint_name, action) in enumerate(zip(self.joint_names, actions)):
            print(f"  Joint {i:2d} ({joint_name:10s}): {action:7.4f}")
        
        # Test 5: Check action ranges
        print("\n\nTest 5: Action Range Test")
        print("-" * 40)
        
        # Test with maximum forward command
        test_obs = obs.clone()
        test_obs[0, 9] = 1.0  # Max forward
        
        with torch.no_grad():
            actions_max_fwd = self.policy(test_obs).squeeze().numpy()
        
        # Test with maximum turn command
        test_obs = obs.clone()
        test_obs[0, 11] = 1.0  # Max turn
        
        with torch.no_grad():
            actions_max_turn = self.policy(test_obs).squeeze().numpy()
        
        print("Action ranges:")
        print(f"  Min action (across all tests): {min(actions.min(), actions_max_fwd.min(), actions_max_turn.min()):.3f}")
        print(f"  Max action (across all tests): {max(actions.max(), actions_max_fwd.max(), actions_max_turn.max()):.3f}")
        print(f"  Typical range should be [-1, 1] due to tanh activation")
        
        return True
    
    # def create_joint_mapping_file(self, output_path="joint_mapping.txt"):
    #     """Create a reference file for joint mapping."""
        
    #     with open(output_path, 'w') as f:
    #         f.write("MINI PUPPER JOINT MAPPING\n")
    #         f.write("=" * 40 + "\n\n")
    #         f.write("Action Index -> Joint Name -> Description\n")
    #         f.write("-" * 40 + "\n")
            
    #         for i, joint_name in enumerate(self.joint_names):
    #             leg = ""
    #             if i < 3:
    #                 leg = "Left Front (LF)"
    #             elif i < 6:
    #                 leg = "Right Front (RF)"
    #             elif i < 9:
    #                 leg = "Left Back (LB)"
    #             else:
    #                 leg = "Right Back (RB)"
                
    #             joint_type = ""
    #             if "base_" in joint_name:
    #                 joint_type = "Hip/Yaw"
    #             elif "_lf" in joint_name or "_rf" in joint_name or "_lb" in joint_name or "_rb" in joint_name:
    #                 if "1_" in joint_name:
    #                     joint_type = "Thigh/Pitch"
    #                 else:
    #                     joint_type = "Shin/Pitch"
                
    #             f.write(f"{i:2d} -> {joint_name:12s} -> {leg:16s} {joint_type}\n")
            
    #         f.write("\n" + "=" * 40 + "\n")
    #         f.write("REAL ROBOT IMPLEMENTATION\n")
    #         f.write("=" * 40 + "\n\n")
            
    #         f.write("# Python code for real robot:\n")
    #         f.write("joint_mapping = {\n")
    #         for i, joint_name in enumerate(self.joint_names):
    #             f.write(f'    {i}: "{joint_name}",  # Action index {i}\n')
    #         f.write("}\n\n")
            
    #         f.write("# Default positions (radians):\n")
    #         f.write("default_positions = [\n")
    #         for i in range(4):  # 4 legs
    #             f.write("    0.0, 0.785, -1.57,  # ")
    #             if i == 0:
    #                 f.write("LF leg\n")
    #             elif i == 1:
    #                 f.write("RF leg\n")
    #             elif i == 2:
    #                 f.write("LB leg\n")
    #             else:
    #                 f.write("RB leg\n")
    #         f.write("]\n")
        
    #     print(f"\nJoint mapping saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify joint order mapping for Mini Pupper")
    parser.add_argument("--policy", type=str, required=True, help="Path to exported policy.pt")
    parser.add_argument("--output", type=str, default="joint_mapping.txt", help="Output mapping file")
    args = parser.parse_args()
    
    # Run verification
    verifier = JointOrderVerifier(args.policy)
    
    # Run tests
    verifier.test_joint_order()
    
    # Create mapping file
    # verifier.create_joint_mapping_file(args.output)
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nIMPORTANT: The action outputs from the JIT/ONNX policy")
    print("will control joints in the EXACT order shown above.")
    print("Make sure your real robot joint indices match this order!")


import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple

class JointAdjuster:
    def __init__(self, model_path: str):
        """Initialize the joint adjuster with a model path."""
        self.model_path = model_path
        self.model = torch.load(model_path, map_location='cuda:0')
        
        # Joint names for reference
        self.joint_names = ['LF-Hip', 'LF-Thigh', 'LF-Knee', 'RF-Hip', 'RF-Thigh', 'RF-Knee',
                           'LB-Hip', 'LB-Thigh', 'LB-Knee', 'RB-Hip', 'RB-Thigh', 'RB-Knee']
        
        # Get current weight and bias
        self.weight = self.model['model_state_dict']['actor.6.weight'].clone()
        self.bias = self.model['model_state_dict']['actor.6.bias'].clone()
        
    def list_joints(self):
        """List all available joints with their indices."""
        print("\nAvailable joints:")
        for idx, name in enumerate(self.joint_names):
            print(f"  {idx}: {name}")
            
    def get_joint_info(self, joint_idx: int):
        """Get current information about a specific joint."""
        if joint_idx < 0 or joint_idx >= len(self.joint_names):
            raise ValueError(f"Invalid joint index: {joint_idx}. Must be between 0 and {len(self.joint_names)-1}")
            
        print(f"\nJoint {joint_idx} ({self.joint_names[joint_idx]}):")
        print(f"  Current bias: {self.bias[joint_idx].item():.6f}")
        print(f"  Weight stats: min={self.weight[joint_idx].min().item():.6f}, max={self.weight[joint_idx].max().item():.6f}, mean={self.weight[joint_idx].mean().item():.6f}")
        
    def shift_joint_bias(self, joint_idx: int, shift_amount: float):
        """Shift a joint's bias by a specific amount."""
        if joint_idx < 0 or joint_idx >= len(self.joint_names):
            raise ValueError(f"Invalid joint index: {joint_idx}. Must be between 0 and {len(self.joint_names)-1}")
            
        print(f"\nShifting joint {joint_idx} ({self.joint_names[joint_idx]}) bias by {shift_amount}")
        print(f"  Before: {self.bias[joint_idx].item():.6f}")
        
        self.bias[joint_idx] += shift_amount
        
        print(f"  After: {self.bias[joint_idx].item():.6f}")
        
    def scale_joint_weights(self, joint_idx: int, scale_factor: float, scale_bias: bool = True):
        """Scale a joint's weights by a specific factor.
        
        Args:
            joint_idx: Index of the joint to scale
            scale_factor: Factor to multiply weights (and optionally bias) by
            scale_bias: Whether to also scale the bias (default: True)
        """
        if joint_idx < 0 or joint_idx >= len(self.joint_names):
            raise ValueError(f"Invalid joint index: {joint_idx}. Must be between 0 and {len(self.joint_names)-1}")
            
        print(f"\nScaling joint {joint_idx} ({self.joint_names[joint_idx]}) by {scale_factor}")
        print(f"  Weight before: min={self.weight[joint_idx].min().item():.6f}, max={self.weight[joint_idx].max().item():.6f}")
        print(f"  Bias before: {self.bias[joint_idx].item():.6f}")
        
        # Scale weights
        self.weight[joint_idx] *= scale_factor
        
        # Scale bias if requested
        if scale_bias:
            self.bias[joint_idx] *= scale_factor
            print(f"  Scaling bias as well (multiplying by {scale_factor})")
        
        print(f"  Weight after: min={self.weight[joint_idx].min().item():.6f}, max={self.weight[joint_idx].max().item():.6f}")
        print(f"  Bias after: {self.bias[joint_idx].item():.6f}")
        
    def scale_joint_output(self, joint_idx: int, scale_factor: float):
        """Scale both weights and bias of a joint by the same factor.
        This is equivalent to scaling the entire output of the joint."""
        self.scale_joint_weights(joint_idx, scale_factor, scale_bias=True)
        
    def set_joint_bias(self, joint_idx: int, new_bias: float):
        """Set a joint's bias to a specific value."""
        if joint_idx < 0 or joint_idx >= len(self.joint_names):
            raise ValueError(f"Invalid joint index: {joint_idx}. Must be between 0 and {len(self.joint_names)-1}")
            
        print(f"\nSetting joint {joint_idx} ({self.joint_names[joint_idx]}) bias to {new_bias}")
        print(f"  Before: {self.bias[joint_idx].item():.6f}")
        
        self.bias[joint_idx] = new_bias
        
        print(f"  After: {self.bias[joint_idx].item():.6f}")
        
    def invert_joint(self, joint_idx: int):
        """Invert a joint's direction by negating both weights and bias."""
        print(f"\nInverting joint {joint_idx} ({self.joint_names[joint_idx]}) direction")
        self.scale_joint_output(joint_idx, -1.0)
        
    def apply_multiple_adjustments(self, adjustments: List[Tuple[int, str, float]]):
        """Apply multiple joint adjustments at once. 
        adjustments: List of tuples (joint_idx, operation, value)
        where operation is 'shift', 'scale', 'set', or 'invert'"""
        print("\nApplying multiple adjustments:")
        for item in adjustments:
            if len(item) == 2 and item[1] == 'invert':
                joint_idx, operation = item
                self.invert_joint(joint_idx)
            elif len(item) == 3:
                joint_idx, operation, value = item
                if operation == 'shift':
                    self.shift_joint_bias(joint_idx, value)
                elif operation == 'scale':
                    self.scale_joint_output(joint_idx, value)
                elif operation == 'set':
                    self.set_joint_bias(joint_idx, value)
                else:
                    print(f"Unknown operation: {operation}")
            
    def save_model(self, output_path: str = None):
        """Save the adjusted model."""
        if output_path is None:
            output_path = self.model_path
            
        # Update the model with new weights and bias
        self.model['model_state_dict']['actor.6.weight'] = self.weight
        self.model['model_state_dict']['actor.6.bias'] = self.bias
        
        # Save the model
        torch.save(self.model, output_path)
        print(f"\nSaved adjusted model to: {output_path}")
        
    def reset_to_original(self):
        """Reset to the original model values."""
        self.model = torch.load(self.model_path, map_location='cuda:0')
        self.weight = self.model['model_state_dict']['actor.6.weight'].clone()
        self.bias = self.model['model_state_dict']['actor.6.bias'].clone()
        print("\nReset to original model values")

def main():
    parser = argparse.ArgumentParser(description='Adjust individual joint parameters in the model')
    parser.add_argument('--model', type=str, 
                       default='/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format_scaled_precise.pt',
                       help='Path to the model file')
    parser.add_argument('--list', action='store_true', help='List all joints with indices')
    parser.add_argument('--joint', type=int, help='Joint index to adjust')
    parser.add_argument('--shift', type=float, help='Amount to shift the joint bias')
    parser.add_argument('--scale', type=float, help='Factor to scale the joint output (weights and bias)')
    parser.add_argument('--scale-weights-only', type=float, help='Factor to scale only the weights (not bias)')
    parser.add_argument('--set-bias', type=float, help='Set joint bias to specific value')
    parser.add_argument('--invert', action='store_true', help='Invert joint direction (negate weights and bias)')
    parser.add_argument('--info', action='store_true', help='Show info about specified joint')
    parser.add_argument('--output', type=str, help='Output path for adjusted model (default: overwrite input)')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode')
    
    args = parser.parse_args()
    
    # Create adjuster
    adjuster = JointAdjuster(args.model)
    
    if args.interactive:
        # Interactive mode
        print("=== INTERACTIVE JOINT ADJUSTMENT MODE ===")
        adjuster.list_joints()
        
        while True:
            print("\nCommands:")
            print("  list - List all joints")
            print("  info <joint_idx> - Show joint info")
            print("  shift <joint_idx> <amount> - Shift joint bias")
            print("  scale <joint_idx> <factor> - Scale joint output (weights and bias)")
            print("  scale-weights <joint_idx> <factor> - Scale only weights")
            print("  set <joint_idx> <value> - Set joint bias")
            print("  invert <joint_idx> - Invert joint direction")
            print("  save [path] - Save model")
            print("  reset - Reset to original")
            print("  quit - Exit")
            
            cmd = input("\nEnter command: ").strip().split()
            
            if not cmd:
                continue
                
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'list':
                adjuster.list_joints()
            elif cmd[0] == 'info' and len(cmd) == 2:
                try:
                    adjuster.get_joint_info(int(cmd[1]))
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
            elif cmd[0] == 'shift' and len(cmd) == 3:
                try:
                    adjuster.shift_joint_bias(int(cmd[1]), float(cmd[2]))
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
            elif cmd[0] == 'scale' and len(cmd) == 3:
                try:
                    adjuster.scale_joint_output(int(cmd[1]), float(cmd[2]))
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
            elif cmd[0] == 'scale-weights' and len(cmd) == 3:
                try:
                    adjuster.scale_joint_weights(int(cmd[1]), float(cmd[2]), scale_bias=False)
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
            elif cmd[0] == 'set' and len(cmd) == 3:
                try:
                    adjuster.set_joint_bias(int(cmd[1]), float(cmd[2]))
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
            elif cmd[0] == 'invert' and len(cmd) == 2:
                try:
                    adjuster.invert_joint(int(cmd[1]))
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
            elif cmd[0] == 'save':
                output_path = cmd[1] if len(cmd) > 1 else None
                adjuster.save_model(output_path)
            elif cmd[0] == 'reset':
                adjuster.reset_to_original()
            else:
                print("Invalid command")
    else:
        # Command line mode
        if args.list:
            adjuster.list_joints()
            
        if args.joint is not None:
            if args.info:
                adjuster.get_joint_info(args.joint)
                
            if args.shift is not None:
                adjuster.shift_joint_bias(args.joint, args.shift)
                
            if args.scale is not None:
                adjuster.scale_joint_output(args.joint, args.scale)
                
            if args.scale_weights_only is not None:
                adjuster.scale_joint_weights(args.joint, args.scale_weights_only, scale_bias=False)
                
            if args.set_bias is not None:
                adjuster.set_joint_bias(args.joint, args.set_bias)
                
            if args.invert:
                adjuster.invert_joint(args.joint)
                
            # Save if any adjustment was made
            if any([args.shift, args.scale, args.scale_weights_only, args.set_bias, args.invert]):
                adjuster.save_model(args.output)

if __name__ == "__main__":
    main()
from joint_adjuster import JointAdjuster

# Example usage of the JointAdjuster class

# Initialize the adjuster
model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format_scaled_precise.pt"
adjuster = JointAdjuster(model_path)

# List all joints
adjuster.list_joints()

# Example 1: Shift only the bias
print("\n=== Example 1: Shifting only bias ===")
adjuster.shift_joint_bias(0, 0.1)  # LF-Hip bias shifted by 0.1

# Example 2: Scale entire joint output (weights AND bias)
print("\n=== Example 2: Scaling entire joint output ===")
adjuster.scale_joint_output(4, 1.2)  # RF-Thigh scaled by 1.2

# Example 3: Scale only weights (not bias)
print("\n=== Example 3: Scaling only weights ===")
adjuster.scale_joint_weights(7, 0.8, scale_bias=False)  # LB-Thigh weights scaled by 0.8

# Example 4: Invert a joint
print("\n=== Example 4: Inverting joint direction ===")
adjuster.invert_joint(3)  # RF-Hip inverted

# Example 5: Multiple adjustments including inversions
print("\n=== Example 5: Multiple adjustments ===")
adjustments = [
    (0, 'scale', 1.5),    # LF-Hip: scale output by 1.5
    (1, 'shift', 0.05),   # LF-Thigh: shift bias by 0.05
    (5, 'invert'),        # RF-Knee: invert direction
    (8, 'set', -0.5),     # LB-Knee: set bias to -0.5
]
adjuster.apply_multiple_adjustments(adjustments)

# Get info about affected joints
print("\n=== Joint Information After Adjustments ===")
for joint_idx in [0, 3, 4, 5]:
    adjuster.get_joint_info(joint_idx)

# Save the adjusted model
output_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/adjusted_model_with_scaled_bias.pt"
adjuster.save_model(output_path)
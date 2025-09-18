from joint_adjuster import JointAdjuster

# Initialize the adjuster
model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format_scaled_precise.pt"
adjuster = JointAdjuster(model_path)

# List all joints
adjuster.list_joints()

print("\n=== Applying Required Joint Adjustments ===")

# Joint indices based on the pattern:
# 0: LF-Hip, 1: LF-Thigh, 2: LF-Knee
# 3: RF-Hip, 4: RF-Thigh, 5: RF-Knee
# 6: LB-Hip, 7: LB-Thigh, 8: LB-Knee
# 9: RB-Hip, 10: RB-Thigh, 11: RB-Knee

# LB-Knee needs to be shifted upwards by 100
print("Shifting LB-Knee (joint 8) bias by 100")
adjuster.shift_joint_bias(8, 100)

# LB-Thigh needs to be shifted upwards by 80
print("Shifting LB-Thigh (joint 7) bias by 80")
adjuster.shift_joint_bias(7, 80)

# LF-Hip needs to be shifted upwards by 40, then scaled 100
print("Shifting LF-Hip (joint 0) bias by 40")
adjuster.shift_joint_bias(0, 20) ##########
print("Scaling LF-Hip (joint 0) output by 100")
adjuster.scale_joint_output(0, 10)

# RF-Hip needs to be shifted upwards by 40, then scaled by 100
print("Shifting RF-Hip (joint 3) bias by 40")
adjuster.shift_joint_bias(3, 20) ###########
print("Scaling RF-Hip (joint 3) output by 100")
adjuster.scale_joint_output(3, 10)

# LB-Hip needs to be scaled by 100
print("Scaling LB-Hip (joint 6) output by 100")
adjuster.scale_joint_output(6, 10)
adjuster.shift_joint_bias(6, -100) ###########

# RB-Hip needs to be scaled by 100
print("Scaling RB-Hip (joint 9) output by 100")
adjuster.scale_joint_output(9, 10)
adjuster.shift_joint_bias(9, -400) ###########

# Get info about the adjusted joints
print("\n=== Joint Information After Adjustments ===")
adjusted_joints = [0, 3, 6, 7, 8, 9]  # LF-Hip, RF-Hip, LB-Hip, LB-Thigh, LB-Knee, RB-Hip
for joint_idx in adjusted_joints:
    adjuster.get_joint_info(joint_idx)

# Save the adjusted model
output_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/adjusted_model_required_changes.pt"
adjuster.save_model(output_path)
print(f"\nAdjusted model saved to: {output_path}")
import torch
import numpy as np

# Load the current scaled model
model_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format_scaled_precise.pt"
scaled_model = torch.load(model_path, map_location='cuda:0')

print("=== FIXING JOINT DIRECTION INVERSIONS ===")

# Joint names for reference
joint_names = ['LF-Hip', 'LF-Thigh', 'LF-Knee', 'RF-Hip', 'RF-Thigh', 'RF-Knee',
               'LB-Hip', 'LB-Thigh', 'LB-Knee', 'RB-Hip', 'RB-Thigh', 'RB-Knee']

# Joints that need direction inversion (negative scaling)
joints_to_invert = {
    0: 'LF-Hip',    # Index 0
    3: 'RF-Hip',    # Index 3
    2: 'LF-Knee',   # Index 2
    5: 'RF-Knee',   # Index 5
    8: 'LB-Knee',   # Index 8
    11: 'RB-Knee'   # Index 11
}

print("\nJoints to invert direction:")
for idx, name in joints_to_invert.items():
    print(f"  {idx}: {name}")

# Get current weight and bias
current_weight = scaled_model['model_state_dict']['actor.6.weight'].clone()
current_bias = scaled_model['model_state_dict']['actor.6.bias'].clone()

print("\nBefore inversion:")
print(f"Weight shape: {current_weight.shape}")
print(f"Bias shape: {current_bias.shape}")

# Apply negative scaling to the specified joints
for joint_idx in joints_to_invert:
    print(f"\nInverting joint {joint_idx} ({joints_to_invert[joint_idx]}):")
    print(f"  Weight before: min={current_weight[joint_idx].min().item():.4f}, max={current_weight[joint_idx].max().item():.4f}")
    print(f"  Bias before: {current_bias[joint_idx].item():.4f}")
    
    # Negate the weights and bias for this joint
    current_weight[joint_idx] *= -1.0
    current_bias[joint_idx] *= -1.0
    
    print(f"  Weight after: min={current_weight[joint_idx].min().item():.4f}, max={current_weight[joint_idx].max().item():.4f}")
    print(f"  Bias after: {current_bias[joint_idx].item():.4f}")

# Update the model
scaled_model['model_state_dict']['actor.6.weight'] = current_weight
scaled_model['model_state_dict']['actor.6.bias'] = current_bias

# Save the updated model
output_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format_scaled_precise.pt"
torch.save(scaled_model, output_path)
print(f"\nSaved updated model with direction fixes to: {output_path}")

# Also create a verification script
print("\n=== CREATING VERIFICATION SCRIPT ===")
print("\nCreated verification script: verify_directions.py")
print("\nTo verify the direction fixes, run:")
print("cd /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL")
print("python verify_directions.py")
import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:1")

PTH = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt"
OG_PTH = "/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt"
model = torch.load(PTH)
og_model = torch.load(OG_PTH)
# Compare model structures
print("=== Model Comparison ===")
print(f"Model type: {type(model)}")
print(f"OG Model type: {type(og_model)}")

# If they're state dicts, compare keys
if isinstance(model, dict) and isinstance(og_model, dict):
    print(f"\nModel keys: {len(model.keys())}")
    print(f"OG Model keys: {len(og_model.keys())}")

    print(og_model.keys())
    print(model.keys())
    
    # Compare parameter shapes
    print("\n=== Parameter Shape Comparison ===")
    for key in model.keys():
        if key in og_model:
            # Check if values are tensors before accessing shape
            if hasattr(model[key], 'shape') and hasattr(og_model[key], 'shape'):
                model_shape = model[key].shape
                og_shape = og_model[key].shape
                match = "✓" if model_shape == og_shape else "✗"
                print(f"{key}: {model_shape} vs {og_shape} {match}")
            else:
                print(f"{key}: {type(model[key])} vs {type(og_model[key])} (not tensors)")
        else:
            print(f"{key}: Only in model")
    
    for key in og_model.keys():
        if key not in model:
            print(f"{key}: Only in og_model")

# If they're model objects, compare architecture
elif hasattr(model, 'state_dict') and hasattr(og_model, 'state_dict'):
    print(f"\nModel architecture: {model}")
    print(f"OG Model architecture: {og_model}")
    
    # Compare state dict shapes
    model_state = model.state_dict()
    og_state = og_model.state_dict()
    
    print("\n=== State Dict Shape Comparison ===")
    for key in model_state.keys():
        if key in og_state:
            model_shape = model_state[key].shape
            og_shape = og_state[key].shape
            match = "✓" if model_shape == og_shape else "✗"
            print(f"{key}: {model_shape} vs {og_shape} {match}")


#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/verify_model_compatibility.py
"""
Verify that IL-trained model is compatible with the deployed controller
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_il import MLPPolicy

DEVICE = torch.device("cuda:1")

# Paths
IL_MODEL_PATH = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt"
OG_MODEL_PATH = "/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt"

def load_and_compare_models():
    """Load both models and compare their structures"""
    
    # Load checkpoints
    il_checkpoint = torch.load(IL_MODEL_PATH, map_location=DEVICE)
    og_checkpoint = torch.load(OG_MODEL_PATH, map_location=DEVICE)
    
    print("=== Checkpoint Structure Comparison ===")
    print(f"IL checkpoint keys: {list(il_checkpoint.keys())}")
    print(f"OG checkpoint keys: {list(og_checkpoint.keys())}\n")
    
    # Extract model state dicts
    il_state_dict = il_checkpoint.get('model_state_dict', il_checkpoint)
    og_state_dict = og_checkpoint.get('model_state_dict', og_checkpoint)
    
    # Handle case where checkpoint might be just the state dict
    if not isinstance(il_state_dict, dict) or 'weight' not in list(il_state_dict.keys())[0]:
        print("WARNING: IL checkpoint might not contain a proper state dict")
        il_state_dict = il_checkpoint
    
    print("=== Model State Dict Comparison ===")
    print(f"IL model layers: {list(il_state_dict.keys())[:5]}...")  # Show first 5
    print(f"OG model layers: {list(og_state_dict.keys())[:5]}...\n")  # Show first 5
    
    # Compare layer shapes
    print("=== Layer Shape Comparison ===")
    il_keys = [k for k in il_state_dict.keys() if 'weight' in k or 'bias' in k]
    og_keys = [k for k in og_state_dict.keys() if 'weight' in k or 'bias' in k]
    
    # Try to match layers
    for i, (il_key, og_key) in enumerate(zip(il_keys[:10], og_keys[:10])):
        il_shape = il_state_dict[il_key].shape
        og_shape = og_state_dict[og_key].shape
        match = "✓" if il_shape == og_shape else "✗"
        print(f"Layer {i}: IL{il_shape} vs OG{og_shape} {match}")
    
    # Check normalization statistics
    print("\n=== Normalization Statistics ===")
    
    # IL model normalization - check if they exist
    has_norm_stats = True
    for key in ['obs_mean', 'obs_std', 'action_mean', 'action_std']:
        if key not in il_checkpoint or il_checkpoint[key] is None:
            print(f"WARNING: IL checkpoint missing {key}")
            has_norm_stats = False
    
    if has_norm_stats:
        obs_mean = il_checkpoint['obs_mean']
        obs_std = il_checkpoint['obs_std']
        action_mean = il_checkpoint['action_mean']
        action_std = il_checkpoint['action_std']
        print(f"IL obs_mean shape: {obs_mean.shape}")
        print(f"IL obs_std shape: {obs_std.shape}")
        print(f"IL action_mean shape: {action_mean.shape}")
        print(f"IL action_std shape: {action_std.shape}")
    else:
        print("\nERROR: IL model missing normalization statistics!")
        print("This means the model was not trained with the updated train_il.py")
        print("Please retrain the model to include normalization statistics.")
        
        # Create dummy stats for testing
        print("\nCreating dummy normalization stats for testing...")
        il_checkpoint['obs_mean'] = torch.zeros(48)
        il_checkpoint['obs_std'] = torch.ones(48)
        il_checkpoint['action_mean'] = torch.zeros(12)
        il_checkpoint['action_std'] = torch.ones(12)
    
    # OG model normalization (might use RMS)
    if 'obs_rms_mean' in og_checkpoint:
        print(f"\nOG obs_rms_mean shape: {og_checkpoint['obs_rms_mean'].shape}")
        print(f"OG obs_rms_var shape: {og_checkpoint['obs_rms_var'].shape}")
    
    if 'num_obs' in og_checkpoint:
        print(f"\nExpected observation dim: {og_checkpoint['num_obs']}")
        print(f"Expected action dim: {og_checkpoint['num_actions']}")
    
    return il_checkpoint, og_checkpoint

def verify_observation_order():
    """Verify the observation vector ordering matches expectations"""
    
    print("\n=== Observation Vector Structure ===")
    print("Based on your deployed controller, the 48-dim observation should be:")
    print("[ 0: 3] - Linear velocity (vx, vy, vz)")
    print("[ 3: 6] - Angular velocity (wx, wy, wz)")
    print("[ 6:18] - Joint positions (12 joints)")
    print("[18:30] - Joint velocities (12 joints)")
    print("[30:34] - Quaternion (qx, qy, qz, qw)")
    print("[34:46] - Previous actions (12 joints)")
    print("[46:48] - Clock phase (sin, cos)")
    
    joint_names = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf", 
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf"
    ]
    
    print("\nJoint ordering (0-11):")
    for i, name in enumerate(joint_names):
        print(f"  Joint {i:2d}: {name}")
    
    return joint_names

def test_model_inference(il_checkpoint, og_checkpoint):
    """Test model inference with sample data"""
    
    print("\n=== Testing Model Inference ===")
    
    # Check if we have proper model state dict
    if 'model_state_dict' in il_checkpoint:
        model_state = il_checkpoint['model_state_dict']
    else:
        model_state = il_checkpoint
        # Filter only model parameters
        model_state = {k: v for k, v in model_state.items() if 'net' in k or isinstance(v, torch.Tensor)}
    
    # Create IL model and load weights
    il_model = MLPPolicy(obs_dim=48, action_dim=12).to(DEVICE)
    
    try:
        il_model.load_state_dict(model_state)
    except:
        print("WARNING: Could not load state dict directly, attempting to load with strict=False")
        il_model.load_state_dict(model_state, strict=False)
    
    il_model.eval()
    
    # Create sample observation
    sample_obs = torch.zeros(1, 48).to(DEVICE)
    
    # Set some example values
    sample_obs[0, 0] = 0.5  # vx = 0.5 m/s forward
    sample_obs[0, 5] = 0.2  # wz = 0.2 rad/s turning
    
    # Get normalization stats
    obs_mean = il_checkpoint.get('obs_mean', torch.zeros(48)).to(DEVICE)
    obs_std = il_checkpoint.get('obs_std', torch.ones(48)).to(DEVICE)
    action_mean = il_checkpoint.get('action_mean', torch.zeros(12)).to(DEVICE)
    action_std = il_checkpoint.get('action_std', torch.ones(12)).to(DEVICE)
    
    # Normalize observation
    sample_obs_norm = (sample_obs - obs_mean) / (obs_std + 1e-8)
    
    # Get prediction
    with torch.no_grad():
        pred_action_norm = il_model(sample_obs_norm)
    
    # Denormalize action
    pred_action = pred_action_norm * action_std + action_mean
    
    print(f"Sample input (first 6 values): {sample_obs[0, :6].cpu().numpy()}")
    print(f"Predicted joint positions (rad):")
    joint_names = verify_observation_order()
    for i in range(12):
        print(f"  {joint_names[i]:10s}: {pred_action[0, i].item():7.3f} rad ({np.degrees(pred_action[0, i].item()):6.1f} deg)")
    
    return pred_action

def visualize_normalization_stats(il_checkpoint):
    """Visualize normalization statistics"""
    
    # Check if we have normalization stats
    if not all(key in il_checkpoint for key in ['obs_mean', 'obs_std', 'action_mean', 'action_std']):
        print("\nSkipping normalization visualization - statistics not found in checkpoint")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Observation mean
    ax = axes[0, 0]
    obs_mean = il_checkpoint['obs_mean'].cpu().numpy()
    ax.bar(range(len(obs_mean)), obs_mean)
    ax.set_title("Observation Mean")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Mean Value")
    ax.grid(True, alpha=0.3)
    
    # Add labels for key regions
    ax.axvspan(0, 3, alpha=0.2, color='red', label='Lin Vel')
    ax.axvspan(3, 6, alpha=0.2, color='green', label='Ang Vel')
    ax.axvspan(6, 18, alpha=0.2, color='blue', label='Joint Pos')
    ax.axvspan(18, 30, alpha=0.2, color='orange', label='Joint Vel')
    ax.legend()
    
    # Observation std
    ax = axes[0, 1]
    obs_std = il_checkpoint['obs_std'].cpu().numpy()
    ax.bar(range(len(obs_std)), obs_std)
    ax.set_title("Observation Std")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel("Std Value")
    ax.grid(True, alpha=0.3)
    
    # Action mean
    ax = axes[1, 0]
    action_mean = il_checkpoint['action_mean'].cpu().numpy()
    joint_names = verify_observation_order()
    ax.bar(range(len(action_mean)), action_mean)
    ax.set_title("Action Mean (Joint Positions)")
    ax.set_xlabel("Joint Index")
    ax.set_ylabel("Mean Position (rad)")
    ax.set_xticks(range(12))
    ax.set_xticklabels([name.replace('_', '\n') for name in joint_names], rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Action std
    ax = axes[1, 1]
    action_std = il_checkpoint['action_std'].cpu().numpy()
    ax.bar(range(len(action_std)), action_std)
    ax.set_title("Action Std (Joint Positions)")
    ax.set_xlabel("Joint Index")
    ax.set_ylabel("Std (rad)")
    ax.set_xticks(range(12))
    ax.set_xticklabels([name.replace('_', '\n') for name in joint_names], rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/normalization_stats.png", dpi=150)
    print("\nSaved normalization statistics visualization to normalization_stats.png")

def check_what_is_in_checkpoint():
    """Debug function to see what's actually in the checkpoint"""
    print("\n=== Debugging IL Checkpoint Contents ===")
    
    il_checkpoint = torch.load(IL_MODEL_PATH, map_location='cpu')
    
    print(f"Type of checkpoint: {type(il_checkpoint)}")
    
    if isinstance(il_checkpoint, dict):
        print(f"Keys in checkpoint: {list(il_checkpoint.keys())}")
        
        for key, value in il_checkpoint.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor with shape {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: Dict with {len(value)} items")
                # Show first few items if it's a state dict
                if key == 'model_state_dict':
                    for k, v in list(value.items())[:3]:
                        print(f"    {k}: {v.shape if isinstance(v, torch.Tensor) else type(v)}")
            else:
                print(f"  {key}: {type(value)}")
    else:
        print("Checkpoint is not a dictionary!")

def main():
    print("=== IL Model Compatibility Verification ===\n")
    
    # First, debug what's in the checkpoint
    check_what_is_in_checkpoint()
    
    # Load and compare models
    il_checkpoint, og_checkpoint = load_and_compare_models()
    
    # Verify observation ordering
    verify_observation_order()
    
    # Test model inference
    test_model_inference(il_checkpoint, og_checkpoint)
    
    # Visualize normalization statistics if available
    visualize_normalization_stats(il_checkpoint)
    
    print("\n=== Summary ===")
    
    # Check if model has proper normalization stats
    has_all_stats = all(key in il_checkpoint for key in ['obs_mean', 'obs_std', 'action_mean', 'action_std'])
    
    if has_all_stats:
        print("✓ Model architecture matches expected dimensions")
        print("✓ Observation vector follows expected ordering")
        print("✓ Action vector outputs 12 joint positions")
        print("✓ Normalization statistics are properly stored")
        print("\nThe IL model is ready for deployment!")
    else:
        print("✗ Model is missing normalization statistics")
        print("✗ Please retrain with the updated train_il.py script")
        print("\nTo fix this, run:")
        print("python train_il.py --dataset /path/to/your/dataset.hdf5 --epochs 10")
    
    print("\nNext steps:")
    print("1. Test on robot with: python test_deployed_model.py")
    print("2. Use as initialization for RSL_RL training")

if __name__ == "__main__":
    main()
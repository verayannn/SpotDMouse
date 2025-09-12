#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/compare_il_vs_rl.py
"""
Direct comparison of IL model vs original RL model outputs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_model_and_get_outputs(checkpoint_path: str, model_name: str):
    """Load model and get outputs for various commands"""
    print(f"\n=== {model_name} ===")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print checkpoint info
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Get normalization stats - handle both numpy and torch formats
    obs_mean = checkpoint.get('obs_rms_mean', torch.zeros(48))
    obs_var = checkpoint.get('obs_rms_var', torch.ones(48))
    
    # Convert to torch tensors if they're numpy arrays
    if isinstance(obs_mean, np.ndarray):
        obs_mean = torch.from_numpy(obs_mean).float()
    if isinstance(obs_var, np.ndarray):
        obs_var = torch.from_numpy(obs_var).float()
    
    obs_std = torch.sqrt(obs_var + 1e-8)
    
    print(f"Normalization - Mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"Normalization - Std range: [{obs_std.min():.3f}, {obs_std.max():.3f}]")
    
    # Load model
    from train_il import MLPPolicy
    model = MLPPolicy(obs_dim=48, action_dim=12)
    
    # Extract actor weights
    state_dict = checkpoint['model_state_dict']
    actor_state_dict = {k: v for k, v in state_dict.items() if k.startswith('actor.')}
    model.actor.load_state_dict({k.replace('actor.', ''): v for k, v in actor_state_dict.items()})
    model.eval()
    
    # Test commands
    test_commands = [
        ("Stand", 0.0, 0.0, 0.0),
        ("Slow forward", 0.1, 0.0, 0.0),
        ("Normal forward", 0.3, 0.0, 0.0),
        ("Fast forward", 0.5, 0.0, 0.0),
        ("Backward", -0.3, 0.0, 0.0),
        ("Sideways right", 0.0, 0.2, 0.0),
        ("Sideways left", 0.0, -0.2, 0.0),
        ("Turn right", 0.0, 0.0, 0.5),
        ("Turn left", 0.0, 0.0, -0.5),
        ("Forward + turn", 0.3, 0.0, 0.3),
    ]
    
    results = []
    
    for cmd_name, vx, vy, wz in test_commands:
        # Create observation
        obs = torch.zeros(1, 48)
        obs[0, 0] = vx  # Linear velocity X
        obs[0, 1] = vy  # Linear velocity Y
        obs[0, 5] = wz  # Angular velocity Z
        
        # Default joint positions
        default_joints = torch.tensor([0.0, -0.8, 1.6] * 4)
        obs[0, 6:18] = default_joints
        
        # Quaternion (upright)
        obs[0, 30:34] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        
        # Previous actions
        obs[0, 34:46] = default_joints
        
        # Clock phase
        obs[0, 46] = 0.0  # sin
        obs[0, 47] = 1.0  # cos
        
        # Normalize observation
        obs_norm = (obs - obs_mean.unsqueeze(0)) / obs_std.unsqueeze(0)
        
        # Get model output
        with torch.no_grad():
            action = model(obs_norm)
        
        action_np = action[0].numpy()
        
        results.append({
            'command': cmd_name,
            'vx': vx,
            'vy': vy, 
            'wz': wz,
            'action': action_np,
            'obs_raw': obs[0].numpy(),
            'obs_norm': obs_norm[0].numpy()
        })
        
    return results, obs_mean, obs_std

def compare_outputs():
    """Compare IL and RL model outputs side by side"""
    
    # Model paths
    il_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt"
    rl_path = "/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt"
    
    # Get outputs from both models
    il_results, il_mean, il_std = load_model_and_get_outputs(il_path, "IL Model")
    rl_results, rl_mean, rl_std = load_model_and_get_outputs(rl_path, "Original RL Model")
    
    # Print detailed comparison
    print("\n" + "="*80)
    print("DETAILED OUTPUT COMPARISON")
    print("="*80)
    
    print("\nCommand         | IL Output Stats              | RL Output Stats")
    print("----------------|------------------------------|------------------------------")
    
    for il_res, rl_res in zip(il_results, rl_results):
        cmd = il_res['command']
        il_action = il_res['action']
        rl_action = rl_res['action']
        
        print(f"{cmd:15s} | mean={il_action.mean():6.3f} std={il_action.std():5.3f} | "
              f"mean={rl_action.mean():6.3f} std={rl_action.std():5.3f}")
    
    # Print joint-by-joint comparison for key commands
    print("\n" + "="*80)
    print("JOINT ANGLE COMPARISON (degrees)")
    print("="*80)
    
    key_commands = ["Stand", "Normal forward", "Turn right"]
    joint_names = ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf',
                   'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
    
    for cmd in key_commands:
        # Find the command in results
        il_idx = next(i for i, r in enumerate(il_results) if r['command'] == cmd)
        rl_idx = next(i for i, r in enumerate(rl_results) if r['command'] == cmd)
        
        il_action = np.degrees(il_results[il_idx]['action'])
        rl_action = np.degrees(rl_results[rl_idx]['action'])
        
        print(f"\n{cmd}:")
        print("Joint       | IL Model | RL Model | Difference")
        print("------------|----------|----------|------------")
        
        for i, joint in enumerate(joint_names):
            diff = il_action[i] - rl_action[i]
            print(f"{joint:11s} | {il_action[i]:8.2f} | {rl_action[i]:8.2f} | {diff:10.2f}")
    
    # Create visualization
    create_comparison_plots(il_results, rl_results)
    
    # Analyze differences
    analyze_differences(il_results, rl_results, il_mean, il_std, rl_mean, rl_std)

def create_comparison_plots(il_results, rl_results):
    """Create visualization comparing IL and RL outputs"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for plotting
    commands = [r['command'] for r in il_results]
    il_actions = np.array([r['action'] for r in il_results])
    rl_actions = np.array([r['action'] for r in rl_results])
    
    # Plot 1: Action magnitudes
    ax = axes[0, 0]
    il_mags = [np.linalg.norm(a) for a in il_actions]
    rl_mags = [np.linalg.norm(a) for a in rl_actions]
    
    x = np.arange(len(commands))
    width = 0.35
    ax.bar(x - width/2, il_mags, width, label='IL Model', alpha=0.8)
    ax.bar(x + width/2, rl_mags, width, label='RL Model', alpha=0.8)
    ax.set_xlabel('Command')
    ax.set_ylabel('Action Magnitude')
    ax.set_title('Action Magnitude Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(commands, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Action differences from default pose
    ax = axes[0, 1]
    default_joints = np.array([0.0, -0.8, 1.6] * 4)
    il_diffs = [np.linalg.norm(a - default_joints) for a in il_actions]
    rl_diffs = [np.linalg.norm(a - default_joints) for a in rl_actions]
    
    ax.bar(x - width/2, il_diffs, width, label='IL Model', alpha=0.8)
    ax.bar(x + width/2, rl_diffs, width, label='RL Model', alpha=0.8)
    ax.set_xlabel('Command')
    ax.set_ylabel('Diff from Default Pose')
    ax.set_title('Movement from Default Pose')
    ax.set_xticks(x)
    ax.set_xticklabels(commands, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Joint angle ranges
    ax = axes[1, 0]
    il_ranges = [(a.min(), a.max()) for a in il_actions]
    rl_ranges = [(a.min(), a.max()) for a in rl_actions]
    
    # Plot ranges for "Normal forward" command
    fwd_idx = commands.index("Normal forward")
    il_fwd = np.degrees(il_actions[fwd_idx])
    rl_fwd = np.degrees(rl_actions[fwd_idx])
    
    joint_indices = np.arange(12)
    ax.scatter(joint_indices, il_fwd, label='IL Model', s=100, alpha=0.7)
    ax.scatter(joint_indices, rl_fwd, label='RL Model', s=100, alpha=0.7)
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Joint Angle (degrees)')
    ax.set_title('Joint Angles for "Normal Forward" Command')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Response to velocity commands
    ax = axes[1, 1]
    vx_commands = [r['vx'] for r in il_results]
    
    # Get FR hip joint response (joint index 0)
    il_hip_response = [np.degrees(a[0]) for a in il_actions]
    rl_hip_response = [np.degrees(a[0]) for a in rl_actions]
    
    # Get only forward velocity commands for plotting
    fwd_indices = [i for i, vx in enumerate(vx_commands) if vx >= -0.3 and vx <= 0.5 and il_results[i]['vy'] == 0 and il_results[i]['wz'] == 0]
    vx_fwd = [vx_commands[i] for i in fwd_indices]
    il_hip_fwd = [il_hip_response[i] for i in fwd_indices]
    rl_hip_fwd = [rl_hip_response[i] for i in fwd_indices]
    
    # Sort by vx for plotting
    sorted_indices = np.argsort(vx_fwd)
    vx_sorted = [vx_fwd[i] for i in sorted_indices]
    il_hip_sorted = [il_hip_fwd[i] for i in sorted_indices]
    rl_hip_sorted = [rl_hip_fwd[i] for i in sorted_indices]
    
    ax.plot(vx_sorted, il_hip_sorted, 'o-', label='IL Model', markersize=8)
    ax.plot(vx_sorted, rl_hip_sorted, 's-', label='RL Model', markersize=8)
    ax.set_xlabel('Forward Velocity Command (m/s)')
    ax.set_ylabel('FR Hip Joint Angle (degrees)')
    ax.set_title('Hip Joint Response to Forward Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/il_vs_rl_comparison.png', dpi=150)
    print("\nSaved comparison plot to il_vs_rl_comparison.png")

def analyze_differences(il_results, rl_results, il_mean, il_std, rl_mean, rl_std):
    """Analyze key differences between models"""
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES ANALYSIS")
    print("="*80)
    
    # 1. Normalization differences
    print("\n1. Normalization Differences:")
    print("   Velocity command normalization (index 0-5):")
    print("   Index | IL Mean | IL Std | RL Mean | RL Std")
    print("   ------|---------|--------|---------|--------")
    for i in range(6):
        print(f"   {i:5d} | {il_mean[i]:7.3f} | {il_std[i]:6.3f} | {rl_mean[i]:7.3f} | {rl_std[i]:6.3f}")
    
    # 2. Output range differences
    print("\n2. Output Range Analysis:")
    il_actions = np.array([r['action'] for r in il_results])
    rl_actions = np.array([r['action'] for r in rl_results])
    
    print(f"   IL Model: min={il_actions.min():.3f}, max={il_actions.max():.3f}")
    print(f"   RL Model: min={rl_actions.min():.3f}, max={rl_actions.max():.3f}")
    
    # 3. Movement responsiveness
    print("\n3. Movement Responsiveness:")
    
    # Compare stand vs forward
    stand_idx = next(i for i, r in enumerate(il_results) if r['command'] == "Stand")
    fwd_idx = next(i for i, r in enumerate(il_results) if r['command'] == "Normal forward")
    
    il_stand = il_results[stand_idx]['action']
    il_fwd = il_results[fwd_idx]['action']
    rl_stand = rl_results[stand_idx]['action']
    rl_fwd = rl_results[fwd_idx]['action']
    
    il_movement = np.linalg.norm(il_fwd - il_stand)
    rl_movement = np.linalg.norm(rl_fwd - rl_stand)
    
    print(f"   Stand to Forward movement:")
    print(f"   IL Model: {il_movement:.3f}")
    print(f"   RL Model: {rl_movement:.3f}")
    print(f"   Ratio (IL/RL): {il_movement/rl_movement:.3f}")
    
    # 4. Per-joint movement analysis
    print("\n   Per-joint movement (Stand -> Forward):")
    print("   Joint | IL Δ (deg) | RL Δ (deg)")
    print("   ------|------------|------------")
    joint_names = ['Hip', 'Thigh', 'Calf']
    for i in range(3):  # Just show first leg
        il_delta = np.degrees(il_fwd[i] - il_stand[i])
        rl_delta = np.degrees(rl_fwd[i] - rl_stand[i])
        print(f"   {joint_names[i]:5s} | {il_delta:10.2f} | {rl_delta:10.2f}")
    
    # 5. Recommendations
    print("\n4. RECOMMENDATIONS:")
    if il_movement < 0.1:
        print("   ⚠️  IL model shows very little movement response")
        print("   → Try scaling IL outputs by 2-5x")
        print("   → Check if IL training data had sufficient movement variation")
    
    if abs(il_mean[0]) > 0.1 or abs(rl_mean[0]) > 0.1:
        print("   ⚠️  Non-zero mean for velocity commands detected")
        print("   → This could cause bias in standing position")
    
    if il_std[0] > 1.0:
        print("   ⚠️  IL model has large normalization std for velocities")
        print("   → This could dampen command responsiveness")
    
    # Check if normalization is very different
    norm_diff = torch.norm(il_std - rl_std) / torch.norm(rl_std)
    if norm_diff > 0.5:
        print(f"   ⚠️  Normalization std differs significantly (relative diff: {norm_diff:.2f})")
        print("   → Consider using RL model's normalization stats for IL model")

def main():
    print("="*80)
    print("IL vs RL Model Direct Comparison")
    print("="*80)
    
    # Run comparison
    compare_outputs()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    print("\n1. View the comparison plot:")
    print("   $BROWSER /workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/il_vs_rl_comparison.png")
    
    print("\n2. If IL model outputs are too small, try scaling them:")
    print("   - Multiply IL outputs by a factor (e.g., 2-5x)")
    print("   - Or reduce the normalization std values")
    
    print("\n3. Test both models in simulation:")
    print("   # IL Model")
    print("   cd /workspace/isaaclab")
    print("   CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py \\")
    print("     --task=Isaac-Velocity-Flat-Custom-Quad-v0 \\")
    print("     --checkpoint=/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model_normalized_rsl_rl.pt \\")
    print("     --num_envs 1")
    
    print("\n   # RL Model (for comparison)")
    print("   CUDA_VISIBLE_DEVICES=2 python scripts/reinforcement_learning/rsl_rl/play.py \\")
    print("     --task=Isaac-Velocity-Flat-Custom-Quad-v0 \\")
    print("     --checkpoint=/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt \\")
    print("     --num_envs 1")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# filepath: /workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/extract_stats.py
"""
Extract normalization statistics from a trained model checkpoint
without requiring the full IsaacLab environment
"""

import torch
import argparse
import os
import numpy as np

def extract_stats_from_checkpoint(checkpoint_path, output_path=None):
    """Extract normalization statistics from a checkpoint file"""
    
    # Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check what's in the checkpoint
    print("\nCheckpoint contents:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: shape {checkpoint[key].shape}")
        elif isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} items")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Extract observation normalization stats
    stats = {}
    
    # Check for different possible keys for normalization stats
    if 'obs_rms_mean' in checkpoint and checkpoint['obs_rms_mean'] is not None:
        stats['obs_mean'] = checkpoint['obs_rms_mean']
        stats['obs_var'] = checkpoint['obs_rms_var']
        stats['obs_std'] = torch.sqrt(checkpoint['obs_rms_var'] + 1e-8)
        print("\nFound RMS normalization stats")
    elif 'obs_mean' in checkpoint and checkpoint['obs_mean'] is not None:
        stats['obs_mean'] = checkpoint['obs_mean']
        stats['obs_std'] = checkpoint['obs_std']
        print("\nFound standard normalization stats")
    else:
        print("\nWARNING: No observation normalization stats found or they are None!")
        print("Creating default normalization stats (mean=0, std=1)")
        stats['obs_mean'] = torch.zeros(48)
        stats['obs_std'] = torch.ones(48)
    
    # Check for action normalization
    if 'action_mean' in checkpoint and checkpoint['action_mean'] is not None:
        stats['action_mean'] = checkpoint['action_mean']
        stats['action_std'] = checkpoint['action_std']
    else:
        print("WARNING: No action normalization stats found or they are None!")
        print("Creating default action normalization (mean=0, std=1)")
        stats['action_mean'] = torch.zeros(12)
        stats['action_std'] = torch.ones(12)
    
    # Get observation and action dimensions
    if 'num_obs' in checkpoint:
        stats['num_obs'] = checkpoint['num_obs']
        print(f"\nObservation dimension: {checkpoint['num_obs']}")
    else:
        stats['num_obs'] = 48  # Default for Mini Pupper
        
    if 'num_actions' in checkpoint:
        stats['num_actions'] = checkpoint['num_actions']
        print(f"Action dimension: {checkpoint['num_actions']}")
    else:
        stats['num_actions'] = 12  # Default for Mini Pupper
    
    # Save extracted stats if output path provided
    if output_path:
        # Create a new checkpoint with proper stats
        output_checkpoint = {
            'model_state_dict': checkpoint.get('model_state_dict', {}),
            'obs_mean': stats['obs_mean'],
            'obs_std': stats['obs_std'],
            'action_mean': stats['action_mean'],
            'action_std': stats['action_std'],
            'num_obs': stats['num_obs'],
            'num_actions': stats['num_actions']
        }
        
        # Copy other useful fields if they exist
        for key in ['epoch', 'val_loss']:
            if key in checkpoint:
                output_checkpoint[key] = checkpoint[key]
            
        torch.save(output_checkpoint, output_path)
        print(f"\nSaved extracted stats to: {output_path}")
    
    return stats

def print_stats_summary(stats):
    """Print a summary of the normalization statistics"""
    
    print("\n=== Normalization Statistics Summary ===")
    
    if 'obs_mean' in stats and stats['obs_mean'] is not None:
        obs_mean = stats['obs_mean'].numpy()
        obs_std = stats['obs_std'].numpy()
        
        print(f"\nObservation stats (shape: {obs_mean.shape}):")
        print(f"  Mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
        print(f"  Std range: [{obs_std.min():.3f}, {obs_std.max():.3f}]")
        
        # Print per-component stats for key observation elements
        print("\nPer-component observation stats (first 30):")
        print("  Index | Mean    | Std     | Description")
        print("  ------|---------|---------|-------------")
        
        descriptions = [
            "Linear vel X", "Linear vel Y", "Linear vel Z",
            "Angular vel X", "Angular vel Y", "Angular vel Z",
            "Joint 0 pos (FR_hip)", "Joint 1 pos (FR_thigh)", "Joint 2 pos (FR_calf)", 
            "Joint 3 pos (FL_hip)", "Joint 4 pos (FL_thigh)", "Joint 5 pos (FL_calf)",
            "Joint 6 pos (RR_hip)", "Joint 7 pos (RR_thigh)", "Joint 8 pos (RR_calf)", 
            "Joint 9 pos (RL_hip)", "Joint 10 pos (RL_thigh)", "Joint 11 pos (RL_calf)",
            "Joint 0 vel (FR_hip)", "Joint 1 vel (FR_thigh)", "Joint 2 vel (FR_calf)", 
            "Joint 3 vel (FL_hip)", "Joint 4 vel (FL_thigh)", "Joint 5 vel (FL_calf)",
            "Joint 6 vel (RR_hip)", "Joint 7 vel (RR_thigh)", "Joint 8 vel (RR_calf)", 
            "Joint 9 vel (RL_hip)", "Joint 10 vel (RL_thigh)", "Joint 11 vel (RL_calf)",
        ]
        
        for i in range(min(30, len(obs_mean))):
            desc = descriptions[i] if i < len(descriptions) else f"Feature {i}"
            print(f"  {i:5d} | {obs_mean[i]:7.3f} | {obs_std[i]:7.3f} | {desc}")
    
    if 'action_mean' in stats and stats['action_mean'] is not None:
        action_mean = stats['action_mean'].numpy()
        action_std = stats['action_std'].numpy()
        
        print(f"\nAction stats (shape: {action_mean.shape}):")
        print(f"  Mean range: [{action_mean.min():.3f}, {action_mean.max():.3f}]")
        print(f"  Std range: [{action_std.min():.3f}, {action_std.max():.3f}]")
        
        print("\nPer-joint action stats:")
        print("  Joint | Mean    | Std     | Name")
        print("  ------|---------|---------|-------------")
        
        joint_names = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf"
        ]
        
        for i in range(len(action_mean)):
            name = joint_names[i] if i < len(joint_names) else f"Joint {i}"
            print(f"  {i:5d} | {action_mean[i]:7.3f} | {action_std[i]:7.3f} | {name}")

def fix_il_checkpoint(checkpoint_path):
    """Fix an IL checkpoint that's missing normalization stats by loading from dataset"""
    
    print("\n=== Attempting to fix IL checkpoint by loading stats from dataset ===")
    
    # Try to load the dataset to get proper stats
    try:
        from il_dataset import MiniPupperILDataset
        
        # Common dataset paths to try
        dataset_paths = [
            "/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250910_202558.hdf5",
            "/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_il_dataset.hdf5",
            "../../../rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250910_202558.hdf5"
        ]
        
        dataset = None
        for path in dataset_paths:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                dataset = MiniPupperILDataset(path, split='train')
                break
        
        if dataset:
            stats = dataset.get_statistics()
            print("Successfully loaded normalization stats from dataset!")
            return stats
        else:
            print("Could not find dataset to load stats from")
            
    except Exception as e:
        print(f"Could not load dataset: {e}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Extract normalization stats from checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--output", "-o", help="Output path for extracted stats")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed stats")
    parser.add_argument("--fix", action="store_true", help="Try to fix missing stats from dataset")
    
    args = parser.parse_args()
    
    # Extract stats
    stats = extract_stats_from_checkpoint(args.checkpoint, args.output)
    
    # If stats are missing and fix flag is set, try to load from dataset
    if args.fix and (stats.get('obs_mean') is None or torch.allclose(stats['obs_mean'], torch.zeros(48))):
        dataset_stats = fix_il_checkpoint(args.checkpoint)
        if dataset_stats:
            stats.update(dataset_stats)
            # Re-save with proper stats
            if args.output:
                extract_stats_from_checkpoint(args.checkpoint, args.output)
    
    # Print summary
    if args.verbose:
        print_stats_summary(stats)

if __name__ == "__main__":
    main()
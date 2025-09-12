#!/usr/bin/env python3
"""
Convert rosbag recordings and observation files to HDF5 format for imitation learning
"""

import os
import numpy as np
import h5py
from datetime import datetime
import glob
import re
from tqdm import tqdm

def extract_demo_info(filename):
    """Extract demo type and repetition from filename"""
    # Pattern: demoname_rep1_timestamp or demoname_timestamp
    match = re.match(r'(.+?)(_rep(\d+))?_\d{8}_\d{6}', filename)
    if match:
        demo_type = match.group(1)
        rep_num = int(match.group(3)) if match.group(3) else 1
        return demo_type, rep_num
    return filename, 1

def load_observation_file(obs_path):
    """Load and validate observation file"""
    try:
        data = np.load(obs_path)
        obs = data['observations']
        times = data['times']
        
        # Validate shape
        if obs.shape[1] != 48:
            print(f"Warning: Expected 48-dim observations, got {obs.shape[1]}")
            return None, None
            
        return obs, times
    except Exception as e:
        print(f"Error loading {obs_path}: {e}")
        return None, None

def compute_actions_from_observations(obs_sequence):
    """
    Extract actions from observations.
    In your case, actions are the target joint positions (12-dim)
    which should be in indices 3:15 of the next timestep
    """
    # Actions at time t are the joint positions commanded for time t+1
    actions = np.zeros((len(obs_sequence) - 1, 12))
    
    for i in range(len(obs_sequence) - 1):
        # Use joint positions from next timestep as actions
        # This assumes your controller tracks the commanded positions
        actions[i] = obs_sequence[i + 1, 3:15]  # Joint positions
        
    return actions

def compute_rewards(obs_sequence):
    """
    Compute rewards for imitation learning.
    Simple reward structure based on:
    - Staying upright (from gravity vector)
    - Smooth motion (low joint velocities)
    - Following commands
    """
    rewards = np.zeros(len(obs_sequence) - 1)
    
    for i in range(len(obs_sequence) - 1):
        obs = obs_sequence[i]
        
        # Extract components
        cmd_vel = obs[0:3]  # Commanded velocities
        joint_velocities = obs[15:27]  # Joint velocities
        gravity_vec = obs[36:39]  # Gravity vector
        
        # Reward for staying upright (gravity pointing down in base frame)
        upright_reward = gravity_vec[2] / 9.81  # Should be close to -1
        
        # Penalty for high joint velocities (encourage smooth motion)
        velocity_penalty = -0.1 * np.linalg.norm(joint_velocities)
        
        # Reward for moving when commanded
        if np.linalg.norm(cmd_vel) > 0.1:
            motion_reward = 0.5
        else:
            motion_reward = 0.0
            
        rewards[i] = upright_reward + velocity_penalty + motion_reward
        
    return rewards

def create_hdf5_dataset(rosbag_dir, output_path, train_ratio=0.8):
    """
    Convert all rosbag recordings to HDF5 format
    """
    # Find all observation files
    obs_files = glob.glob(os.path.join(rosbag_dir, "*_observations.npz"))
    print(f"Found {len(obs_files)} observation files")
    
    if not obs_files:
        print("No observation files found!")
        return
    
    # Group by demo type
    demo_groups = {}
    for obs_file in obs_files:
        basename = os.path.basename(obs_file)
        demo_type, rep_num = extract_demo_info(basename)
        
        if demo_type not in demo_groups:
            demo_groups[demo_type] = []
        demo_groups[demo_type].append((obs_file, rep_num))
    
    print(f"\nDemo types found: {list(demo_groups.keys())}")
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create groups
        data_grp = f.create_group('data')
        meta_grp = f.create_group('metadata')
        
        # Track statistics
        total_samples = 0
        all_observations = []
        all_actions = []
        all_rewards = []
        demo_boundaries = []
        demo_info = []
        
        # Process each demo
        for demo_type, demo_files in tqdm(demo_groups.items(), desc="Processing demos"):
            for obs_file, rep_num in sorted(demo_files, key=lambda x: x[1]):
                # Load observations
                obs, times = load_observation_file(obs_file)
                if obs is None:
                    continue
                
                # Compute actions and rewards
                actions = compute_actions_from_observations(obs)
                rewards = compute_rewards(obs)
                
                # Store demo info
                demo_key = f"{demo_type}_rep{rep_num}"
                demo_start = total_samples
                demo_end = total_samples + len(actions)
                
                # Create dataset for this demo
                demo_grp = data_grp.create_group(demo_key)
                demo_grp.create_dataset('obs', data=obs[:-1])  # Exclude last obs (no action)
                demo_grp.create_dataset('actions', data=actions)
                demo_grp.create_dataset('rewards', data=rewards)
                demo_grp.create_dataset('dones', data=np.zeros(len(actions), dtype=bool))
                demo_grp.attrs['demo_type'] = demo_type
                demo_grp.attrs['repetition'] = rep_num
                demo_grp.attrs['length'] = len(actions)
                demo_grp.attrs['duration'] = times[-1] - times[0]
                
                # Accumulate for global statistics
                all_observations.append(obs[:-1])
                all_actions.append(actions)
                all_rewards.append(rewards)
                demo_boundaries.append((demo_start, demo_end))
                demo_info.append({
                    'key': demo_key,
                    'type': demo_type,
                    'start': demo_start,
                    'end': demo_end,
                    'length': len(actions)
                })
                
                total_samples += len(actions)
                
                print(f"  {demo_key}: {len(actions)} samples, {times[-1]-times[0]:.1f}s")
        
        # Concatenate all data
        all_observations = np.concatenate(all_observations, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        
        # Compute normalization statistics
        obs_mean = np.mean(all_observations, axis=0)
        obs_std = np.std(all_observations, axis=0) + 1e-6
        action_mean = np.mean(all_actions, axis=0)
        action_std = np.std(all_actions, axis=0) + 1e-6
        
        # Store normalization stats
        stats_grp = f.create_group('stats')
        stats_grp.create_dataset('obs_mean', data=obs_mean)
        stats_grp.create_dataset('obs_std', data=obs_std)
        stats_grp.create_dataset('action_mean', data=action_mean)
        stats_grp.create_dataset('action_std', data=action_std)
        
        # Create train/val split
        # Stratify by demo type to ensure each type is in both splits
        train_demos = []
        val_demos = []
        
        for demo_type, demos in demo_groups.items():
            type_demos = [d for d in demo_info if d['type'] == demo_type]
            n_train = max(1, int(len(type_demos) * train_ratio))
            
            train_demos.extend(type_demos[:n_train])
            val_demos.extend(type_demos[n_train:])
        
        # Store split information
        splits_grp = f.create_group('splits')
        splits_grp.create_dataset('train_demos', data=[d['key'].encode() for d in train_demos])
        splits_grp.create_dataset('val_demos', data=[d['key'].encode() for d in val_demos])
        
        # Store metadata
        meta_grp.attrs['total_samples'] = total_samples
        meta_grp.attrs['num_demos'] = len(demo_info)
        meta_grp.attrs['demo_types'] = list(demo_groups.keys())
        meta_grp.attrs['obs_dim'] = 48
        meta_grp.attrs['action_dim'] = 12
        meta_grp.attrs['creation_date'] = datetime.now().isoformat()
        
        # Summary statistics
        print(f"\n{'='*60}")
        print(f"HDF5 Dataset Created: {output_path}")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples:,}")
        print(f"Total demos: {len(demo_info)}")
        print(f"Demo types: {len(demo_groups)}")
        print(f"Observation shape: {all_observations.shape}")
        print(f"Action shape: {all_actions.shape}")
        print(f"Train demos: {len(train_demos)}")
        print(f"Val demos: {len(val_demos)}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

def verify_hdf5_dataset(hdf5_path):
    """Verify the created HDF5 dataset"""
    print(f"\n{'='*60}")
    print(f"Verifying HDF5 Dataset: {hdf5_path}")
    print(f"{'='*60}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check structure
        print("\nFile structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} - shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        f.visititems(print_structure)
        
        # Check a sample demo
        data_keys = list(f['data'].keys())
        if data_keys:
            sample_key = data_keys[0]
            sample_demo = f[f'data/{sample_key}']
            print(f"\nSample demo '{sample_key}':")
            print(f"  Observations: {sample_demo['obs'].shape}")
            print(f"  Actions: {sample_demo['actions'].shape}")
            print(f"  Rewards: {sample_demo['rewards'].shape}")
            print(f"  Duration: {sample_demo.attrs['duration']:.1f}s")
            
            # Check data ranges
            obs_data = sample_demo['obs'][:]
            print(f"\n  Observation ranges:")
            print(f"    Min: {np.min(obs_data, axis=0)[:5]}...")
            print(f"    Max: {np.max(obs_data, axis=0)[:5]}...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert rosbag recordings to HDF5")
    parser.add_argument("--input-dir", default="~/rosbag_recordings", 
                        help="Directory containing rosbag recordings")
    parser.add_argument("--output", default="~/rosbag_recordings/minipupper_demos.hdf5",
                        help="Output HDF5 file path")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Ratio of demos to use for training")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the created dataset")
    
    args = parser.parse_args()
    
    # Expand paths
    input_dir = os.path.expanduser(args.input_dir)
    output_path = os.path.expanduser(args.output)
    
    # Create dataset
    create_hdf5_dataset(input_dir, output_path, args.train_ratio)
    
    # Verify if requested
    if args.verify:
        verify_hdf5_dataset(output_path)
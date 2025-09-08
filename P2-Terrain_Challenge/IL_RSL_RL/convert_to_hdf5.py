#!/usr/bin/env python3
"""
Convert recorded rosbag data and observations into HDF5 format for imitation learning.
Following common IL dataset conventions (e.g., robomimic, Isaac Lab).
"""

import os
import h5py
import numpy as np
from datetime import datetime
import subprocess
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm

class RosbagToHDF5Converter:
    def __init__(self, recordings_dir="~/rosbag_recordings"):
        self.recordings_dir = Path(recordings_dir).expanduser()
        self.output_dir = self.recordings_dir / "hdf5_datasets"
        self.output_dir.mkdir(exist_ok=True)
        
    def find_demo_pairs(self):
        """Find matching rosbag and observation files"""
        demo_pairs = []
        
        # Find all observation files
        obs_files = list(self.recordings_dir.glob("*_observations.npz"))
        
        for obs_file in obs_files:
            # Extract demo name from observation file
            demo_name = "_".join(obs_file.stem.split("_")[:-2])  # Remove timestamp and 'observations'
            
            # Find matching rosbag directory
            rosbag_dirs = list(self.recordings_dir.glob(f"{demo_name}_*"))
            rosbag_dirs = [d for d in rosbag_dirs if d.is_dir() and "observations" not in d.name]
            
            if rosbag_dirs:
                demo_pairs.append({
                    "name": demo_name,
                    "obs_file": obs_file,
                    "rosbag_dir": rosbag_dirs[0]  # Take most recent if multiple
                })
                
        return demo_pairs
    
    def extract_actions_from_rosbag(self, rosbag_dir):
        """Extract joint trajectory commands from rosbag"""
        # Convert rosbag to CSV for easier parsing
        csv_dir = rosbag_dir / "csv_output"
        csv_dir.mkdir(exist_ok=True)
        
        # Export joint trajectory topic to CSV
        cmd = [
            "ros2", "bag", "play", str(rosbag_dir),
            "--topics", "/joint_group_effort_controller/joint_trajectory",
            "--storage-config-file", "sqlite3.yaml"
        ]
        
        # Note: In practice, you might want to use rosbag2_py API instead
        # This is a simplified approach - you may need to implement proper rosbag reading
        
        # For now, return placeholder actions matching observation length
        # In production, extract actual commanded joint positions from trajectory messages
        return None
    
    def create_hdf5_dataset(self, demo_pairs, dataset_name="mini_pupper_demos"):
        """Create HDF5 dataset following IL conventions"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{dataset_name}_{timestamp}.hdf5"
        
        with h5py.File(output_file, 'w') as f:
            # Create metadata
            f.attrs['total_demos'] = len(demo_pairs)
            f.attrs['obs_dim'] = 48
            f.attrs['action_dim'] = 12
            f.attrs['created'] = timestamp
            f.attrs['robot'] = 'mini_pupper'
            
            # Create data group
            data_grp = f.create_group('data')
            
            total_transitions = 0
            
            for idx, demo_info in enumerate(tqdm(demo_pairs, desc="Converting demos")):
                demo_name = demo_info['name']
                
                # Load observations
                obs_data = np.load(demo_info['obs_file'])
                observations = obs_data['observations']
                timestamps = obs_data['times']
                
                # Create demo group
                demo_grp = data_grp.create_group(f'demo_{idx}')
                
                # Store observations
                obs_dataset = demo_grp.create_dataset(
                    'obs', 
                    data=observations,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Store actions (joint positions from observations for now)
                # In practice, extract from joint trajectory commands
                actions = observations[:, 3:15]  # Joint positions as actions
                action_dataset = demo_grp.create_dataset(
                    'actions',
                    data=actions,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Store rewards (optional - set to 0 for pure IL)
                rewards = np.zeros((len(observations), 1))
                reward_dataset = demo_grp.create_dataset(
                    'rewards',
                    data=rewards,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Store done flags
                dones = np.zeros(len(observations), dtype=bool)
                dones[-1] = True  # Mark last timestep as done
                done_dataset = demo_grp.create_dataset(
                    'dones',
                    data=dones,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Store timestamps
                time_dataset = demo_grp.create_dataset(
                    'timestamps',
                    data=timestamps,
                    compression='gzip',
                    compression_opts=4
                )
                
                # Demo metadata
                demo_grp.attrs['num_samples'] = len(observations)
                demo_grp.attrs['duration'] = timestamps[-1] - timestamps[0]
                demo_grp.attrs['demo_type'] = demo_name
                demo_grp.attrs['success'] = True  # Assume all demos successful
                
                total_transitions += len(observations)
            
            # Global statistics
            f.attrs['total_transitions'] = total_transitions
            
            # Create mask for train/val split (90/10 split)
            num_demos = len(demo_pairs)
            mask = np.ones(num_demos, dtype=bool)
            val_indices = np.random.choice(num_demos, size=max(1, num_demos//10), replace=False)
            mask[val_indices] = False
            
            f.create_dataset('train_mask', data=mask)
            f.attrs['num_train'] = np.sum(mask)
            f.attrs['num_val'] = np.sum(~mask)
            
        print(f"\nDataset created: {output_file}")
        print(f"Total demos: {len(demo_pairs)}")
        print(f"Total transitions: {total_transitions}")
        print(f"Train demos: {np.sum(mask)}, Val demos: {np.sum(~mask)}")
        
        return output_file
    
    def create_normalized_dataset(self, hdf5_file):
        """Create normalized version of dataset with statistics"""
        with h5py.File(hdf5_file, 'r') as f_in:
            # Compute statistics across all training demos
            train_mask = f_in['train_mask'][:]
            
            all_obs = []
            all_actions = []
            
            for idx in range(len(train_mask)):
                if train_mask[idx]:  # Only use training data for stats
                    demo = f_in[f'data/demo_{idx}']
                    all_obs.append(demo['obs'][:])
                    all_actions.append(demo['actions'][:])
            
            all_obs = np.concatenate(all_obs, axis=0)
            all_actions = np.concatenate(all_actions, axis=0)
            
            # Compute statistics
            obs_mean = np.mean(all_obs, axis=0)
            obs_std = np.std(all_obs, axis=0) + 1e-6
            action_mean = np.mean(all_actions, axis=0)
            action_std = np.std(all_actions, axis=0) + 1e-6
            
            # Create normalized dataset
            norm_file = hdf5_file.parent / f"{hdf5_file.stem}_normalized.hdf5"
            
            with h5py.File(norm_file, 'w') as f_out:
                # Copy attributes
                for key, val in f_in.attrs.items():
                    f_out.attrs[key] = val
                
                # Store normalization stats
                stats_grp = f_out.create_group('stats')
                stats_grp.create_dataset('obs_mean', data=obs_mean)
                stats_grp.create_dataset('obs_std', data=obs_std)
                stats_grp.create_dataset('action_mean', data=action_mean)
                stats_grp.create_dataset('action_std', data=action_std)
                
                # Copy and normalize data
                f_out.create_dataset('train_mask', data=f_in['train_mask'][:])
                data_grp = f_out.create_group('data')
                
                for idx in tqdm(range(len(train_mask)), desc="Normalizing"):
                    demo_in = f_in[f'data/demo_{idx}']
                    demo_out = data_grp.create_group(f'demo_{idx}')
                    
                    # Normalize observations and actions
                    obs_norm = (demo_in['obs'][:] - obs_mean) / obs_std
                    action_norm = (demo_in['actions'][:] - action_mean) / action_std
                    
                    demo_out.create_dataset('obs', data=obs_norm, compression='gzip')
                    demo_out.create_dataset('actions', data=action_norm, compression='gzip')
                    demo_out.create_dataset('rewards', data=demo_in['rewards'][:], compression='gzip')
                    demo_out.create_dataset('dones', data=demo_in['dones'][:], compression='gzip')
                    demo_out.create_dataset('timestamps', data=demo_in['timestamps'][:], compression='gzip')
                    
                    # Copy attributes
                    for key, val in demo_in.attrs.items():
                        demo_out.attrs[key] = val
                        
        print(f"\nNormalized dataset created: {norm_file}")
        return norm_file

def create_dataset_info(hdf5_file):
    """Create a YAML info file for the dataset"""
    info_file = hdf5_file.parent / f"{hdf5_file.stem}_info.yaml"
    
    with h5py.File(hdf5_file, 'r') as f:
        info = {
            'dataset_name': hdf5_file.stem,
            'robot': 'mini_pupper',
            'total_demos': int(f.attrs['total_demos']),
            'total_transitions': int(f.attrs['total_transitions']),
            'num_train': int(f.attrs['num_train']),
            'num_val': int(f.attrs['num_val']),
            'obs_dim': int(f.attrs['obs_dim']),
            'action_dim': int(f.attrs['action_dim']),
            'created': str(f.attrs['created']),
            
            'observation_space': {
                'cmd_vel': [0, 3],
                'joint_positions': [3, 15],
                'joint_velocities': [15, 27],
                'previous_actions': [27, 39],
                'gravity_vector': [39, 42],
                'angular_velocity': [42, 45],
                'gait_phase': [45, 47],
                'foot_contact': [47, 48]
            },
            
            'action_space': {
                'joint_positions': [0, 12]
            },
            
            'demos': []
        }
        
        # Add demo information
        for idx in range(f.attrs['total_demos']):
            demo = f[f'data/demo_{idx}']
            info['demos'].append({
                'index': idx,
                'type': str(demo.attrs['demo_type']),
                'num_samples': int(demo.attrs['num_samples']),
                'duration': float(demo.attrs['duration']),
                'success': bool(demo.attrs['success'])
            })
    
    with open(info_file, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)
    
    print(f"Dataset info saved to: {info_file}")
    return info_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rosbag recordings to HDF5 for IL")
    parser.add_argument("--recordings-dir", default="~/rosbag_recordings", 
                       help="Directory containing rosbag recordings")
    parser.add_argument("--dataset-name", default="mini_pupper_demos",
                       help="Name for the output dataset")
    parser.add_argument("--normalize", action="store_true",
                       help="Also create normalized version of dataset")
    
    args = parser.parse_args()
    
    # Create converter
    converter = RosbagToHDF5Converter(args.recordings_dir)
    
    # Find demo pairs
    demo_pairs = converter.find_demo_pairs()
    
    if not demo_pairs:
        print("No matching rosbag and observation files found!")
        exit(1)
    
    print(f"Found {len(demo_pairs)} demo pairs:")
    for demo in demo_pairs:
        print(f"  - {demo['name']}")
    
    # Convert to HDF5
    hdf5_file = converter.create_hdf5_dataset(demo_pairs, args.dataset_name)
    
    # Create normalized version if requested
    if args.normalize:
        norm_file = converter.create_normalized_dataset(hdf5_file)
        create_dataset_info(norm_file)
    else:
        create_dataset_info(hdf5_file)
    
    print("\nConversion complete!")
    print("\nTo load the dataset in Python:")
    print("```python")
    print("import h5py")
    print(f"f = h5py.File('{hdf5_file.name}', 'r')")
    print("demo_0 = f['data/demo_0']")
    print("obs = demo_0['obs'][:]")
    print("actions = demo_0['actions'][:]")
    print("```")
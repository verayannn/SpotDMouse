# Save as fix_hdf5_for_robomimic_v2.py
import h5py
import json
import shutil
import numpy as np
from pathlib import Path

def fix_hdf5_for_robomimic(input_path, output_path=None):
    """Fix HDF5 file to be compatible with robomimic"""
    
    # Create backup if modifying in place
    if output_path is None:
        backup_path = input_path.replace('.hdf5', '_backup.hdf5')
        shutil.copy2(input_path, backup_path)
        print(f"Created backup: {backup_path}")
        output_path = input_path
    
    # Create a new file with proper structure
    temp_path = output_path + '.tmp'
    
    with h5py.File(input_path, 'r') as f_in:
        with h5py.File(temp_path, 'w') as f_out:
            # Create data group
            data_grp = f_out.create_group('data')
            
            # Add required env_args metadata with ALL required fields
            env_args = {
                "env_name": "Isaac-Velocity-Flat-Custom-Quad-v0",
                "env_type": "isaaclab",  # This is what robomimic uses to identify env type
                "type": "isaaclab",       # Add this field that robomimic is looking for
                "env_kwargs": {}
            }
            data_grp.attrs['env_args'] = json.dumps(env_args)
            
            # Get all demo keys
            demo_keys = [k for k in f_in['data'].keys() if k.startswith('demo_')]
            data_grp.attrs['total'] = len(demo_keys)
            
            print(f"Processing {len(demo_keys)} demos...")
            
            # Process each demo
            for demo_key in demo_keys:
                demo_in = f_in[f'data/{demo_key}']
                demo_out = data_grp.create_group(demo_key)
                
                # Copy demo attributes
                for attr_name, attr_value in demo_in.attrs.items():
                    demo_out.attrs[attr_name] = attr_value
                
                # Copy actions, rewards, dones, timestamps
                for dataset_name in ['actions', 'rewards', 'dones', 'timestamps']:
                    if dataset_name in demo_in:
                        data = demo_in[dataset_name][:]
                        demo_out.create_dataset(dataset_name, data=data)
                
                # Handle observations - convert from dataset to group structure
                if 'obs' in demo_in:
                    obs_data = demo_in['obs'][:]
                    obs_grp = demo_out.create_group('obs')
                    # Store as 'policy' key for robomimic
                    obs_grp.create_dataset('policy', data=obs_data)
                
                # Also create next_obs (robomimic might expect this)
                if 'obs' in demo_in:
                    obs_data = demo_in['obs'][:]
                    next_obs_grp = demo_out.create_group('next_obs')
                    # For next_obs, shift by 1 and repeat last observation
                    next_obs_data = np.zeros_like(obs_data)
                    next_obs_data[:-1] = obs_data[1:]
                    next_obs_data[-1] = obs_data[-1]  # Repeat last obs
                    next_obs_grp.create_dataset('policy', data=next_obs_data)
                
                print(f"  Processed {demo_key}: {len(demo_in['actions'])} samples")
            
            # Copy train_mask if it exists
            if 'train_mask' in f_in:
                f_out.create_dataset('train_mask', data=f_in['train_mask'][:])
                
            # Add mask group for train/validation split
            mask_grp = f_out.create_group('mask')
            
            # Use existing train_mask or create default (90% train, 10% validation)
            if 'train_mask' in f_in:
                train_indices = np.where(f_in['train_mask'][:])[0]
                valid_indices = np.where(~f_in['train_mask'][:])[0]
            else:
                # Default 90/10 split
                num_demos = len(demo_keys)
                indices = np.arange(num_demos)
                np.random.seed(42)
                np.random.shuffle(indices)
                split_idx = int(0.9 * num_demos)
                train_indices = indices[:split_idx]
                valid_indices = indices[split_idx:]
            
            mask_grp.create_dataset('train', data=[f"demo_{i}" for i in train_indices])
            mask_grp.create_dataset('valid', data=[f"demo_{i}" for i in valid_indices])
    
    # Replace original file with fixed version
    shutil.move(temp_path, output_path)
    print(f"\nFixed HDF5 file saved to: {output_path}")
    
    # Verify the fix
    print("\n=== Verifying fixed file ===")
    with h5py.File(output_path, 'r') as f:
        print(f"data attributes: {list(f['data'].attrs.keys())}")
        env_meta = json.loads(f['data'].attrs['env_args'])
        print(f"env_args: {env_meta}")
        print(f"env type field: {env_meta.get('type', 'NOT FOUND')}")
        
        # Check first demo
        first_demo = list(f['data'].keys())[0]
        print(f"\nFirst demo '{first_demo}' structure:")
        print(f"  Keys: {list(f[f'data/{first_demo}'].keys())}")
        if 'obs' in f[f'data/{first_demo}']:
            print(f"  obs type: {type(f[f'data/{first_demo}/obs'])}")
            print(f"  obs keys: {list(f[f'data/{first_demo}/obs'].keys())}")
            print(f"  obs/policy shape: {f[f'data/{first_demo}/obs/policy'].shape}")

if __name__ == "__main__":
    input_file = "/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250910_202558.hdf5"
    output_file = "/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_robomimic_v2.hdf5"
    
    fix_hdf5_for_robomimic(input_file, output_file)
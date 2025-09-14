import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

hdf5_path = "/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250910_202558.hdf5"

# Create a directory for saving plots
plot_dir = "/workspace/SpotDMouse/plots"
os.makedirs(plot_dir, exist_ok=True)

print("=== HDF5 File Structure ===")
with h5py.File(hdf5_path, 'r') as f:
    def print_structure(name, obj):
        print(f"{name}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            if hasattr(obj, 'attrs'):
                print(f"  Attributes: {list(obj.attrs.keys())}")
    
    f.visititems(print_structure)
    
    # Check specific attributes
    if 'data' in f:
        print("\n=== 'data' group attributes ===")
        print(f"Attributes: {list(f['data'].attrs.keys())}")
        
        # Plot and save data from demos 0, 1, 2, 3
        for demo_idx in range(4):
            demo_key = f"demo_{demo_idx}"
            if demo_key in f['data']:
                print(f"\n=== Processing demo '{demo_key}' ===")
                
                # Get observation data
                if 'obs' in f[f'data/{demo_key}']:
                    obs_path = f[f'data/{demo_key}/obs']
                    if isinstance(obs_path, h5py.Dataset):
                        # Handle 'obs' as a dataset
                        print(f"Observation is a Dataset with shape: {obs_path.shape}, dtype: {obs_path.dtype}")
                        # Directly process the dataset as needed
                        continue
                    else:
                        # Handle 'obs' as a group
                        obs_group = obs_path
                        obs_keys = list(obs_group.keys())
                        print(f"Observation keys: {obs_keys}")
                    
                    # Plot joint positions if available
                    if 'joint_positions' in obs_keys:
                        joint_positions = obs_group['joint_positions'][:]
                        
                        plt.figure(figsize=(12, 6))
                        for i in range(joint_positions.shape[1]):
                            plt.plot(joint_positions[:, i], label=f'Joint {i}')
                        
                        plt.title(f'Joint Positions - Demo {demo_idx}')
                        plt.xlabel('Time step')
                        plt.ylabel('Position')
                        plt.legend()
                        plt.tight_layout()
                        
                        # Save the plot
                        plt.savefig(f"{plot_dir}/demo_{demo_idx}_joint_positions.png")
                        plt.close()
                        print(f"Saved joint positions plot for demo {demo_idx}")
                    
                    # Plot other numerical data (first few found)
                    plotted = 0
                    for key in obs_keys:
                        if plotted >= 2:  # Limit to 2 additional plots per demo
                            break
                            
                        data = obs_group[key][:]
                        if isinstance(data, np.ndarray) and data.size > 0 and np.issubdtype(data.dtype, np.number):
                            plt.figure(figsize=(12, 6))
                            
                            if data.ndim == 1:
                                plt.plot(data)
                                plt.title(f'{key} - Demo {demo_idx}')
                            elif data.ndim == 2 and data.shape[1] <= 10:
                                for i in range(data.shape[1]):
                                    plt.plot(data[:, i], label=f'Dim {i}')
                                plt.legend()
                                plt.title(f'{key} - Demo {demo_idx}')
                            
                            plt.xlabel('Time step')
                            plt.ylabel('Value')
                            plt.tight_layout()
                            
                            # Save the plot
                            plt.savefig(f"{plot_dir}/demo_{demo_idx}_{key}.png")
                            plt.close()
                            print(f"Saved {key} plot for demo {demo_idx}")
                            plotted += 1

print(f"\nPlots saved to {plot_dir}")
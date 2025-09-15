import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import os

hdf5_path = "/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5"

# Check if file exists
if not os.path.exists(hdf5_path):
    print(f"ERROR: HDF5 file not found at {hdf5_path}")
    exit(1)
else:
    print(f"Found HDF5 file at {hdf5_path}")

# Create a directory for saving plots
plot_dir = "/workspace/SpotDMouse/plots"
os.makedirs(plot_dir, exist_ok=True)
print(f"Plot directory: {plot_dir}")

plot_count = 0  # Track how many plots we save

print("=== HDF5 File Structure ===")
with h5py.File(hdf5_path, 'r') as f:
    # Print top-level keys
    print(f"Top level keys: {list(f.keys())}")
    
    def print_structure(name, obj):
        print(f"{name}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
            if len(obj.shape) > 0 and obj.shape[0] == 0:
                print(f"  WARNING: Dataset {name} has zero length")
        elif isinstance(obj, h5py.Group):
            if hasattr(obj, 'attrs'):
                print(f"  Attributes: {list(obj.attrs.keys())}")
    
    f.visititems(print_structure)
    
    # Check specific attributes
    if 'data' in f:
        print("\n=== 'data' group attributes ===")
        print(f"Attributes: {list(f['data'].attrs.keys())}")
        print(f"Children of data: {list(f['data'].keys())}")
        
        # Plot and save action data from demos 1, 2, 3, 4
        for demo_idx in range(1, 5):  # Changed from range(4) to range(1, 5)
            demo_key = f"demo_{demo_idx}"
            if demo_key in f['data']:
                print(f"\n=== Processing demo '{demo_key}' ===")
                
                # Try to get demo name from attributes
                demo_name = demo_key  # Default name
                demo_group = f[f'data/{demo_key}']
                
                # Check for name in various possible locations
                if 'name' in demo_group.attrs:
                    demo_name = demo_group.attrs['name']
                elif 'demo_name' in demo_group.attrs:
                    demo_name = demo_group.attrs['demo_name']
                elif 'description' in demo_group.attrs:
                    demo_name = demo_group.attrs['description']
                
                # Print all attributes for this demo
                if hasattr(demo_group, 'attrs') and len(demo_group.attrs) > 0:
                    print(f"Demo attributes: {dict(demo_group.attrs)}")
                
                # Get action data
                if 'actions' in f[f'data/{demo_key}']:
                    actions = f[f'data/{demo_key}/actions'][:]
                    print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
                    
                    if actions.size == 0:
                        print("WARNING: Actions data is empty")
                        continue
                    
                    # Create subplot figure for all action dimensions
                    n_actions = actions.shape[1] if actions.ndim > 1 else 1
                    fig, axes = plt.subplots(n_actions, 1, figsize=(12, 2*n_actions), sharex=True)
                    if n_actions == 1:
                        axes = [axes]
                    
                    # Plot each action dimension
                    for i in range(n_actions):
                        if actions.ndim > 1:
                            axes[i].plot(actions[:, i], linewidth=1.5)
                            axes[i].set_ylabel(f'Action {i}')
                            axes[i].grid(True, alpha=0.3)
                        else:
                            axes[i].plot(actions, linewidth=1.5)
                            axes[i].set_ylabel('Action')
                            axes[i].grid(True, alpha=0.3)
                    
                    axes[-1].set_xlabel('Time step')
                    # Use the demo name in the title
                    fig.suptitle(f'Actions - {demo_name} (Demo {demo_idx})', fontsize=14)
                    plt.tight_layout()
                    
                    # Save the plot with demo name in filename
                    safe_demo_name = demo_name.replace(' ', '_').replace('/', '_')
                    plot_path = f"{plot_dir}/{safe_demo_name}_actions.png"
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    if os.path.exists(plot_path):
                        print(f"Successfully saved actions plot to {plot_path}")
                        plot_count += 1
                    else:
                        print(f"ERROR: Failed to save plot to {plot_path}")
                else:
                    print(f"WARNING: No 'actions' data found in {demo_key}")
            else:
                print(f"Demo {demo_idx} not found in data group")
    else:
        print("ERROR: 'data' group not found in HDF5 file")

print(f"\nPlots saved: {plot_count} to {plot_dir}")
print("To view a plot, run: \"$BROWSER /workspace/SpotDMouse/plots/FILENAME.png\"")
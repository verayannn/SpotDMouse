import h5py
import json

hdf5_path = "/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250910_202558.hdf5"

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
        
        # Check first demo
        demo_keys = list(f['data'].keys())
        if demo_keys:
            first_demo = demo_keys[0]
            print(f"\n=== First demo '{first_demo}' structure ===")
            if 'obs' in f[f'data/{first_demo}']:
                print("obs keys:", list(f[f'data/{first_demo}/obs'].keys()))
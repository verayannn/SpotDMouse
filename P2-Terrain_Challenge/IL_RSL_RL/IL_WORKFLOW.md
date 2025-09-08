# Mini Pupper Imitation Learning Workflow

This guide explains how to record teleoperated demonstrations and use them for imitation learning (IL) training.

## Overview

The workflow consists of:
1. Recording teleoperated demonstrations with synchronized observations
2. Converting recordings to HDF5 format for IL training
3. Loading data for training with PyTorch

## Step 1: Record Demonstrations

### Prerequisites
- Mini Pupper robot running with ROS2
- All required topics publishing (joint_states, imu, cmd_vel, etc.)

### Recording Process

```bash
# Start your Mini Pupper controller first
ros2 run mini_pupper_control mlpcontrolnode

# In another terminal, run the recording script
python3 teleop_record.py
```

The script will:
- Check if all required topics are available
- Show publishing rates for each topic
- Record both rosbag data AND 48-dim observation vectors
- Save compressed rosbags and observation NPZ files

### What Gets Recorded

For each demonstration:
- **Rosbag**: All raw sensor data (joint states, IMU, commands, etc.)
- **Observations**: 48-dimensional vectors matching your MLP format:
  - Command velocities (3)
  - Joint positions (12)
  - Joint velocities (12)
  - Previous actions (12)
  - Gravity vector (3)
  - Angular velocity (3)
  - Gait phase (2)
  - Foot contact (1)

## Step 2: Convert to HDF5

After recording, convert your data to HDF5 format:

```bash
# Basic conversion
python3 convert_to_hdf5.py

# With normalization (recommended)
python3 convert_to_hdf5.py --normalize

# Specify custom directory
python3 convert_to_hdf5.py --recordings-dir ~/my_recordings --dataset-name my_demos
```

This creates:
- `mini_pupper_demos_TIMESTAMP.hdf5` - Raw dataset
- `mini_pupper_demos_TIMESTAMP_normalized.hdf5` - Normalized dataset
- `mini_pupper_demos_TIMESTAMP_info.yaml` - Dataset metadata

### HDF5 Structure

```
dataset.hdf5
├── attrs/
│   ├── total_demos: 8
│   ├── obs_dim: 48
│   ├── action_dim: 12
│   └── ...
├── data/
│   ├── demo_0/
│   │   ├── obs: [N, 48]
│   │   ├── actions: [N, 12]
│   │   ├── rewards: [N, 1]
│   │   ├── dones: [N]
│   │   └── timestamps: [N]
│   └── demo_1/...
├── train_mask: [num_demos] (90/10 train/val split)
└── stats/ (if normalized)
    ├── obs_mean: [48]
    ├── obs_std: [48]
    ├── action_mean: [12]
    └── action_std: [12]
```

## Step 3: Load Data for Training

### Basic Usage

```python
from il_dataset import MiniPupperILDataset, create_il_dataloaders

# Load dataset
dataset = MiniPupperILDataset(
    "~/rosbag_recordings/hdf5_datasets/mini_pupper_demos_normalized.hdf5",
    split="train",
    normalize=True
)

# Create dataloaders
train_loader, val_loader = create_il_dataloaders(
    hdf5_path,
    batch_size=32,
    seq_len=None,  # Single timestep
    device="cuda"
)

# Train your policy
for batch in train_loader:
    obs = batch['obs']        # [batch_size, 48]
    actions = batch['action']  # [batch_size, 12]
    # ... training code ...
```

### Sequence Learning

For temporal models (RNN, Transformer):

```python
# Load sequences of length 50
train_loader, val_loader = create_il_dataloaders(
    hdf5_path,
    batch_size=16,
    seq_len=50,  # Return sequences
    device="cuda"
)

for batch in train_loader:
    obs = batch['obs']        # [batch_size, 50, 48]
    actions = batch['action']  # [batch_size, 50, 12]
```

### Variable Length Sequences

For curriculum learning:

```python
from il_dataset import SequenceSampler

sampler = SequenceSampler(dataset, min_seq_len=10, max_seq_len=100)

# Sample variable length sequences
for _ in range(num_iterations):
    seq = sampler.sample_sequence()
    obs = seq['obs']        # [seq_len, 48]
    actions = seq['actions']  # [seq_len, 12]
```

## Example Training Script

```python
import torch
import torch.nn as nn
from il_dataset import create_il_dataloaders

# Simple BC policy
class BCPolicy(nn.Module):
    def __init__(self, obs_dim=48, action_dim=12):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs):
        return self.mlp(obs)

# Load data
train_loader, val_loader = create_il_dataloaders(
    "path/to/normalized.hdf5",
    batch_size=256,
    device="cuda"
)

# Initialize policy
policy = BCPolicy().cuda()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    for batch in train_loader:
        obs = batch['obs']
        actions = batch['action']
        
        # Forward pass
        pred_actions = policy(obs)
        loss = loss_fn(pred_actions, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Tips for Success

1. **Recording Quality**:
   - Ensure smooth, consistent demonstrations
   - Keep demos around 60-90 seconds for good coverage
   - Record multiple variations of each behavior

2. **Data Balance**:
   - Use equal duration demos for balanced dataset
   - Include failure recovery demonstrations
   - Add noise/perturbations for robustness

3. **Training Tips**:
   - Start with normalized data
   - Use sequence modeling for dynamic behaviors
   - Monitor validation loss to prevent overfitting
   - Consider data augmentation (add noise to observations)

## Troubleshooting

### Topic not publishing
```bash
# Check topic list
ros2 topic list

# Check publishing rate
ros2 topic hz /joint_states
```

### Observation dimension mismatch
- Verify your controller uses 48-dim observations
- Check joint ordering matches between recording and playback
- Ensure IMU is publishing quaternion orientation

### Poor policy performance
- Record more diverse demonstrations
- Try different sequence lengths
- Increase model capacity
- Add regularization (dropout, weight decay)

## Integration with Isaac Lab

The HDF5 format is compatible with standard IL frameworks. To use with Isaac Lab:

```python
# Adapt the dataset loader
class IsaacLabDataset(MiniPupperILDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        # Remap to Isaac Lab format
        return {
            'observation': data['obs'],
            'action': data['action'],
            # ... other fields
        }
```
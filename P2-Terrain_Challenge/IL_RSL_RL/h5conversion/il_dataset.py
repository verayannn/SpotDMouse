#!/usr/bin/env python3
"""
PyTorch Dataset class for loading Mini Pupper IL demonstrations from HDF5.
Compatible with common IL training frameworks.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import random

class MiniPupperILDataset(Dataset):
    """PyTorch Dataset for Mini Pupper IL demonstrations."""
    
    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        seq_len: Optional[int] = None,
        normalize: bool = True,
        device: str = "cpu"
    ):
        """
        Args:
            hdf5_path: Path to HDF5 dataset file
            split: 'train' or 'val'
            seq_len: If specified, return sequences of this length
            normalize: Whether to use normalized data
            device: Device to load tensors to
        """
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.seq_len = seq_len
        self.normalize = normalize
        self.device = device
        
        # Open HDF5 file and load metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            self.train_mask = f['train_mask'][:]
            self.num_demos = len(self.train_mask)
            
            # Get demo indices for this split
            if split == "train":
                self.demo_indices = np.where(self.train_mask)[0]
            else:  # val
                self.demo_indices = np.where(~self.train_mask)[0]
            
            # Load normalization statistics if available
            if 'stats' in f and normalize:
                self.obs_mean = torch.tensor(f['stats/obs_mean'][:], dtype=torch.float32)
                self.obs_std = torch.tensor(f['stats/obs_std'][:], dtype=torch.float32)
                self.action_mean = torch.tensor(f['stats/action_mean'][:], dtype=torch.float32)
                self.action_std = torch.tensor(f['stats/action_std'][:], dtype=torch.float32)
            else:
                self.obs_mean = None
                self.obs_std = None
                self.action_mean = None
                self.action_std = None
            
            # Build index mapping for efficient sampling
            self._build_index_map(f)
    
    def _build_index_map(self, f):
        """Build mapping from dataset index to (demo_idx, timestep)"""
        self.index_map = []
        
        for demo_idx in self.demo_indices:
            demo = f[f'data/demo_{demo_idx}']
            num_samples = demo.attrs['num_samples']
            
            if self.seq_len is None:
                # Single timestep sampling
                for t in range(num_samples):
                    self.index_map.append((demo_idx, t))
            else:
                # Sequence sampling
                for t in range(num_samples - self.seq_len + 1):
                    self.index_map.append((demo_idx, t))
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample or sequence."""
        demo_idx, start_t = self.index_map[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            demo = f[f'data/demo_{demo_idx}']
            
            if self.seq_len is None:
                # Single timestep
                obs = torch.tensor(demo['obs'][start_t], dtype=torch.float32)
                action = torch.tensor(demo['actions'][start_t], dtype=torch.float32)
                reward = torch.tensor(demo['rewards'][start_t], dtype=torch.float32)
                done = torch.tensor(demo['dones'][start_t], dtype=torch.bool)
                
            else:
                # Sequence
                end_t = start_t + self.seq_len
                obs = torch.tensor(demo['obs'][start_t:end_t], dtype=torch.float32)
                action = torch.tensor(demo['actions'][start_t:end_t], dtype=torch.float32)
                reward = torch.tensor(demo['rewards'][start_t:end_t], dtype=torch.float32)
                done = torch.tensor(demo['dones'][start_t:end_t], dtype=torch.bool)
            
            # Apply normalization if needed
            if self.normalize and self.obs_mean is not None:
                obs = (obs - self.obs_mean) / self.obs_std
                action = (action - self.action_mean) / self.action_std
            
            return {
                'obs': obs.to(self.device),
                'action': action.to(self.device),
                'reward': reward.to(self.device),
                'done': done.to(self.device),
                'demo_idx': demo_idx,
                'timestep': start_t
            }
    
    def get_demo(self, demo_idx: int) -> Dict[str, np.ndarray]:
        """Get full demonstration by index."""
        with h5py.File(self.hdf5_path, 'r') as f:
            demo = f[f'data/demo_{demo_idx}']
            return {
                'obs': demo['obs'][:],
                'actions': demo['actions'][:],
                'rewards': demo['rewards'][:],
                'dones': demo['dones'][:],
                'timestamps': demo['timestamps'][:],
                'demo_type': demo.attrs['demo_type']
            }
    
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get dataset statistics."""
        if self.obs_mean is not None:
            return {
                'obs_mean': self.obs_mean,
                'obs_std': self.obs_std,
                'action_mean': self.action_mean,
                'action_std': self.action_std
            }
        else:
            # Compute statistics on the fly
            all_obs = []
            all_actions = []
            
            with h5py.File(self.hdf5_path, 'r') as f:
                for demo_idx in self.demo_indices:
                    demo = f[f'data/demo_{demo_idx}']
                    all_obs.append(demo['obs'][:])
                    all_actions.append(demo['actions'][:])
            
            all_obs = np.concatenate(all_obs, axis=0)
            all_actions = np.concatenate(all_actions, axis=0)
            
            return {
                'obs_mean': torch.tensor(np.mean(all_obs, axis=0), dtype=torch.float32),
                'obs_std': torch.tensor(np.std(all_obs, axis=0) + 1e-6, dtype=torch.float32),
                'action_mean': torch.tensor(np.mean(all_actions, axis=0), dtype=torch.float32),
                'action_std': torch.tensor(np.std(all_actions, axis=0) + 1e-6, dtype=torch.float32)
            }


class SequenceSampler:
    """Sample sequences of varying lengths for curriculum learning."""
    
    def __init__(self, dataset: MiniPupperILDataset, min_seq_len: int = 10, max_seq_len: int = 100):
        self.dataset = dataset
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        
    def sample_sequence(self) -> Dict[str, torch.Tensor]:
        """Sample a variable-length sequence."""
        # Randomly select sequence length
        seq_len = random.randint(self.min_seq_len, self.max_seq_len)
        
        # Sample from dataset
        with h5py.File(self.dataset.hdf5_path, 'r') as f:
            # Random demo
            demo_idx = random.choice(self.dataset.demo_indices)
            demo = f[f'data/demo_{demo_idx}']
            max_start = demo.attrs['num_samples'] - seq_len
            
            if max_start <= 0:
                # Demo too short, use full demo
                start_t = 0
                seq_len = demo.attrs['num_samples']
            else:
                start_t = random.randint(0, max_start)
            
            end_t = start_t + seq_len
            
            # Extract sequence
            obs = torch.tensor(demo['obs'][start_t:end_t], dtype=torch.float32)
            actions = torch.tensor(demo['actions'][start_t:end_t], dtype=torch.float32)
            
            # Normalize if needed
            if self.dataset.normalize and self.dataset.obs_mean is not None:
                obs = (obs - self.dataset.obs_mean) / self.dataset.obs_std
                actions = (actions - self.dataset.action_mean) / self.dataset.action_std
            
            return {
                'obs': obs,
                'actions': actions,
                'seq_len': seq_len,
                'demo_idx': demo_idx,
                'start_t': start_t
            }


def create_il_dataloaders(
    hdf5_path: str,
    batch_size: int = 32,
    seq_len: Optional[int] = None,
    normalize: bool = True,
    device: str = "cpu",
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Create datasets
    train_dataset = MiniPupperILDataset(
        hdf5_path, split="train", seq_len=seq_len, normalize=normalize, device=device
    )
    val_dataset = MiniPupperILDataset(
        hdf5_path, split="val", seq_len=seq_len, normalize=normalize, device=device
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]
    else:
        hdf5_path = "~/rosbag_recordings/hdf5_datasets/mini_pupper_demos_normalized.hdf5"
    
    # Load dataset
    dataset = MiniPupperILDataset(hdf5_path, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Get single sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Observation shape: {sample['obs'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nObservation mean shape: {stats['obs_mean'].shape}")
    print(f"Action std shape: {stats['action_std'].shape}")
    
    # Create dataloaders
    train_loader, val_loader = create_il_dataloaders(
        hdf5_path,
        batch_size=32,
        seq_len=50  # Use sequences of length 50
    )
    
    # Iterate through one batch
    for batch in train_loader:
        print(f"\nBatch shapes:")
        print(f"  obs: {batch['obs'].shape}")  # [batch_size, seq_len, obs_dim]
        print(f"  action: {batch['action'].shape}")  # [batch_size, seq_len, action_dim]
        break
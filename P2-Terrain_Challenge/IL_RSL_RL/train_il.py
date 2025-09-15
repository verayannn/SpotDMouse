#!/usr/bin/env python3
"""
Train imitation learning policy using recorded demonstrations
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import wandb
from tqdm import tqdm
import argparse

from il_dataset import MiniPupperILDataset

class MLPPolicy(nn.Module):
    """MLP policy matching your deployed controller architecture"""
    def __init__(self, obs_dim=48, action_dim=12, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.actor = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, obs):
        return self.actor(obs)

class ILTrainer:
    def __init__(self, 
                 dataset_path,
                 save_dir="/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models",
                 device="cuda",
                 use_wandb=True):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = os.path.expanduser(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        self.train_dataset = MiniPupperILDataset(dataset_path, split='train')
        self.val_dataset = MiniPupperILDataset(dataset_path, split='val')
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        # Create model
        self.model = MLPPolicy().to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.MSELoss()
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="minipupper-il",
                config={
                    "dataset": dataset_path,
                    "model": "MLP",
                    "hidden_dims": [512, 256, 128],
                    "lr": 3e-4,
                    "batch_size": 256
                }
            )
            
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            obs = batch['obs'].to(self.device)
            actions = batch['action'].to(self.device)  # Changed from 'actions' to 'action'
            
            # Forward pass
            pred_actions = self.model(obs)
            loss = self.criterion(pred_actions, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                obs = batch['obs'].to(self.device)
                actions = batch['action'].to(self.device)  # Changed from 'actions' to 'action'
                
                pred_actions = self.model(obs)
                loss = self.criterion(pred_actions, actions)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, num_epochs=100, batch_size=256, save_every=10):
        """Main training loop"""
        
        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.scheduler.get_last_lr()[0]
            }
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss)
                print(f"New best model saved (val_loss: {val_loss:.6f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, val_loss)
                
        print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
        
    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'obs_mean': self.train_dataset.obs_mean,
            'obs_std': self.train_dataset.obs_std,
            'action_mean': self.train_dataset.action_mean,
            'action_std': self.train_dataset.action_std,
        }
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def export_for_deployment(self, checkpoint_path=None):
        """Export model for deployment on robot"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pt')
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create traced model for faster inference
        example_input = torch.randn(1, 48).to(self.device)
        traced_model = torch.jit.trace(self.model, example_input)
        
        # Save deployment package
        deployment_path = os.path.join(self.save_dir, 'deployment_model.pt')
        torch.save({
            'model': traced_model,
            'obs_mean': checkpoint['obs_mean'],
            'obs_std': checkpoint['obs_std'],
            'action_mean': checkpoint['action_mean'],
            'action_std': checkpoint['action_std'],
        }, deployment_path)
        
        print(f"Exported deployment model to {deployment_path}")
        return deployment_path

def main():
    parser = argparse.ArgumentParser(description="Train IL policy")
    parser.add_argument("--dataset", default="/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250910_202558.hdf5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--device", default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--export", action="store_true",
                        help="Export model for deployment after training")
    
    args = parser.parse_args()
    
    # Expand dataset path
    dataset_path = os.path.expanduser(args.dataset)
    
    # Create trainer
    trainer = ILTrainer(
        dataset_path=dataset_path,
        device=args.device,
        use_wandb=not args.no_wandb
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Export for deployment
    if args.export:
        trainer.export_for_deployment()
        
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
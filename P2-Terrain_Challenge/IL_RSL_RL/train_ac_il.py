#!/usr/bin/env python3
"""
Train imitation learning policy with both actor and critic in RSL RL format
Including action standard deviation (std) for full RSL RL compatibility
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

class ActorCriticMLP(nn.Module):
    """Actor-Critic MLP matching RSL RL's expected structure exactly"""
    def __init__(self, obs_dim=48, action_dim=12, hidden_dims=[512, 256, 128], 
                 init_action_std=1.0):
        super().__init__()
        
        # Actor network
        actor_layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        actor_layers.append(nn.Linear(in_dim, action_dim))
        
        # Store as 'actor' to match RSL RL naming
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
            ])
            in_dim = hidden_dim
            
        critic_layers.append(nn.Linear(in_dim, 1))  # Critic outputs value
        
        # Store as 'critic' to match RSL RL naming
        self.critic = nn.Sequential(*critic_layers)
        
        # Action standard deviation (log scale) - RSL RL compatibility
        # This is typically used for stochastic policies
        self.std = nn.Parameter(torch.ones(action_dim) * np.log(init_action_std))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, obs):
        # Return both actor output and critic value
        return self.actor(obs), self.critic(obs)
    
    def get_action(self, obs, deterministic=True):
        """Get action with optional stochasticity"""
        mean = self.actor(obs)
        if deterministic:
            return mean
        else:
            # Sample from Gaussian distribution
            std = torch.exp(self.std)
            dist = torch.distributions.Normal(mean, std)
            return dist.sample()
    
    def get_value(self, obs):
        # Get critic value
        return self.critic(obs)

class RSLRLFormatILTrainer:
    def __init__(self, 
                 dataset_path,
                 save_dir="/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format",
                 device="cuda",
                 use_wandb=True,
                 use_normalization=False,
                 critic_weight=0.5,
                 train_action_std=False):  # Whether to train the action std
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = os.path.expanduser(save_dir)
        self.use_normalization = use_normalization
        self.critic_weight = critic_weight
        self.train_action_std = train_action_std
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        self.train_dataset = MiniPupperILDataset(dataset_path, split='train')
        self.val_dataset = MiniPupperILDataset(dataset_path, split='val')
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        # Create model with RSL RL structure
        self.model = ActorCriticMLP().to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Actor parameters: {sum(p.numel() for p in self.model.actor.parameters()):,}")
        print(f"Critic parameters: {sum(p.numel() for p in self.model.critic.parameters()):,}")
        print(f"Action std (std) shape: {self.model.std.shape}")
        
        # Training setup - optimize actor, critic, and optionally std
        if self.train_action_std:
            self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        else:
            # Exclude std from optimization
            params = list(self.model.actor.parameters()) + list(self.model.critic.parameters())
            self.optimizer = optim.Adam(params, lr=3e-4)
            
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Loss functions
        self.actor_criterion = nn.MSELoss()
        self.critic_criterion = nn.MSELoss()
        
        # Setup normalization (RSL RL style)
        if self.use_normalization:
            # Initialize running statistics
            self.obs_rms_mean = torch.zeros(48)
            self.obs_rms_var = torch.ones(48)
            self.obs_count = 0
        else:
            # Identity normalization
            self.obs_rms_mean = torch.zeros(48)
            self.obs_rms_var = torch.ones(48)
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="minipupper-il-rsl-format",
                config={
                    "dataset": dataset_path,
                    "model": "ActorCritic-RSL-Format",
                    "hidden_dims": [512, 256, 128],
                    "lr": 3e-4,
                    "batch_size": 256,
                    "use_normalization": use_normalization,
                    "critic_weight": critic_weight,
                    "train_action_std": train_action_std
                }
            )
            
    def compute_returns(self, trajectory_length, gamma=0.99):
        """
        Compute discounted returns for a trajectory
        For IL, we can use a simple heuristic: good demonstrations get high returns
        """
        # Simple approach: constant positive reward for following demonstration
        rewards = torch.ones(trajectory_length) * 1.0
        
        # Compute discounted returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(trajectory_length)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
            
        return returns
            
    def update_obs_statistics(self, obs_batch):
        """Update running mean and variance (Welford's algorithm)"""
        if not self.use_normalization:
            return
            
        batch_mean = obs_batch.mean(dim=0)
        batch_var = obs_batch.var(dim=0, unbiased=False)
        batch_count = obs_batch.shape[0]
        
        delta = batch_mean - self.obs_rms_mean
        total_count = self.obs_count + batch_count
        
        self.obs_rms_mean = self.obs_rms_mean + delta * batch_count / total_count
        m_a = self.obs_rms_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.obs_count * batch_count / total_count
        self.obs_rms_var = M2 / total_count
        
        self.obs_count = total_count
            
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        num_batches = 0
        
        # Track per-joint errors
        joint_errors = torch.zeros(12)
        
        for batch in tqdm(dataloader, desc="Training"):
            obs = batch['obs'].to(self.device)
            actions = batch['action'].to(self.device)
            batch_size = obs.shape[0]
            
            # Update observation statistics
            self.update_obs_statistics(obs.cpu())
            
            # Normalize observations if using normalization
            if self.use_normalization:
                obs_mean = self.obs_rms_mean.to(self.device)
                obs_std = torch.sqrt(self.obs_rms_var + 1e-8).to(self.device)
                obs = (obs - obs_mean) / obs_std
            
            # Forward pass - get both actor and critic outputs
            pred_actions, pred_values = self.model(obs)
            
            # Compute returns for critic training
            # Since we're doing IL, we assume demonstrations are optimal
            # So we assign high value to demonstrated states
            returns = self.compute_returns(batch_size).to(self.device)
            returns = returns.unsqueeze(1)  # Shape: [batch_size, 1]
            
            # Actor loss: match demonstrated actions
            actor_loss = self.actor_criterion(pred_actions, actions)
            
            # Track per-joint errors
            with torch.no_grad():
                batch_joint_errors = torch.abs(pred_actions - actions).mean(dim=0)
                joint_errors += batch_joint_errors.cpu()
            
            # Critic loss: predict returns
            critic_loss = self.critic_criterion(pred_values, returns)
            
            # Combined loss
            loss = actor_loss + self.critic_weight * critic_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        # Log per-joint errors
        joint_errors /= num_batches
        joint_names = ['FR-Hip', 'FR-Thigh', 'FR-Knee', 
                       'FL-Hip', 'FL-Thigh', 'FL-Knee',
                       'RR-Hip', 'RR-Thigh', 'RR-Knee',
                       'RL-Hip', 'RL-Thigh', 'RL-Knee']
        
        print("\nPer-joint errors:")
        for i, (name, error) in enumerate(zip(joint_names, joint_errors)):
            print(f"  {name}: {error:.4f}")
            if self.use_wandb:
                wandb.log({f'joint_error/{name}': error})
            
        return {
            'total_loss': total_loss / num_batches,
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches,
            'joint_errors': joint_errors
        }
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                obs = batch['obs'].to(self.device)
                actions = batch['action'].to(self.device)
                batch_size = obs.shape[0]
                
                # Normalize observations if using normalization
                if self.use_normalization:
                    obs_mean = self.obs_rms_mean.to(self.device)
                    obs_std = torch.sqrt(self.obs_rms_var + 1e-8).to(self.device)
                    obs = (obs - obs_mean) / obs_std
                
                pred_actions, pred_values = self.model(obs)
                
                # Compute returns
                returns = self.compute_returns(batch_size).to(self.device).unsqueeze(1)
                
                # Losses
                actor_loss = self.actor_criterion(pred_actions, actions)
                critic_loss = self.critic_criterion(pred_values, returns)
                loss = actor_loss + self.critic_weight * critic_loss
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_loss += loss.item()
                num_batches += 1
                
        return {
            'total_loss': total_loss / num_batches,
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches
        }
    
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
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                'train_loss': train_metrics['total_loss'],
                'train_actor_loss': train_metrics['actor_loss'],
                'train_critic_loss': train_metrics['critic_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_actor_loss': val_metrics['actor_loss'],
                'val_critic_loss': val_metrics['critic_loss'],
                'lr': self.scheduler.get_last_lr()[0]
            }
            
            # Log action std
            action_std = torch.exp(self.model.std).mean().item()
            metrics['action_std'] = action_std
            
            print(f"Train - Total: {train_metrics['total_loss']:.6f}, "
                  f"Actor: {train_metrics['actor_loss']:.6f}, "
                  f"Critic: {train_metrics['critic_loss']:.6f}")
            print(f"Val - Total: {val_metrics['total_loss']:.6f}, "
                  f"Actor: {val_metrics['actor_loss']:.6f}, "
                  f"Critic: {val_metrics['critic_loss']:.6f}")
            print(f"Action Std: {action_std:.4f}")
            
            if self.use_wandb:
                wandb.log(metrics)
            
            # Save best model based on actor loss (most important for IL)
            if val_metrics['actor_loss'] < best_val_loss:
                best_val_loss = val_metrics['actor_loss']
                self.save_checkpoint('best_model_rsl_format.pt', epoch, val_metrics)
                print(f"New best model saved (val_actor_loss: {val_metrics['actor_loss']:.6f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}_rsl_format.pt', epoch, val_metrics)
                
        print(f"\nTraining complete! Best val actor loss: {best_val_loss:.6f}")
        
    def save_checkpoint(self, filename, epoch, val_metrics):
        """Save model checkpoint in RSL RL format"""
        
        # Get the full state dict with proper naming
        model_state_dict = {}
        
        # Add actor weights with 'actor.' prefix
        for name, param in self.model.actor.named_parameters():
            model_state_dict[f'actor.{name}'] = param.data
            
        # Add critic weights with 'critic.' prefix
        for name, param in self.model.critic.named_parameters():
            model_state_dict[f'critic.{name}'] = param.data
        
        # Add action std (std) - this is what was missing!
        model_state_dict['std'] = self.model.std.data
        
        # Create RSL RL compatible checkpoint
        checkpoint = {
            'model_state_dict': model_state_dict,
            'obs_rms_mean': self.obs_rms_mean,
            'obs_rms_var': self.obs_rms_var,
            'num_obs': 48,
            'num_actions': 12,
            'iter': epoch,  # RSL RL uses 'iter' instead of 'epoch'
            'infos': {
                'val_actor_loss': val_metrics['actor_loss'],
                'val_critic_loss': val_metrics['critic_loss'],
                'val_total_loss': val_metrics['total_loss'],
                'trained_with_il': True,
                'il_dataset': self.train_dataset.hdf5_path,
                'epoch': epoch,
                'val_metrics': val_metrics,
                'use_normalization': self.use_normalization,
                'critic_weight': self.critic_weight,
                'obs_mean': self.train_dataset.obs_mean,
                'obs_std': self.train_dataset.obs_std,
                'action_mean': self.train_dataset.action_mean,
                'action_std': self.train_dataset.action_std,
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved RSL RL format checkpoint to {path}")

def main():
    parser = argparse.ArgumentParser(description="Train IL policy in RSL RL format with actor-critic")
    parser.add_argument("--dataset", default="/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--device", default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--use-normalization", action="store_true",
                        help="Use observation normalization (RSL RL style)")
    parser.add_argument("--critic-weight", type=float, default=0.5,
                        help="Weight for critic loss (default: 0.5)")
    parser.add_argument("--train-action-std", action="store_true",
                        help="Train the action standard deviation")
    parser.add_argument("--test-only", action="store_true",
                        help="Only create test script, don't train")
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Test mode not implemented in this version.")
        return
    
    # Expand dataset path
    dataset_path = os.path.expanduser(args.dataset)
    
    # Create trainer
    trainer = RSLRLFormatILTrainer(
        dataset_path=dataset_path,
        device=args.device,
        use_wandb=not args.no_wandb,
        use_normalization=args.use_normalization,
        critic_weight=args.critic_weight,
        train_action_std=args.train_action_std
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
        
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
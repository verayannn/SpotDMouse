#!/usr/bin/env python3
"""
Extract and add observation normalization statistics to an existing trained model.
Run this AFTER training or on any existing checkpoint.
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add isaaclab to path if needed
sys.path.append('/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl')

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Add observation stats to existing checkpoint")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Spot-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=64, help="Number of envs for stats collection")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to collect stats")
# AppLauncher will add its own headless argument
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest of the script after sim is launched"""

import gymnasium as gym
import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCritic


def extract_stats_from_trained_model(checkpoint_path, task_name, num_envs=64, num_steps=1000):
    """
    Load a trained model and collect observation statistics by running it in the environment.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        task_name: Name of the Isaac Lab task
        num_envs: Number of environments to run in parallel
        num_steps: Number of steps to collect statistics
    """
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING OBSERVATION STATISTICS")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {task_name}")
    print(f"Num envs: {num_envs}")
    print(f"Collection steps: {num_steps}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine device - use same device as the checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load checkpoint on the appropriate device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")  # Show first 5 keys
    
    # Check if stats already exist
    if 'obs_rms_mean' in checkpoint and 'obs_rms_var' in checkpoint:
        print("\nWARNING: Checkpoint already contains observation statistics!")
        print(f"Existing mean range: [{checkpoint['obs_rms_mean'].min():.3f}, {checkpoint['obs_rms_mean'].max():.3f}]")
        print(f"Existing var range: [{checkpoint['obs_rms_var'].min():.3f}, {checkpoint['obs_rms_var'].max():.3f}]")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create environment
    print(f"\nCreating environment...")
    env_cfg = gym.spec(task_name).kwargs['cfg']
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    
    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    print(f"Environment created:")
    print(f"  - Num observations: {env.num_obs}")
    print(f"  - Num actions: {env.num_actions}")
    
    # Create and load the policy
    print(f"\nLoading policy...")
    actor_critic = ActorCritic(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        hidden_dims=[512, 256, 128]  # Match your architecture
    ).to(device)
    
    # Load the model weights
    actor_critic.load_state_dict(checkpoint['model_state_dict'])
    actor_critic.eval()
    print("Policy loaded successfully!")
    
    # Collect observations using Welford's algorithm for numerical stability
    print(f"\nCollecting observation statistics...")
    
    class WelfordStats:
        """Online mean and variance calculation"""
        def __init__(self, shape):
            self.n = 0
            self.mean = np.zeros(shape, dtype=np.float64)
            self.M2 = np.zeros(shape, dtype=np.float64)
        
        def update(self, batch):
            # batch shape: (num_envs, obs_dim)
            batch_np = batch.cpu().numpy().astype(np.float64)
            for x in batch_np:
                self.n += 1
                delta = x - self.mean
                self.mean += delta / self.n
                delta2 = x - self.mean
                self.M2 += delta * delta2
        
        def get_stats(self):
            variance = self.M2 / (self.n - 1) if self.n > 1 else self.M2
            return self.mean.astype(np.float32), variance.astype(np.float32)
    
    # Initialize statistics collector
    stats_collector = WelfordStats(env.num_obs)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Collect observations
    with torch.no_grad():
        for step in range(num_steps):
            # Update statistics with current observations
            stats_collector.update(obs)
            
            # Get actions from policy
            actions = actor_critic.act_inference(obs)
            
            # Step environment
            obs, _, dones, _, _ = env.step(actions)
            
            # Progress bar
            if step % 100 == 0:
                print(f"  Step {step}/{num_steps}...")
    
    # Get final statistics
    obs_mean, obs_var = stats_collector.get_stats()
    
    print(f"\nStatistics collected from {stats_collector.n} observations")
    print(f"  Mean shape: {obs_mean.shape}")
    print(f"  Mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"  Var range: [{obs_var.min():.3f}, {obs_var.max():.3f}]")
    
    # Add statistics to checkpoint
    checkpoint['obs_rms_mean'] = obs_mean
    checkpoint['obs_rms_var'] = obs_var
    checkpoint['num_obs'] = env.num_obs
    checkpoint['num_actions'] = env.num_actions
    
    # Save enhanced checkpoint
    output_path = checkpoint_path.replace('.pt', '_with_stats.pt')
    torch.save(checkpoint, output_path)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS: Saved enhanced checkpoint")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Clean up
    env.close()
    
    return output_path


def main():
    """Main function"""
    try:
        # Run the stats extraction
        output_path = extract_stats_from_trained_model(
            checkpoint_path=args.checkpoint,
            task_name=args.task,
            num_envs=args.num_envs,
            num_steps=args.num_steps
        )
        
        # Create a convenient symlink
        if output_path:
            symlink_path = Path(args.checkpoint).parent / "latest_with_stats.pt"
            if symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(Path(output_path).name)
            print(f"Created symlink: {symlink_path}")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
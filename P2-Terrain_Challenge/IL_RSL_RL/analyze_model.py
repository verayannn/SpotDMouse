#!/usr/bin/env python3
"""
Analyze IL model performance by comparing with dataset and testing random commands
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import h5py
from tqdm import tqdm
from collections import defaultdict

# Import your dataset class
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
        
    def forward(self, obs):
        return self.actor(obs)


class ModelAnalyzer:
    def __init__(self, checkpoint_path, dataset_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = MLPPolicy().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine state dict
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Print keys for debugging
        print(f"Checkpoint keys: {list(state_dict.keys())[:8]}")
        
        # Try loading with and without 'actor.' prefix
        try:
            self.model.load_state_dict(state_dict)
            print("Loaded model state dict directly.")
        except RuntimeError as e:
            print(f"Direct load failed: {e}\nTrying with 'actor.' prefix...")
            # Add 'actor.' prefix if missing
            new_state_dict = {f'actor.{k}': v for k, v in state_dict.items()}
            try:
                self.model.load_state_dict(new_state_dict)
                print("Loaded model state dict with 'actor.' prefix.")
            except RuntimeError as e2:
                print(f"Failed to load model state dict: {e2}")
                raise
            
        self.model.eval()
        
        # Load normalization stats with proper defaults
        self.obs_mean = checkpoint.get('obs_mean', None)
        self.obs_std = checkpoint.get('obs_std', None)
        self.action_mean = checkpoint.get('action_mean', None)
        self.action_std = checkpoint.get('action_std', None)
        
        # Ensure stats are not None and use appropriate defaults
        if self.obs_mean is None:
            print("Warning: obs_mean not found, using zeros")
            self.obs_mean = np.zeros(48)
        if self.obs_std is None:
            print("Warning: obs_std not found, using ones")
            self.obs_std = np.ones(48)
        if self.action_mean is None:
            print("Warning: action_mean not found, using zeros")
            self.action_mean = np.zeros(12)
        if self.action_std is None:
            print("Warning: action_std not found, using ones")
            self.action_std = np.ones(12)

        # Convert to tensors
        self.obs_mean = torch.tensor(self.obs_mean, dtype=torch.float32).to(self.device)
        self.obs_std = torch.tensor(self.obs_std, dtype=torch.float32).to(self.device)
        self.action_mean = torch.tensor(self.action_mean, dtype=torch.float32).to(self.device)
        self.action_std = torch.tensor(self.action_std, dtype=torch.float32).to(self.device)
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}")
        self.dataset = MiniPupperILDataset(dataset_path, split='val')
        print(f"Dataset size: {len(self.dataset)} samples")
        
        # Joint names for plotting
        self.joint_names = [
            'FL_hip', 'FL_thigh', 'FL_calf',
            'FR_hip', 'FR_thigh', 'FR_calf', 
            'RL_hip', 'RL_thigh', 'RL_calf',
            'RR_hip', 'RR_thigh', 'RR_calf'
        ]
        
    def normalize_obs(self, obs):
        """Normalize observation"""
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    def denormalize_action(self, action):
        """Denormalize action"""
        return action * self.action_std + self.action_mean
    
    def compare_with_dataset(self, num_samples=1000):
        """Compare model predictions with dataset actions"""
        print(f"\nComparing model predictions with dataset ({num_samples} samples)...")
        
        # Sample from dataset
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        
        predictions = []
        ground_truths = []
        observations = []
        errors = []
        
        with torch.no_grad():
            for idx in tqdm(indices):
                sample = self.dataset[idx]
                obs = sample['obs'].to(self.device)
                action_gt = sample['action'].to(self.device)
                
                # Get model prediction
                obs_norm = self.normalize_obs(obs.unsqueeze(0))
                action_pred_norm = self.model(obs_norm)
                action_pred = self.denormalize_action(action_pred_norm).squeeze(0)
                
                # Store results
                predictions.append(action_pred.cpu().numpy())
                ground_truths.append(action_gt.cpu().numpy())
                observations.append(obs.cpu().numpy())
                
                # Calculate error
                error = torch.abs(action_pred - action_gt).cpu().numpy()
                errors.append(error)
        
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        observations = np.array(observations)
        errors = np.array(errors)
        
        # Calculate statistics
        mse = np.mean((predictions - ground_truths) ** 2)
        mae = np.mean(np.abs(predictions - ground_truths))
        
        print(f"Overall MSE: {mse:.6f}")
        print(f"Overall MAE: {mae:.6f}")
        
        # Per-joint statistics
        joint_mse = np.mean((predictions - ground_truths) ** 2, axis=0)
        joint_mae = np.mean(np.abs(predictions - ground_truths), axis=0)
        
        print("\nPer-joint errors:")
        for i, joint_name in enumerate(self.joint_names):
            print(f"  {joint_name}: MSE={joint_mse[i]:.6f}, MAE={joint_mae[i]:.6f}")
        
        return {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'observations': observations,
            'errors': errors,
            'mse': mse,
            'mae': mae,
            'joint_mse': joint_mse,
            'joint_mae': joint_mae
        }
    
    def test_random_commands(self, num_tests=10):
        """Test model with random command velocities"""
        print(f"\nTesting model with {num_tests} random commands...")
        
        results = []
        
        with torch.no_grad():
            for i in range(num_tests):
                # Generate random command velocities similar to teleop_record
                cmd_vel = self._generate_random_command()
                
                # Create observation with this command
                obs = self._create_observation_with_command(cmd_vel)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get model prediction
                obs_norm = self.normalize_obs(obs_tensor)
                action_pred_norm = self.model(obs_norm)
                action_pred = self.denormalize_action(action_pred_norm).squeeze(0).cpu().numpy()
                
                results.append({
                    'command': cmd_vel,
                    'observation': obs,
                    'action': action_pred
                })
        
        return results
    
    def _generate_random_command(self):
        """Generate random command velocity similar to teleop demos"""
        # Command types from teleop_record
        command_types = [
            # Standing still
            {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': 0.0},
            # Forward/backward
            {'linear_x': np.random.uniform(-0.4, 0.4), 'linear_y': 0.0, 'angular_z': 0.0},
            # Sideways
            {'linear_x': 0.0, 'linear_y': np.random.uniform(-0.3, 0.3), 'angular_z': 0.0},
            # Turning
            {'linear_x': 0.0, 'linear_y': 0.0, 'angular_z': np.random.uniform(-0.4, 0.4)},
            # Combined motion
            {'linear_x': np.random.uniform(-0.3, 0.3), 
             'linear_y': np.random.uniform(-0.2, 0.2), 
             'angular_z': np.random.uniform(-0.3, 0.3)},
        ]
        
        # Randomly select a command type
        cmd = np.random.choice(command_types)
        return np.array([cmd['linear_x'], cmd['linear_y'], cmd['angular_z']])
    
    def _create_observation_with_command(self, cmd_vel):
        """Create a synthetic observation with given command velocity"""
        obs = np.zeros(48)
        
        # Set command velocities (first 3 dims)
        obs[0:3] = cmd_vel
        
        # Set reasonable defaults for other dimensions
        # Joint positions (12 dims) - near zero is standing
        obs[3:15] = np.random.normal(0, 0.1, 12)
        
        # Joint velocities (12 dims)
        obs[15:27] = np.random.normal(0, 0.05, 12)
        
        # Previous actions (12 dims)
        obs[27:39] = obs[3:15]  # Same as current positions
        
        # Gravity vector (3 dims) - pointing down
        obs[39:42] = np.array([0., 0., -9.81]) + np.random.normal(0, 0.1, 3)
        
        # Angular velocity (3 dims)
        obs[42:45] = np.random.normal(0, 0.1, 3)
        
        # Gait phase (2 dims) - sin/cos of phase
        phase = np.random.uniform(0, 2*np.pi)
        obs[45:47] = np.array([np.sin(phase), np.cos(phase)])
        
        # Foot contact (1 dim)
        obs[47] = 1.0
        
        return obs
    
    def plot_dataset_comparison(self, results, save_path='analysis_plots'):
        """Plot comparison between model predictions and dataset"""
        os.makedirs(save_path, exist_ok=True)
        
        predictions = results['predictions']
        ground_truths = results['ground_truths']
        errors = results['errors']
        
        # 1. Joint-wise prediction vs ground truth scatter plots
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, joint_name in enumerate(self.joint_names):
            ax = axes[i]
            ax.scatter(ground_truths[:, i], predictions[:, i], alpha=0.5, s=10)
            ax.plot([-1, 1], [-1, 1], 'r--', label='Perfect prediction')
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Prediction')
            ax.set_title(f'{joint_name}')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'joint_predictions_scatter.png'), dpi=150)
        plt.close()
        
        # 2. Error distribution per joint
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Box plot of errors
        box_data = [errors[:, i] for i in range(12)]
        bp = ax.boxplot(box_data, labels=self.joint_names, patch_artist=True)
        
        # Color by leg
        colors = ['lightblue', 'lightblue', 'lightblue',  # FL
                  'lightgreen', 'lightgreen', 'lightgreen',  # FR
                  'lightcoral', 'lightcoral', 'lightcoral',  # RL
                  'lightyellow', 'lightyellow', 'lightyellow']  # RR
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Absolute Error')
        ax.set_title('Prediction Error Distribution by Joint')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'error_distribution.png'), dpi=150)
        plt.close()
        
        # 3. Time series comparison for a few samples
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Select a few random samples to plot
        sample_indices = np.random.choice(len(predictions), 5, replace=False)
        
        for leg_idx, (leg_name, joint_indices) in enumerate([
            ('Front Left', [0, 1, 2]),
            ('Front Right', [3, 4, 5]),
            ('Rear Left', [6, 7, 8]),
            ('Rear Right', [9, 10, 11])
        ]):
            ax = axes[leg_idx]
            
            for sample_idx in sample_indices:
                # Plot ground truth
                for j, joint_idx in enumerate(joint_indices):
                    ax.plot(joint_idx + 0.2*j, ground_truths[sample_idx, joint_idx], 
                           'go', markersize=8, alpha=0.7)
                    ax.plot(joint_idx + 0.2*j, predictions[sample_idx, joint_idx], 
                           'ro', markersize=6, alpha=0.7)
            
            ax.set_ylabel(f'{leg_name}\nJoint Angle')
            ax.set_xticks(joint_indices)
            ax.set_xticklabels(['Hip', 'Thigh', 'Calf'])
            ax.grid(True, alpha=0.3)
            ax.legend(['Ground Truth', 'Prediction'], loc='upper right')
        
        axes[0].set_title('Sample Joint Angle Comparisons')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'sample_comparisons.png'), dpi=150)
        plt.close()
        
        # 4. Command velocity vs prediction error heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract command velocities from observations
        cmd_vels = results['observations'][:, :3]  # First 3 dims are cmd_vel
        
        # Create 2D histogram for forward velocity vs turning velocity
        H, xedges, yedges = np.histogram2d(
            cmd_vels[:, 0],  # linear.x
            cmd_vels[:, 2],  # angular.z
            bins=20,
            weights=np.mean(errors, axis=1)  # Average error across joints
        )
        
        im = ax.imshow(H.T, origin='lower', aspect='auto', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                      cmap='hot')
        
        ax.set_xlabel('Linear Velocity X (m/s)')
        ax.set_ylabel('Angular Velocity Z (rad/s)')
        ax.set_title('Average Prediction Error vs Command Velocity')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Mean Absolute Error')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'error_vs_command.png'), dpi=150)
        plt.close()
        
        print(f"\nDataset comparison plots saved to {save_path}/")
    
    def plot_random_command_results(self, results, save_path='analysis_plots'):
        """Plot results from random command tests"""
        os.makedirs(save_path, exist_ok=True)
        
        num_tests = len(results)
        
        # 1. Command velocities and resulting actions
        fig, axes = plt.subplots(num_tests, 2, figsize=(12, 3*num_tests))
        
        if num_tests == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(results):
            cmd = result['command']
            action = result['action']
            
            # Plot command
            ax = axes[i, 0]
            ax.bar(['Linear X', 'Linear Y', 'Angular Z'], cmd)
            ax.set_ylabel('Velocity')
            ax.set_title(f'Test {i+1}: Command Velocities')
            ax.grid(True, alpha=0.3)
            
            # Plot resulting actions
            ax = axes[i, 1]
            x = np.arange(12)
            ax.bar(x, action)
            ax.set_xticks(x)
            ax.set_xticklabels(self.joint_names, rotation=45, ha='right')
            ax.set_ylabel('Joint Position')
            ax.set_title(f'Test {i+1}: Predicted Joint Positions')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'random_commands_test.png'), dpi=150)
        plt.close()
        
        # 2. Action patterns for different command types
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Group results by command type
        forward_actions = []
        turn_actions = []
        sideways_actions = []
        combined_actions = []
        
        for result in results:
            cmd = result['command']
            action = result['action']
            
            if abs(cmd[0]) > 0.1 and abs(cmd[1]) < 0.1 and abs(cmd[2]) < 0.1:
                forward_actions.append(action)
            elif abs(cmd[2]) > 0.1 and abs(cmd[0]) < 0.1 and abs(cmd[1]) < 0.1:
                turn_actions.append(action)
            elif abs(cmd[1]) > 0.1 and abs(cmd[0]) < 0.1 and abs(cmd[2]) < 0.1:
                sideways_actions.append(action)
            else:
                combined_actions.append(action)
        
        # Plot average patterns
        action_groups = [
            ('Forward/Backward', forward_actions),
            ('Turning', turn_actions),
            ('Sideways', sideways_actions),
            ('Combined', combined_actions)
        ]
        
        for ax, (title, actions) in zip(axes.flatten(), action_groups):
            if actions:
                actions = np.array(actions)
                mean_action = np.mean(actions, axis=0)
                std_action = np.std(actions, axis=0)
                
                x = np.arange(12)
                ax.bar(x, mean_action, yerr=std_action, capsize=5)
                ax.set_xticks(x)
                ax.set_xticklabels(self.joint_names, rotation=45, ha='right')
                ax.set_ylabel('Joint Position')
                ax.set_title(f'{title} Motion Pattern')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No samples', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'action_patterns.png'), dpi=150)
        plt.close()
        
        # 3. Joint coordination heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate correlation between joints across all actions
        all_actions = np.array([r['action'] for r in results])
        corr_matrix = np.corrcoef(all_actions.T)
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        ax.set_xticklabels(self.joint_names, rotation=45, ha='right')
        ax.set_yticklabels(self.joint_names)
        ax.set_title('Joint Coordination Correlation Matrix')
        
        # Add correlation values
        for i in range(12):
            for j in range(12):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'joint_coordination.png'), dpi=150)
        plt.close()
        
        print(f"\nRandom command test plots saved to {save_path}/")
    
    def generate_report(self, dataset_results, random_results, save_path='analysis_plots'):
        """Generate a summary report"""
        report_path = os.path.join(save_path, 'analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Mini Pupper IL Model Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset comparison summary
            f.write("1. Dataset Comparison Results\n")
            f.write("-" * 30 + "\n")
            f.write(f"Samples analyzed: {len(dataset_results['predictions'])}\n")
            f.write(f"Overall MSE: {dataset_results['mse']:.6f}\n")
            f.write(f"Overall MAE: {dataset_results['mae']:.6f}\n\n")
            
            f.write("Per-joint errors (MAE):\n")
            for i, joint_name in enumerate(self.joint_names):
                f.write(f"  {joint_name:12s}: {dataset_results['joint_mae'][i]:.6f}\n")
            
            # Find best and worst joints
            best_joint_idx = np.argmin(dataset_results['joint_mae'])
            worst_joint_idx = np.argmax(dataset_results['joint_mae'])
            f.write(f"\nBest predicted joint: {self.joint_names[best_joint_idx]} "
                   f"(MAE: {dataset_results['joint_mae'][best_joint_idx]:.6f})\n")
            f.write(f"Worst predicted joint: {self.joint_names[worst_joint_idx]} "
                   f"(MAE: {dataset_results['joint_mae'][worst_joint_idx]:.6f})\n\n")
            
            # Random command test summary
            f.write("2. Random Command Test Results\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of tests: {len(random_results)}\n\n")
            
            f.write("Command types tested:\n")
            for i, result in enumerate(random_results):
                cmd = result['command']
                f.write(f"  Test {i+1}: linear_x={cmd[0]:6.3f}, "
                       f"linear_y={cmd[1]:6.3f}, angular_z={cmd[2]:6.3f}\n")
            
            f.write("\n3. Recommendations\n")
            f.write("-" * 30 + "\n")
            
            # Generate recommendations based on results
            if dataset_results['mae'] < 0.05:
                f.write("- Model shows excellent prediction accuracy (MAE < 0.05)\n")
            elif dataset_results['mae'] < 0.1:
                f.write("- Model shows good prediction accuracy (MAE < 0.1)\n")
            else:
                f.write("- Model may benefit from additional training (MAE > 0.1)\n")
            
            # Check for specific joint issues
            joint_mae = dataset_results['joint_mae']
            if np.std(joint_mae) > 0.05:
                f.write("- Large variance in per-joint errors suggests uneven learning\n")
                f.write("  Consider collecting more diverse demonstrations\n")
            
            # Check for leg-specific patterns
            leg_errors = [
                np.mean(joint_mae[0:3]),  # FL
                np.mean(joint_mae[3:6]),  # FR
                np.mean(joint_mae[6:9]),  # RL
                np.mean(joint_mae[9:12])  # RR
            ]
            
            if np.std(leg_errors) > 0.03:
                f.write("- Uneven performance across legs detected\n")
                worst_leg_idx = np.argmax(leg_errors)
                leg_names = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
                f.write(f"  {leg_names[worst_leg_idx]} leg shows highest errors\n")
        
        print(f"\nAnalysis report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze IL model performance")
    parser.add_argument("--checkpoint", 
                       default="/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--dataset",
                       default="/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5",
                       help="Path to HDF5 dataset")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of dataset samples to analyze")
    parser.add_argument("--num-random", type=int, default=10,
                       help="Number of random command tests")
    parser.add_argument("--output", default="analysis_plots",
                       help="Output directory for plots and report")
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ModelAnalyzer(args.checkpoint, args.dataset, args.device)
    
    # 1. Compare with dataset
    print("\n=== Dataset Comparison ===")
    dataset_results = analyzer.compare_with_dataset(args.num_samples)
    analyzer.plot_dataset_comparison(dataset_results, args.output)
    
    # 2. Test random commands
    print("\n=== Random Command Testing ===")
    random_results = analyzer.test_random_commands(args.num_random)
    analyzer.plot_random_command_results(random_results, args.output)
    
    # 3. Generate report
    analyzer.generate_report(dataset_results, random_results, args.output)
    
    print(f"\nAnalysis complete! Results saved to {args.output}/")
    print("\nGenerated files:")
    print("  - joint_predictions_scatter.png: Prediction vs ground truth for each joint")
    print("  - error_distribution.png: Error distribution across joints")
    print("  - sample_comparisons.png: Sample-wise comparison")
    print("  - error_vs_command.png: Error heatmap vs command velocity")
    print("  - random_commands_test.png: Random command test results")
    print("  - action_patterns.png: Action patterns for different command types")
    print("  - joint_coordination.png: Joint coordination correlation")
    print("  - analysis_report.txt: Summary report with recommendations")


if __name__ == "__main__":
    main()
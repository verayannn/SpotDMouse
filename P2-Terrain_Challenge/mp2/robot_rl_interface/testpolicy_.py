import numpy as np
import torch
import time
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

class DiagnosticMLPController:
    """
    Diagnostic version to understand what the MLP expects and outputs.
    """
    def __init__(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        self.isaac_defaults = np.array([
             0.0, 0.785, -1.57,
             0.0, 0.785, -1.57,
             0.0, 0.785, -1.57,
             0.0, 0.785, -1.57
        ])
        
        self.ACTION_SCALE = 0.5
        
    def test_policy_response(self):
        """
        Test what the policy outputs for various synthetic observations.
        This helps us understand if the policy is working correctly.
        """
        print("\n" + "="*70)
        print("POLICY RESPONSE DIAGNOSTIC")
        print("="*70)
        
        # Test 1: Standing still, zero command
        print("\n--- Test 1: Standing, zero velocity command ---")
        obs = self._make_observation(
            base_lin_vel=[0, 0, 0],
            base_ang_vel=[0, 0, 0],
            projected_gravity=[0, 0, -1],
            velocity_cmd=[0, 0, 0],
            joint_pos_rel=np.zeros(12),
            joint_vel=np.zeros(12),
            joint_effort=np.zeros(12),
            prev_actions=np.zeros(12)
        )
        actions = self._run_policy(obs)
        print(f"  Actions: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")
        print(f"  First 4: {actions[:4]}")
        
        # Test 2: Standing, forward command
        print("\n--- Test 2: Standing, forward velocity command (0.2 m/s) ---")
        obs = self._make_observation(
            base_lin_vel=[0, 0, 0],
            base_ang_vel=[0, 0, 0],
            projected_gravity=[0, 0, -1],
            velocity_cmd=[0.2, 0, 0],  # Forward command
            joint_pos_rel=np.zeros(12),
            joint_vel=np.zeros(12),
            joint_effort=np.zeros(12),
            prev_actions=np.zeros(12)
        )
        actions = self._run_policy(obs)
        print(f"  Actions: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")
        print(f"  First 4: {actions[:4]}")
        
        # Test 3: Simulate a few steps with forward command
        print("\n--- Test 3: Simulated rollout (10 steps, forward command) ---")
        prev_actions = np.zeros(12)
        joint_pos_rel = np.zeros(12)
        
        for step in range(10):
            obs = self._make_observation(
                base_lin_vel=[0.05, 0, 0],  # Some forward velocity feedback
                base_ang_vel=[0, 0, 0],
                projected_gravity=[0, 0, -1],
                velocity_cmd=[0.2, 0, 0],
                joint_pos_rel=joint_pos_rel,
                joint_vel=np.zeros(12),
                joint_effort=np.zeros(12),
                prev_actions=prev_actions
            )
            actions = self._run_policy(obs)
            
            # Simulate joint position update (actions are position targets relative to default)
            joint_pos_rel = actions * self.ACTION_SCALE * 0.1  # Partial movement toward target
            prev_actions = actions.copy()
            
            if step % 2 == 0:
                print(f"  Step {step}: actions=[{actions[0]:.2f}, {actions[1]:.2f}, {actions[2]:.2f}, ...] "
                      f"pos_rel=[{joint_pos_rel[0]:.3f}, {joint_pos_rel[1]:.3f}, {joint_pos_rel[2]:.3f}, ...]")
        
        # Test 4: What if robot is tilted?
        print("\n--- Test 4: Robot tilted forward (pitch), forward command ---")
        obs = self._make_observation(
            base_lin_vel=[0, 0, 0],
            base_ang_vel=[0, 0, 0],
            projected_gravity=[0.1, 0, -0.995],  # Slight forward tilt
            velocity_cmd=[0.2, 0, 0],
            joint_pos_rel=np.zeros(12),
            joint_vel=np.zeros(12),
            joint_effort=np.zeros(12),
            prev_actions=np.zeros(12)
        )
        actions = self._run_policy(obs)
        print(f"  Actions: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")
        
        # Test 5: Simulate what sim would see - with realistic dynamics
        print("\n--- Test 5: Realistic sim-like rollout (20 steps) ---")
        prev_actions = np.zeros(12)
        joint_pos_rel = np.zeros(12)
        joint_vel = np.zeros(12)
        
        print("  Step | Action Range | Joint Pos Range | Action[0:3]")
        print("  " + "-"*60)
        
        for step in range(20):
            # Add some noise like in training
            noise_pos = np.random.uniform(-0.01, 0.01, 12)
            noise_vel = np.random.uniform(-0.1, 0.1, 12)
            
            obs = self._make_observation(
                base_lin_vel=[0.1 * np.sin(step * 0.5), 0, 0],
                base_ang_vel=[0, 0, 0.05 * np.sin(step * 0.3)],
                projected_gravity=[0.02 * np.sin(step * 0.2), 0, -0.9998],
                velocity_cmd=[0.2, 0, 0],
                joint_pos_rel=joint_pos_rel + noise_pos,
                joint_vel=joint_vel + noise_vel,
                joint_effort=np.zeros(12),
                prev_actions=prev_actions
            )
            actions = self._run_policy(obs)
            
            # Simulate dynamics
            target_pos = actions * self.ACTION_SCALE
            joint_vel = (target_pos - joint_pos_rel) * 10  # Simple dynamics
            joint_vel = np.clip(joint_vel, -1.5, 1.5)
            joint_pos_rel = joint_pos_rel + joint_vel * 0.02  # 50Hz
            prev_actions = actions.copy()
            
            if step % 4 == 0:
                print(f"  {step:4d} | [{actions.min():+.2f}, {actions.max():+.2f}] | "
                      f"[{joint_pos_rel.min():+.3f}, {joint_pos_rel.max():+.3f}] | "
                      f"[{actions[0]:+.2f}, {actions[1]:+.2f}, {actions[2]:+.2f}]")
    
    def _make_observation(self, base_lin_vel, base_ang_vel, projected_gravity,
                          velocity_cmd, joint_pos_rel, joint_vel, joint_effort, prev_actions):
        """Construct observation in the exact order expected by policy."""
        return np.concatenate([
            np.array(base_lin_vel),      # 3
            np.array(base_ang_vel),      # 3
            np.array(projected_gravity), # 3
            np.array(velocity_cmd),      # 3
            np.array(joint_pos_rel),     # 12
            np.array(joint_vel),         # 12
            np.array(joint_effort),      # 12
            np.array(prev_actions)       # 12
        ]).astype(np.float32)
    
    def _run_policy(self, obs):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            actions = self.policy(obs_tensor).squeeze().numpy()
        return actions


if __name__ == "__main__":
    diag = DiagnosticMLPController("/home/ubuntu/mp2_mlp/policy_only.pt")
    diag.test_policy_response()
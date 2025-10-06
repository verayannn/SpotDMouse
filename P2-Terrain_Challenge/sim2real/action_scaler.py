import numpy as np
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

class SimToRealActionScaler:
    """
    Scales and filters MLP outputs from simulation-trained policies to real hardware.
    Accounts for differences in joint limits, dynamics, and servo characteristics.
    """
    
    def __init__(self):
        # Simulation parameters (numerically stable)
        self.sim_params = {
            'saturation_effort': 2.5,
            'velocity_limit': 10.0,  # rad/s
            'stiffness': 45.0,#OG: 80, Final/Works: 45.0
            'damping': 1.3,#OG: 2.0, Final/Works: 1.3
            'friction': 0.02,
            'armature': 0.005
        }
        
        # Real servo parameters (based on specs)
        self.real_params = {
            'saturation_effort': 0.343,  # 3.5 kg·cm converted to N·m
            'velocity_limit': 10.47,     # 0.1s/60° = 10.47 rad/s
            'stiffness': 100.0,
            'damping': 3.0,
            'friction': 0.02,
            'armature': 0.00001  # 12.5g servo
        }
        
        # Servo operational limits
        self.servo_limits = {
            'voltage_range': (4.8, 7.4),  # V
            'pulse_width_range': (500, 2500),  # μsec
            'angular_range': np.radians(180),  # Total range
            'dead_band': 2e-6,  # 2μsec in seconds
            'max_current': 1.2,  # A at stall
            'nominal_current': 0.17  # A no-load
        }
        
        # Calculate scaling factors
        self._compute_scaling_factors()
        
        # Initialize low-pass filter state for smooth transitions
        self.filter_state = None
        self.filter_alpha = 0.3  # Low-pass filter coefficient
        
    def _compute_scaling_factors(self):
        """Compute scaling factors between sim and real parameters."""
        # Effort scaling: real servo has much lower torque
        self.effort_scale = self.real_params['saturation_effort'] / self.sim_params['saturation_effort']
        
        # Velocity scaling: similar limits but account for dynamics
        self.velocity_scale = self.real_params['velocity_limit'] / self.sim_params['velocity_limit']
        
        # Damping compensation: real has higher damping
        self.damping_ratio = self.real_params['damping'] / self.sim_params['damping']
        
        # Inertia compensation: significant difference in armature
        self.inertia_ratio = self.real_params['armature'] / self.sim_params['armature']
        
    def scale_actions(self, 
                      actions: np.ndarray, 
                      current_joint_pos: Optional[np.ndarray] = None,
                      current_joint_vel: Optional[np.ndarray] = None,
                      use_filtering: bool = True) -> np.ndarray:
        """
        Scale actions from simulation to real hardware.
        
        Args:
            actions: Joint position targets from MLP (shape: [12] for Mini Pupper)
            current_joint_pos: Current joint positions for feedback
            current_joint_vel: Current joint velocities for damping compensation
            use_filtering: Apply low-pass filtering for smooth transitions
            
        Returns:
            Scaled and filtered actions suitable for real hardware
        """
        # Ensure numpy array
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
            
        scaled_actions = actions.copy()
        
        # 1. Apply effort-based scaling
        # Since sim uses higher torques, scale down position commands
        scaled_actions *= self.effort_scale
        
        # 2. Compensate for damping differences
        if current_joint_vel is not None:
            # Higher real damping requires anticipatory compensation
            damping_compensation = self.damping_ratio * current_joint_vel * 0.1
            scaled_actions -= damping_compensation
            
        # 3. Apply velocity limiting based on real servo specs
        if current_joint_pos is not None:
            # Calculate desired velocity
            desired_vel = (scaled_actions - current_joint_pos) / 0.01  # Assuming 100Hz control
            
            # Clip to real velocity limits
            vel_magnitude = np.abs(desired_vel)
            vel_limit_mask = vel_magnitude > self.real_params['velocity_limit']
            if np.any(vel_limit_mask):
                scale_factor = self.real_params['velocity_limit'] / (vel_magnitude[vel_limit_mask] + 1e-6)
                desired_vel[vel_limit_mask] *= scale_factor
                scaled_actions = current_joint_pos + desired_vel * 0.01
                
        # 4. Apply joint limits (Mini Pupper specific)
        scaled_actions = self._apply_joint_limits(scaled_actions)
        
        # 5. Apply low-pass filtering for smooth transitions
        if use_filtering:
            if self.filter_state is None:
                self.filter_state = scaled_actions.copy()
            else:
                scaled_actions = (self.filter_alpha * scaled_actions + 
                                 (1 - self.filter_alpha) * self.filter_state)
                self.filter_state = scaled_actions.copy()
                
        # 6. Apply dead-band compensation
        scaled_actions = self._apply_deadband_compensation(scaled_actions)
        
        return scaled_actions
    
    def _apply_joint_limits(self, actions: np.ndarray) -> np.ndarray:
        """Apply Mini Pupper joint limits."""
        # Mini Pupper joint limits (radians)
        # Format: [hip_fl, thigh_fl, calf_fl, hip_fr, thigh_fr, calf_fr, 
        #          hip_bl, thigh_bl, calf_bl, hip_br, thigh_br, calf_br]
        
        hip_limits = (-0.7, 0.7)      # ±40 degrees
        thigh_limits = (-1.0, 2.5)    # -57 to 143 degrees  
        calf_limits = (-2.5, -0.5)    # -143 to -28 degrees
        
        limited_actions = actions.copy()
        
        # Apply limits to each leg
        for leg_idx in range(4):
            base_idx = leg_idx * 3
            # Hip joint
            limited_actions[base_idx] = np.clip(actions[base_idx], *hip_limits)
            # Thigh joint  
            limited_actions[base_idx + 1] = np.clip(actions[base_idx + 1], *thigh_limits)
            # Calf joint
            limited_actions[base_idx + 2] = np.clip(actions[base_idx + 2], *calf_limits)
            
        return limited_actions
    
    def _apply_deadband_compensation(self, actions: np.ndarray) -> np.ndarray:
        """Compensate for servo dead-band."""
        # Add small perturbation to overcome dead-band
        deadband_threshold = 0.001  # radians
        
        # Only apply to small movements
        small_movement_mask = np.abs(actions) < deadband_threshold
        actions[small_movement_mask] = np.sign(actions[small_movement_mask]) * deadband_threshold
        
        return actions
    
    def scale_torques(self, torques: np.ndarray) -> np.ndarray:
        """
        Scale torque commands from simulation to real hardware.
        Used if your controller outputs torques instead of positions.
        """
        if isinstance(torques, torch.Tensor):
            torques = torques.detach().cpu().numpy()
            
        # Direct torque scaling
        scaled_torques = torques * self.effort_scale
        
        # Clip to servo limits
        scaled_torques = np.clip(scaled_torques, 
                                 -self.real_params['saturation_effort'],
                                 self.real_params['saturation_effort'])
        
        return scaled_torques
    
    def adapt_control_gains(self, kp: float, kd: float) -> Tuple[float, float]:
        """
        Adapt PD control gains from simulation to real hardware.
        
        Args:
            kp: Proportional gain from simulation
            kd: Derivative gain from simulation
            
        Returns:
            Adapted (kp, kd) for real hardware
        """
        # Scale gains based on dynamics differences
        real_kp = kp * self.effort_scale * (self.sim_params['stiffness'] / self.real_params['stiffness'])
        real_kd = kd * self.effort_scale * self.damping_ratio
        
        return real_kp, real_kd
    
    def reset_filter(self):
        """Reset the low-pass filter state."""
        self.filter_state = None


class MLPActionWrapper(nn.Module):
    """
    Wrapper for your trained MLP that applies sim-to-real scaling.
    """
    
    def __init__(self, trained_mlp: nn.Module, scaler: Optional[SimToRealActionScaler] = None):
        super().__init__()
        self.mlp = trained_mlp
        self.scaler = scaler if scaler is not None else SimToRealActionScaler()
        
    def forward(self, obs: torch.Tensor, 
                current_joint_pos: Optional[torch.Tensor] = None,
                current_joint_vel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with automatic scaling.
        
        Args:
            obs: Observation tensor
            current_joint_pos: Current joint positions
            current_joint_vel: Current joint velocities
            
        Returns:
            Scaled actions ready for real hardware
        """
        # Get raw actions from MLP
        raw_actions = self.mlp(obs)
        
        # Convert current states to numpy if provided
        curr_pos_np = current_joint_pos.detach().cpu().numpy() if current_joint_pos is not None else None
        curr_vel_np = current_joint_vel.detach().cpu().numpy() if current_joint_vel is not None else None
        
        # Apply scaling
        if len(raw_actions.shape) == 2:  # Batch processing
            scaled_actions = []
            for i in range(raw_actions.shape[0]):
                curr_pos_i = curr_pos_np[i] if curr_pos_np is not None else None
                curr_vel_i = curr_vel_np[i] if curr_vel_np is not None else None
                scaled = self.scaler.scale_actions(
                    raw_actions[i].detach().cpu().numpy(),
                    curr_pos_i,
                    curr_vel_i
                )
                scaled_actions.append(scaled)
            scaled_actions = np.stack(scaled_actions)
        else:  # Single action
            scaled_actions = self.scaler.scale_actions(
                raw_actions.detach().cpu().numpy(),
                curr_pos_np,
                curr_vel_np
            )
            
        return torch.tensor(scaled_actions, dtype=raw_actions.dtype, device=raw_actions.device)


# Example usage
def deploy_to_real_robot(trained_policy_path: str):
    """
    Example deployment function.
    """
    import torch
    
    # Load your trained MLP
    trained_mlp = torch.load(trained_policy_path)
    trained_mlp.eval()
    
    # Create scaler
    scaler = SimToRealActionScaler()
    
    # Wrap the MLP
    real_policy = MLPActionWrapper(trained_mlp, scaler)
    
    # Example control loop
    def control_loop(obs, current_joint_pos, current_joint_vel):
        with torch.no_grad():
            # Get scaled actions
            actions = real_policy(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(current_joint_pos, dtype=torch.float32),
                torch.tensor(current_joint_vel, dtype=torch.float32)
            )
            
        return actions.numpy()
    
    return real_policy, control_loop


# Testing the scaler
if __name__ == "__main__":
    # Create scaler
    scaler = SimToRealActionScaler()
    
    # Test with random actions
    test_actions = np.random.randn(12) * 0.5  # 12 joints for Mini Pupper
    current_pos = np.zeros(12)
    current_vel = np.random.randn(12) * 0.1
    
    scaled_actions = scaler.scale_actions(test_actions, current_pos, current_vel)
    
    print("Original actions range:", test_actions.min(), "to", test_actions.max())
    print("Scaled actions range:", scaled_actions.min(), "to", scaled_actions.max())
    print("Effort scale factor:", scaler.effort_scale)
    print("Damping ratio:", scaler.damping_ratio)
    
    # Test gain adaptation
    sim_kp, sim_kd = 100.0, 5.0
    real_kp, real_kd = scaler.adapt_control_gains(sim_kp, sim_kd)
    print(f"\nSim gains: Kp={sim_kp}, Kd={sim_kd}")
    print(f"Real gains: Kp={real_kp:.2f}, Kd={real_kd:.2f}")

#PIPELINE INTEGRATION
# In your ROS2 node or deployment script
# from your_training import load_trained_policy

# # Load your Isaac Lab trained MLP
# mlp = load_trained_policy("path/to/checkpoint")

# # Create the wrapped policy
# scaler = SimToRealActionScaler()
# real_policy = MLPActionWrapper(mlp, scaler)

# # In your control loop
# def ros2_callback(msg):
#     # Get observations (normalized as in training)
#     obs = process_observations(msg)
    
#     # Get current joint states
#     current_pos = get_joint_positions()
#     current_vel = get_joint_velocities()
    
#     # Get scaled actions
#     actions = real_policy(obs, current_pos, current_vel)
    
#     # Send to servos
#     publish_joint_commands(actions)
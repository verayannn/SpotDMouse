import torch
import numpy as np
from typing import Optional, List
from isaacsim.core.api import World
from isaacsim.core.articulations import ArticulationView
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.rotations import quat_to_rot_matrix

# Import the actual robot config from your training
import sys
# Ensure this path is correct for your environment
sys.path.append("/workspace/isaaclab")
from isaaclab_tasks.manager_based.locomotion.velocity.config.custom_quadruped_2.custom_quad import CUSTOM_QUAD_CFG

class MiniPupperIsaacLabPolicy:
    """Mini Pupper with IsaacLab-trained policy"""
    
    def __init__(
        self,
        prim_path: str,
        name: str = "minipupper",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ):
        self.prim_path = prim_path
        self.name = name
        self.position = position
        
        # Load the JIT policy
        # NOTE: Ensure this path points to the 'jit.pt' or 'policy.pt' exported 
        # specifically for deployment (which usually includes normalization).
        self.policy = torch.jit.load(
            "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/exported/policy.pt"
        )
        self.policy.eval()
        
        # --- CONFIGURATION MATCHING ---
        # 1. Expected Policy Joint Order (MUST match SpotActionsCfg)
        self.policy_joint_names = [
            "base_lf1", "lf1_lf2", "lf2_lf3",
            "base_rf1", "rf1_rf2", "rf2_rf3",
            "base_lb1", "lb1_lb2", "lb2_lb3",
            "base_rb1", "rb1_rb2", "rb2_rb3"
        ]
        
        # 2. Action Scale (From SpotActionsCfg)
        self.action_scale = 0.5
        
        # 3. Default Positions (Must match the order of self.policy_joint_names)
        # Verify these match the default angles of your robot in the idle standing pose
        self.default_pos = np.array([
            0.0, 0.785, -1.57,  # LF
            0.0, 0.785, -1.57,  # RF
            0.0, 0.785, -1.57,  # LB
            0.0, 0.785, -1.57,  # RB
        ])
        
        self._decimation = 10 
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        
        self.articulation_view = None
        self.joint_indices = []

    def initialize(self, world: World):
        """Initialize after world reset"""
        # Create articulation view
        self.articulation_view = ArticulationView(
            prim_paths_expr=self.prim_path,
            name=self.name + "_view"
        )
        world.scene.add(self.articulation_view)
        
        # CRITICAL FIX: Get the simulation joint names and map them to policy order
        # This ensures we don't send LF commands to the RB leg
        sim_joint_names = self.articulation_view.dof_names
        self.joint_indices = [sim_joint_names.index(name) for name in self.policy_joint_names]
        
        print(f"Policy Joint Order: {self.policy_joint_names}")
        print(f"Sim Joint Indices Mapping: {self.joint_indices}")

        # Set initial joint positions
        # We must reorder default_pos to match the SIMULATION order for initialization
        sim_ordered_defaults = np.zeros(12)
        for policy_i, sim_i in enumerate(self.joint_indices):
            sim_ordered_defaults[sim_i] = self.default_pos[policy_i]

        self.articulation_view.set_joint_positions(
            sim_ordered_defaults,
            joint_indices=list(range(12))
        )
        
    def get_observations(self, command):
        """Compute observations matching your training setup"""
        # Get states from articulation view
        root_vel = self.articulation_view.get_root_velocities()[0]
        root_pos, root_quat = self.articulation_view.get_world_poses()[0]
        
        # Transform velocities to body frame
        R_IB = quat_to_rot_matrix(root_quat)
        R_BI = R_IB.T
        
        lin_vel_b = R_BI @ root_vel[:3]
        ang_vel_b = R_BI @ root_vel[3:]
        
        # CRITICAL FIX 1: Gravity
        # In IsaacLab mdp.projected_gravity returns 3 dims (x,y,z).
        # Your previous code sliced it to [:2], causing a frame shift.
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0]) 
        
        # CRITICAL FIX 2: Joint Order
        # Get all joints from sim
        raw_joint_pos = self.articulation_view.get_joint_positions()[0]
        raw_joint_vel = self.articulation_view.get_joint_velocities()[0]
        raw_joint_effort = self.articulation_view.get_measured_joint_efforts()[0] # If simulating effort
        
        # Reorder them to match POLICY order
        policy_joint_pos = raw_joint_pos[self.joint_indices]
        policy_joint_vel = raw_joint_vel[self.joint_indices]
        policy_joint_effort = raw_joint_effort[self.joint_indices]

        # Build observation vector 
        # Training Config: 
        # base_lin (3) + base_ang (3) + gravity (3) + cmd (3) + 
        # pos (12) + vel (12) + effort (12) + actions (12) = 60 Dims
        obs = np.concatenate([
            lin_vel_b,                    # 3
            ang_vel_b,                    # 3
            gravity_b,                    # 3 (WAS WRONG IN PREVIOUS CODE)
            command,                      # 3
            policy_joint_pos - self.default_pos, # 12
            policy_joint_vel,             # 12
            policy_joint_effort,                 # 12
            self._previous_action         # 12
        ])
        
        # Ensure float32 for PyTorch
        return obs.astype(np.float32)
    
    def forward(self, dt, command):
        """Step the policy"""
        if self.articulation_view is None:
            return
            
        if self._policy_counter % self._decimation == 0:
            obs = self.get_observations(command)
            
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # If your model expects CUDA, move tensor: obs_tensor.cuda()
                action = self.policy(obs_tensor).squeeze(0).numpy()
            
            self.action = action
            self._previous_action = action.copy()
        
        # Apply actions
        # The policy outputs actions in POLICY order (LF, RF...)
        # We must map these back to SIMULATION order to apply them
        
        sim_ordered_targets = np.zeros(12)
        
        # Calculate targets in Policy Order first
        policy_targets = self.default_pos + (self.action * self.action_scale)
        
        # Map to Sim Order
        for policy_i, sim_i in enumerate(self.joint_indices):
            sim_ordered_targets[sim_i] = policy_targets[policy_i]

        self.articulation_view.set_joint_position_targets(
            sim_ordered_targets,
            joint_indices=list(range(12)) # Apply to all 12 sim joints
        )
        
        self._policy_counter += 1
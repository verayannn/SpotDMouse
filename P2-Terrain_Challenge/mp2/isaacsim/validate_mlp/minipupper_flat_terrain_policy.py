import torch
import numpy as np
from typing import Optional
from isaacsim.core.api import World
from isaacsim.core.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.articulations import ArticulationView
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.rotations import quat_to_rot_matrix

# Import the actual robot config from your training
import sys
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
        
        # Load the USD from your training config
        robot_cfg = CUSTOM_QUAD_CFG.scene.robot
        usd_path = robot_cfg.spawn.usd_path
        
        # Add robot to stage
        add_reference_to_stage(usd_path, prim_path)
        
        # Position the robot
        if position is not None:
            prim = get_prim_at_path(prim_path)
            prim.GetAttribute("xformOp:translate").Set(position.tolist())
        
        # Load the JIT policy
        self.policy = torch.jit.load(
            "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/exported/policy.pt"
        )
        self.policy.eval()
        
        # Get action config from training
        self.joint_names = [
            "base_lf1", "lf1_lf2", "lf2_lf3",
            "base_rf1", "rf1_rf2", "rf2_rf3",
            "base_lb1", "lb1_lb2", "lb2_lb3",
            "base_rb1", "rb1_rb2", "rb2_rb3"
        ]
        self.action_scale = 0.5  # From your SpotActionsCfg
        
        # Default joint positions (from your training config)
        self.default_pos = np.array([
            0.0, 0.785, -1.57,  # LF
            0.0, 0.785, -1.57,  # RF
            0.0, 0.785, -1.57,  # LB
            0.0, 0.785, -1.57,  # RB
        ])
        
        # Policy state
        self._decimation = 10  # From your env config
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        
        # Will be set after world.reset()
        self.articulation_view = None
        
    def initialize(self, world: World):
        """Initialize after world reset"""
        # Create articulation view
        self.articulation_view = ArticulationView(
            prim_paths_expr=self.prim_path,
            name=self.name + "_view"
        )
        world.scene.add(self.articulation_view)
        
        # Set initial joint positions
        self.articulation_view.set_joint_positions(
            self.default_pos,
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
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])
        
        # Get joint states
        joint_pos = self.articulation_view.get_joint_positions()[0]
        joint_vel = self.articulation_view.get_joint_velocities()[0]
        joint_effort = self.articulation_view.get_measured_joint_efforts()[0]
        
        # Build observation vector (59 dims total)
        obs = np.concatenate([
            lin_vel_b,                    # 3
            ang_vel_b,                    # 3
            gravity_b[:2],                # 2
            command,                      # 3
            joint_pos - self.default_pos, # 12
            joint_vel,                    # 12
            joint_effort,                 # 12
            self._previous_action         # 12
        ])
        
        return obs
    
    def forward(self, dt, command):
        """Step the policy"""
        if self.articulation_view is None:
            return
            
        if self._policy_counter % self._decimation == 0:
            # Get observations
            obs = self.get_observations(command)
            
            # Run policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                action = self.policy(obs_tensor).squeeze(0).numpy()
            
            self.action = action
            self._previous_action = action.copy()
        
        # Apply actions
        target_positions = self.default_pos + (self.action * self.action_scale)
        self.articulation_view.set_joint_position_targets(
            target_positions,
            joint_indices=list(range(12))
        )
        
        self._policy_counter += 1
from typing import Optional
import numpy as np
import omni
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path


class MiniPupperFlatTerrainPolicy(PolicyController):
    """The Mini Pupper quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "minipupper",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        # Use the Mini Pupper USD file
        if usd_path is None:
            usd_path = "/workspace/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper_2/mini_pupper_description/mini_pupper_description.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        # Load your trained policy with the existing YAML file
        self.load_policy(
            "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/model_19999.pt",
            "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/params/env.yaml",
        )
        
        # From the YAML file
        self._action_scale = 0.5  # Matches the action scale in YAML
        self._decimation = 10     # From the YAML decimation setting
        self._previous_action = np.zeros(12)  # 12 joints for Mini Pupper
        self._policy_counter = 0
        
        # Joint names in order (matching your training config)
        self._joint_names = [
            "base_lf1", "lf1_lf2", "lf2_lf3",
            "base_rf1", "rf1_rf2", "rf2_rf3",
            "base_lb1", "lb1_lb2", "lb2_lb3",
            "base_rb1", "rb1_rb2", "rb2_rb3"
        ]

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy
        Matching your SpotObservationsCfg exactly

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.
        """
        # Get robot state
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        # Transform to body frame
        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # Build observation vector matching your policy
        obs_list = []
        
        # Base linear velocity (3 dims)
        obs_list.append(lin_vel_b)
        
        # Base angular velocity (3 dims)
        obs_list.append(ang_vel_b)
        
        # Projected gravity (only x,y components = 2 dims)
        obs_list.append(gravity_b[:2])
        
        # Velocity commands (3 dims)
        obs_list.append(command)
        
        # Joint positions relative to default (12 dims)
        current_joint_pos = self.robot.get_joint_positions()
        obs_list.append(current_joint_pos - self.default_pos)
        
        # Joint velocities (12 dims)
        current_joint_vel = self.robot.get_joint_velocities()
        obs_list.append(current_joint_vel)
        
        # Joint efforts/torques (12 dims)
        current_joint_effort = self.robot.get_measured_joint_efforts()
        obs_list.append(current_joint_effort)
        
        # Previous actions (12 dims)
        obs_list.append(self._previous_action)
        
        # Concatenate all observations
        obs = np.concatenate(obs_list)
        
        return obs

    def forward(self, dt, command):
        """
        Compute the desired joint positions and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)
        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # Apply position control
        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1
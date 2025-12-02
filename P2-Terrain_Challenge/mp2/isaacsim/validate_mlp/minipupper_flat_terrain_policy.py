from typing import Optional
import numpy as np
import omni
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path

# Import the Isaac Lab configurations
from isaaclab_tasks.manager_based.locomotion.velocity.config.custom_quadruped_2.custom_quad import CUSTOM_QUAD_CFG

# Try to import the config, but handle if it's not accessible
try:
    from isaaclab_tasks.manager_based.locomotion.velocity.config.custom_quadruped_2.flat_env_cfg import SpotActionsCfg, SpotObservationsCfg
    # Create an instance to access the attributes
    spot_actions_cfg = SpotActionsCfg()
    ACTION_SCALE = spot_actions_cfg.joint_pos.scale if hasattr(spot_actions_cfg, 'joint_pos') else 0.5
except Exception as e:
    print(f"Warning: Could not import action scale from config: {e}")
    ACTION_SCALE = 0.5  # Use the known value directly

class MPFlatTerrainPolicy(PolicyController):
    """The MP2 quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "mp2",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.
        """
        assets_root_path = get_assets_root_path()
        
        # Use USD path from CUSTOM_QUAD_CFG
        if usd_path is None:
            usd_path = CUSTOM_QUAD_CFG.spawn.usd_path

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        self.load_policy(
            "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/exported/policy.pt",
            "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/params/env.yaml",
        )
        
        # Use the action scale we know from the config
        self._action_scale = ACTION_SCALE  # 0.5 from SpotActionsCfg
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        
        # Extract joint names and default positions from CUSTOM_QUAD_CFG
        self.joint_names = CUSTOM_QUAD_CFG.actuators["leg_joints"].joint_names_expr
        self.default_joint_positions = CUSTOM_QUAD_CFG.init_state.joint_pos
        
        # Store observation noise ranges from SpotObservationsCfg for potential use
        self.obs_noise_ranges = {
            "base_lin_vel": (-0.1, 0.1),
            "base_ang_vel": (-0.2, 0.2),
            "projected_gravity": (-0.05, 0.05),
            "joint_pos": (-0.01, 0.01),
            "joint_vel": (-1.5, 1.5),
            "joint_effort": (-0.1, 0.1)
        }

    def _set_articulation_props(self):
        """Override to skip the problematic articulation property setting.
        
        Since the robot is already properly configured in the USD file from Isaac Lab,
        we don't need to modify articulation properties.
        """
        pass  # Do nothing - the robot is already properly configured

    def initialize(self):
        """Initialize the robot without modifying articulation properties"""
        # Initialize the robot
        self.robot.initialize()
        
        # Set default positions from CUSTOM_QUAD_CFG
        if hasattr(self, 'default_joint_positions'):
            # Convert dict to array in correct joint order
            self.default_pos = np.array([
                self.default_joint_positions[joint] for joint in self.joint_names
            ])
        else:
            # Fallback to current positions
            self.default_pos = self.robot.get_joint_positions()
        
        self._initialized = True

    def _apply_observation_noise(self, value, noise_range):
        """Apply uniform noise to observation values (optional - for testing robustness)"""
        if hasattr(self, 'apply_noise') and self.apply_noise:
            noise = np.random.uniform(noise_range[0], noise_range[1], value.shape)
            return value + noise
        return value

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy
        
        This matches the exact order and structure from SpotObservationsCfg:
        1. Base linear velocity (3 dims)
        2. Base angular velocity (3 dims) 
        3. Projected gravity (3 dims)
        4. Velocity commands (3 dims)
        5. Joint positions relative to default (12 dims)
        6. Joint velocities (12 dims)
        7. Joint efforts (12 dims)
        8. Previous actions (12 dims)
        
        Total: 60 dimensions
        """
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(60)
        
        # Base lin vel (3 dims) - matches mdp.base_lin_vel
        obs[0:3] = self._apply_observation_noise(lin_vel_b, self.obs_noise_ranges["base_lin_vel"])
        
        # Base ang vel (3 dims) - matches mdp.base_ang_vel
        obs[3:6] = self._apply_observation_noise(ang_vel_b, self.obs_noise_ranges["base_ang_vel"])
        
        # Projected gravity (3 dims) - matches mdp.projected_gravity
        obs[6:9] = self._apply_observation_noise(gravity_b, self.obs_noise_ranges["projected_gravity"])
        
        # Velocity commands (3 dims) - matches mdp.generated_commands
        obs[9:12] = command
        
        # Joint positions relative to default (12 dims) - matches mdp.joint_pos_rel
        current_joint_pos = self.robot.get_joint_positions()
        joint_pos_rel = current_joint_pos - self.default_pos
        obs[12:24] = self._apply_observation_noise(joint_pos_rel, self.obs_noise_ranges["joint_pos"])
        
        # Joint velocities (12 dims) - matches mdp.joint_vel_rel
        current_joint_vel = self.robot.get_joint_velocities()
        obs[24:36] = self._apply_observation_noise(current_joint_vel, self.obs_noise_ranges["joint_vel"])
        
        # Joint efforts/torques (12 dims) - matches mdp.joint_effort
        try:
            # The training uses applied torques which represents servo loads
            if hasattr(self.robot, 'get_applied_joint_efforts'):
                current_joint_efforts = self.robot.get_applied_joint_efforts()
            elif hasattr(self.robot, 'get_joint_efforts'):
                current_joint_efforts = self.robot.get_joint_efforts()
            elif hasattr(self.robot, 'get_measured_joint_efforts'):
                current_joint_efforts = self.robot.get_measured_joint_efforts()
            else:
                # If no method available, estimate from current error and gains
                position_error = self._previous_action * self._action_scale  # commanded position delta
                current_joint_efforts = position_error * CUSTOM_QUAD_CFG.actuators["leg_joints"].stiffness
        except Exception as e:
            print(f"Warning: Could not get joint efforts: {e}")
            current_joint_efforts = np.zeros(12)
            
        obs[36:48] = self._apply_observation_noise(current_joint_efforts, self.obs_noise_ranges["joint_effort"])
        
        # Previous actions (12 dims) - matches mdp.last_action
        obs[48:60] = self._previous_action

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation
        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        # Apply action with the same scaling as training
        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1
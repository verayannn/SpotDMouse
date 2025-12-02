import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Direct MLP control of Mini Pupper")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--policy_path", type=str, 
                   default="/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/harvardrun_45/2025-11-18_00-57-12/exported/policy.pt",
                   help="Path to the trained policy")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import DCMotorCfg

# Define CUSTOM_QUAD_CFG directly to avoid the import issue
CUSTOM_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper_2/mini_pupper_description/mini_pupper_description.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),
        joint_pos={
            "base_lf1": 0.0,      
            "lf1_lf2": 0.785,
            "lf2_lf3": -1.57,
            "base_rf1": 0.0,      
            "rf1_rf2": 0.785,
            "rf2_rf3": -1.57,
            "base_lb1": 0.0,      
            "lb1_lb2": 0.785,
            "lb2_lb3": -1.57,
            "base_rb1": 0.0,      
            "rb1_rb2": 0.785,            
            "rb2_rb3": -1.57,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "leg_joints": DCMotorCfg(
            joint_names_expr=[
                "base_lf1", "lf1_lf2", "lf2_lf3",  
                "base_rf1", "rf1_rf2", "rf2_rf3",
                "base_lb1", "lb1_lb2", "lb2_lb3",
                "base_rb1", "rb1_rb2", "rb2_rb3"
            ],
            saturation_effort=0.35,
            velocity_limit=10.5,
            stiffness=80.0,
            damping=2.5,
            friction=0.03,        
            armature=0.005,
        ),
    }
)

class DirectMLPController:
    def __init__(self, policy_path, device='cuda'):
        """Initialize the MLP controller with just the policy"""
        self.device = device
        
        # Load the policy
        self.policy = torch.jit.load(policy_path).to(device)
        self.policy.eval()
        
        # Action scale from your training config
        self.action_scale = 0.5
        
        # Previous action for observation
        self.previous_action = torch.zeros(12, device=device)
        
        # Joint names in the correct order
        self.joint_names = [
            "base_lf1", "lf1_lf2", "lf2_lf3",
            "base_rf1", "rf1_rf2", "rf2_rf3",
            "base_lb1", "lb1_lb2", "lb2_lb3",
            "base_rb1", "rb1_rb2", "rb2_rb3"
        ]
        
        # Default joint positions from CUSTOM_QUAD_CFG
        self.default_positions = torch.tensor([
            0.0, 0.785, -1.57,    # LF
            0.0, 0.785, -1.57,    # RF
            0.0, 0.785, -1.57,    # LB
            0.0, 0.785, -1.57,    # RB
        ], device=device)
    
    def compute_observation(self, robot_data, command):
        """Build observation vector matching your training setup"""
        # Get base velocities in body frame
        lin_vel_world = robot_data.root_vel_w[0, :3]
        ang_vel_world = robot_data.root_vel_w[0, 3:6]
        quat = robot_data.root_quat_w[0]
        
        # Convert quaternion to rotation matrix
        # Simple conversion for body frame velocities
        rot_matrix = self._quat_to_rot_matrix(quat)
        lin_vel_body = torch.matmul(rot_matrix.T, lin_vel_world)
        ang_vel_body = torch.matmul(rot_matrix.T, ang_vel_world)
        
        # Projected gravity
        gravity_world = torch.tensor([0., 0., -1.], device=self.device)
        gravity_body = torch.matmul(rot_matrix.T, gravity_world)
        
        # Get joint data (only the 12 actuated joints)
        joint_positions = robot_data.joint_pos[0, :12]
        joint_velocities = robot_data.joint_vel[0, :12]
        
        # Build observation vector (60 dims total)
        obs = torch.zeros(60, device=self.device)
        obs[0:3] = lin_vel_body          # base linear velocity (3)
        obs[3:6] = ang_vel_body          # base angular velocity (3)
        obs[6:9] = gravity_body          # projected gravity (3)
        obs[9:12] = command              # velocity commands (3)
        obs[12:24] = joint_positions - self.default_positions  # joint pos relative (12)
        obs[24:36] = joint_velocities    # joint velocities (12)
        obs[36:48] = torch.zeros(12)     # joint efforts (12) - zeros for now
        obs[48:60] = self.previous_action # previous actions (12)
        
        return obs
    
    def _quat_to_rot_matrix(self, quat):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        R = torch.zeros((3, 3), device=self.device)
        R[0, 0] = 1 - 2*(y**2 + z**2)
        R[0, 1] = 2*(x*y - w*z)
        R[0, 2] = 2*(x*z + w*y)
        R[1, 0] = 2*(x*y + w*z)
        R[1, 1] = 1 - 2*(x**2 + z**2)
        R[1, 2] = 2*(y*z - w*x)
        R[2, 0] = 2*(x*z - w*y)
        R[2, 1] = 2*(y*z + w*x)
        R[2, 2] = 1 - 2*(x**2 + y**2)
        
        return R
    
    def get_actions(self, robot_data, command):
        """Get actions from the policy"""
        obs = self.compute_observation(robot_data, command)
        
        # Run policy
        with torch.no_grad():
            actions = self.policy(obs.unsqueeze(0))[0]
        
        # Store for next observation
        self.previous_action = actions.clone()
        
        # Scale actions and add to default positions
        target_positions = self.default_positions + actions * self.action_scale
        
        return target_positions


class MLPControlSceneCfg(InteractiveSceneCfg):
    """Simple scene with just the robot and ground"""
    
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # Robot - use CUSTOM_QUAD_CFG
    robot = CUSTOM_QUAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_mlp_control(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the MLP controller directly"""
    print("[INFO]: Initializing direct MLP control...")
    
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Initialize controller
    controller = DirectMLPController(args_cli.policy_path, device=sim.device)
    
    # Command (you can make this dynamic later)
    command = torch.tensor([0.2, 0.0, 0.0], device=sim.device)  # Forward at 0.2 m/s
    
    print(f"[INFO]: MLP Controller initialized")
    print(f"[INFO]: Command: vx={command[0]:.2f}, vy={command[1]:.2f}, wz={command[2]:.2f}")
    print(f"[INFO]: Policy decimation: every 4 steps (assuming 250Hz sim, 62.5Hz policy)")
    
    # Reset robot once at start
    root_state = scene["robot"].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    scene["robot"].write_root_pose_to_sim(root_state[:, :7])
    scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
    
    # Set initial joint positions
    full_joint_positions = torch.zeros(len(scene["robot"].data.joint_names), device=sim.device)
    full_joint_positions[:12] = controller.default_positions
    scene["robot"].write_joint_state_to_sim(full_joint_positions, torch.zeros_like(full_joint_positions))
    
    scene.reset()
    
    while simulation_app.is_running():
        # Get actions from policy every 4 steps (decimation)
        if count % 4 == 0:
            target_positions = controller.get_actions(scene["robot"].data, command)
            
            # Build full joint target (including fixed joints)
            full_target = torch.zeros(len(scene["robot"].data.joint_names), device=sim.device)
            full_target[:12] = target_positions
            
            # Debug every 100 policy steps
            if count % 400 == 0:
                print(f"\n[Step {count}]")
                print(f"Root pos: {scene['robot'].data.root_pos_w[0].cpu().numpy()}")
                print(f"Root vel: {scene['robot'].data.root_vel_w[0, :3].cpu().numpy()}")
                print(f"Target joints (first 3): {target_positions[:3].cpu().numpy()}")
                print(f"Actual joints (first 3): {scene['robot'].data.joint_pos[0, :3].cpu().numpy()}")
        
        # Apply joint targets
        scene["robot"].set_joint_position_target(full_target)
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function"""
    # Simulation config
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=0.004,  # 250Hz
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS solver
            enable_stabilization=True,
            bounce_threshold_velocity=0.05,
            friction_offset_threshold=0.02,
            friction_correlation_distance=0.01,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set camera
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.3])
    
    # Create scene
    scene_cfg = MLPControlSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset and run
    sim.reset()
    print("[INFO]: Setup complete! Starting MLP control...")
    
    run_mlp_control(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
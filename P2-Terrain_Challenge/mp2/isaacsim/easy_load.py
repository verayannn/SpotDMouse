import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import os
from math import pi

import textwrap
import re

#tutorial here: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html

# ACCURATE for real MiniPupper: 560g total weight
cfg_robot = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper_2/mini_pupper_description/mini_pupper_description.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
        # In your init_state, try slightly straighter legs:
        joint_pos={
            #Legs @ 45°
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
    # Main leg joints (the ones that do the real work)
    "leg_joints": DCMotorCfg(
        joint_names_expr=[
            # LF leg (front-left)
            "base_lf1", "lf1_lf2", "lf2_lf3",
            # RF leg (front-right)  
            "base_rf1", "rf1_rf2", "rf2_rf3",
            # LB leg (back-left)
            "base_lb1", "lb1_lb2", "lb2_lb3",
            # RB leg (back-right)
            "base_rb1", "rb1_rb2", "rb2_rb3"
            ],
        # saturation_effort=2.5,
        # velocity_limit=10.0,
        # stiffness=45.0,#80.0 Official/Final: 45.0       
        # damping=1.3,#2.0 Official/Final: 1.3        
        # friction=0.02,        
        # armature=0.005,#0.004269, # Sweet spot - jitters in place, no drift
        saturation_effort=0.35, #3.5kg
        velocity_limit=10.5,
        stiffness=80.0,#80.0 Official/Final: 45.0       
        damping=2.5,#2.0 Official/Final: 1.3        
        friction=0.03,        
        armature=0.005,#0.004269, # Sweet spot - jitters in place, no drift
    ),
    }
)

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # robot
    robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Spot")

# WITH MOVEMENT!
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Create a dictionary for easier lookup (if you prefer, otherwise list directly in order)
    desired_joint_angles_dict = {
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
        # Fixed-but-revolute joints (usually 0.0 unless they have a non-zero default angle in your URDF/USD)
        "lf3_foot": 0.0, "rf3_foot": 0.0, "lb3_foot": 0.0, "rb3_foot": 0.0,
        "lf1_plate": 0.0, "rf1_plate": 0.0, "lb1_plate": 0.0, "rb1_plate": 0.0,
        "lf2_plate": 0.0, "rf2_plate": 0.0, "lb2_plate": 0.0, "rb2_plate": 0.0,
        "base_lidar": 0.0, "imu_joint": 0.0
    }

    # Construct the target tensor in the correct joint order from scene["robot"].data.joint_names
    target_standing_pos_tensor = torch.zeros(len(scene["robot"].data.joint_names), device=sim.device, dtype=torch.float)
    for i, joint_name in enumerate(scene["robot"].data.joint_names):
        if joint_name in desired_joint_angles_dict:
            target_standing_pos_tensor[i] = desired_joint_angles_dict[joint_name]
        # Else, it's already 0.0 from initialization of the tensor

    # Cache this as the standing joint position target
    default_joint_pos = target_standing_pos_tensor.clone() # <<< THIS IS THE KEY CHANGE

    # --- END OF MODIFIED SECTION ---

    # Add test parameters for calf movement
    test_amplitude = 0.5  # Start with 0.5 radians (~28 degrees)
    test_frequency = 0.5  # 0.5 Hz = 2 seconds per cycle
    lf2_lf3_index = scene["robot"].data.joint_names.index("lf2_lf3")
    
    print(f"[INFO] Testing left front calf (lf2_lf3) at index {lf2_lf3_index}")
    print(f"[INFO] Test amplitude: {test_amplitude} rad, frequency: {test_frequency} Hz")

    # Add small settling period after spawn
    # settling_steps = 100
    
    print(f"[DEBUG] Sim Dt Value:{sim_dt}")
    # Now, this debug line should show your desired bent-leg targets!
    print(f"[DEBUG] Joint targets: {default_joint_pos}")
    print(f"[DEBUG] Actual joints: {scene['robot'].data.joint_pos[0]}")
    print(f"[DEBUG] Joint errors: {default_joint_pos - scene['robot'].data.joint_pos[0]}")
    # print("Leg order:", cfg_robot.actuators["leg_joints"].joint_names_expr)

    
    while simulation_app.is_running():
        # Periodic reset
        if count % 1000 == 0:  # Increased reset interval
            count = 0
            print("[INFO]: Resetting Mini Pupper state...")
            
            # Reset root pose and velocity
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            # Add small random perturbation to test stability (keep this for testing robustness)
            root_state[:, 2] += torch.randn_like(root_state[:, 2]) * 0.01
            
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            
            # Reset joints with the correct target pose
            # joint_pos_reset = default_joint_pos.clone() # This is fine now as default_joint_pos is correct
            scene["robot"].write_joint_state_to_sim(default_joint_pos, scene["robot"].data.default_joint_vel.clone())
            
            # Clear internal buffers
            scene.reset()
            # check_joint_order(scene, cfg_robot)
            # check_groups(scene, cfg_robot)
            print("Leg order:", cfg_robot.actuators["leg_joints"].joint_names_expr)

        # Create test target with sine wave on left front calf
        test_joint_pos = default_joint_pos.clone()
        
        # After initial settling (e.g., after 200 steps), apply sine wave to calf
        if count > 200:
            # Sine wave motion on left front calf
            sine_offset = test_amplitude * torch.sin(torch.tensor(2 * np.pi * test_frequency * sim_time, device=sim.device))
            test_joint_pos[lf2_lf3_index] = default_joint_pos[lf2_lf3_index] + sine_offset
            
            # Print debug info every 50 steps
            if count % 50 == 0:
                actual_pos = scene["robot"].data.joint_pos[0, lf2_lf3_index]
                target_pos = test_joint_pos[lf2_lf3_index]
                error = target_pos - actual_pos
                print(f"[TEST] Step {count}: LF Calf Target: {target_pos:.3f}, Actual: {actual_pos:.3f}, Error: {error:.3f}")
                
        scene["robot"].set_joint_position_target(test_joint_pos)
        
        # Step sim
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        
        # Debug output every 100 steps
        if count % 100 == 0:
            root_pos = scene["robot"].data.root_pos_w[0]
            root_quat = scene["robot"].data.root_quat_w[0]
            print(f"[DEBUG] Step {count}: Root pos: {root_pos}, Root quat: {root_quat}")
# ...existing code...

# Standing-only, no test movement!

# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     # Create a dictionary for easier lookup (if you prefer, otherwise list directly in order)
#     desired_joint_angles_dict = {
#         "base_lf1": 0.0,
#         "lf1_lf2": 0.52,
#         "lf2_lf3": -1.05,
#         "base_rf1": 0.0,
#         "rf1_rf2": 0.52,
#         "rf2_rf3": -1.05,
#         "base_lb1": 0.0,
#         "lb1_lb2": 0.52,
#         "lb2_lb3": -1.05,
#         "base_rb1": 0.0,
#         "rb1_rb2": 0.52,
#         "rb2_rb3": -1.05,
#         # Fixed-but-revolute joints (usually 0.0 unless they have a non-zero default angle in your URDF/USD)
#         "lf3_foot": 0.0, "rf3_foot": 0.0, "lb3_foot": 0.0, "rb3_foot": 0.0,
#         "lf1_plate": 0.0, "rf1_plate": 0.0, "lb1_plate": 0.0, "rb1_plate": 0.0,
#         "lf2_plate": 0.0, "rf2_plate": 0.0, "lb2_plate": 0.0, "rb2_plate": 0.0,
#         "base_lidar": 0.0, "imu_joint": 0.0
#     }

#     # Construct the target tensor in the correct joint order from scene["robot"].data.joint_names
#     target_standing_pos_tensor = torch.zeros(len(scene["robot"].data.joint_names), device=sim.device, dtype=torch.float)
#     for i, joint_name in enumerate(scene["robot"].data.joint_names):
#         if joint_name in desired_joint_angles_dict:
#             target_standing_pos_tensor[i] = desired_joint_angles_dict[joint_name]
#         # Else, it's already 0.0 from initialization of the tensor

#     # Cache this as the standing joint position target
#     default_joint_pos = target_standing_pos_tensor.clone() # <<< THIS IS THE KEY CHANGE

#     # --- END OF MODIFIED SECTION ---

#     # Add small settling period after spawn
#     # settling_steps = 100
    
#     print(f"[DEBUG] Sim Dt Value:{sim_dt}")
#     # Now, this debug line should show your desired bent-leg targets!
#     print(f"[DEBUG] Joint targets: {default_joint_pos}")
#     print(f"[DEBUG] Actual joints: {scene['robot'].data.joint_pos[0]}")
#     print(f"[DEBUG] Joint errors: {default_joint_pos - scene['robot'].data.joint_pos[0]}")
#     # print("Leg order:", cfg_robot.actuators["leg_joints"].joint_names_expr)

    
#     while simulation_app.is_running():
#         # Periodic reset
#         if count % 1000 == 0:  # Increased reset interval
#             count = 0
#             print("[INFO]: Resetting Mini Pupper state...")
            
#             # Reset root pose and velocity
#             root_state = scene["robot"].data.default_root_state.clone()
#             root_state[:, :3] += scene.env_origins
#             # Add small random perturbation to test stability (keep this for testing robustness)
#             root_state[:, 2] += torch.randn_like(root_state[:, 2]) * 0.01
            
#             scene["robot"].write_root_pose_to_sim(root_state[:, :7])
#             scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            
#             # Reset joints with the correct target pose
#             # joint_pos_reset = default_joint_pos.clone() # This is fine now as default_joint_pos is correct
#             scene["robot"].write_joint_state_to_sim(default_joint_pos, scene["robot"].data.default_joint_vel.clone())
            
#             # Clear internal buffers
#             scene.reset()
#             # check_joint_order(scene, cfg_robot)
#             # check_groups(scene, cfg_robot)
#             print("Leg order:", cfg_robot.actuators["leg_joints"].joint_names_expr)

#         scene["robot"].set_joint_position_target(default_joint_pos)
        
#         # Step sim
#         scene.write_data_to_sim()
#         sim.step()
#         sim_time += sim_dt
#         count += 1
#         scene.update(sim_dt)
        
#         # Debug output every 100 steps
#         if count % 100 == 0:
#             root_pos = scene["robot"].data.root_pos_w[0]
#             root_quat = scene["robot"].data.root_quat_w[0]
#             print(f"[DEBUG] Step {count}: Root pos: {root_pos}, Root quat: {root_quat}")

def main():
    """Main function."""
    # Simulation tuned for 2 lb robot with carbon fiber legs
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=0.004,  # 250Hz - good balance for lightweight robot
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS solver
            enable_stabilization=True,
            bounce_threshold_velocity=0.05,  # Appropriate for lightweight
            friction_offset_threshold=0.02,
            friction_correlation_distance=0.01,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.3])
    
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Robot should spawn and settle into stable standing position")
    
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()


#'Harvard 30 Params
# ACCURATE for real MiniPupper: 560g total weight
# cfg_robot = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/workspace/ros2_ws/src/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper/mini_pupper_description/mini_pupper_description.usd",
#         activate_contact_sensors=True,
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
#         # In your init_state, try slightly straighter legs:
#         joint_pos={
#             # Less bent legs (30° instead of 45°)
#             "base_lf1": 0.0,      
#             "lf1_lf2": 0.52,      # ~30° (π/6 radians)
#             "lf2_lf3": -1.05,     # ~60° to make foot flat
            
#             "base_rf1": 0.0,      
#             "rf1_rf2": 0.52,      
#             "rf2_rf3": -1.05,     
            
#             "base_lb1": 0.0,      
#             "lb1_lb2": 0.52,      
#             "lb2_lb3": -1.05,     
            
#             "base_rb1": 0.0,      
#             "rb1_rb2": 0.52,      
#             "rb2_rb3": -1.05,     
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.95,
#     actuators={
#     # Main leg joints (the ones that do the real work)
#     "leg_joints": DCMotorCfg(
#         joint_names_expr=[
#             # LF leg (front-left)
#             "base_lf1", "lf1_lf2", "lf2_lf3",
#             # RF leg (front-right)  
#             "base_rf1", "rf1_rf2", "rf2_rf3",
#             # LB leg (back-left)
#             "base_lb1", "lb1_lb2", "lb2_lb3",
#             # RB leg (back-right)
#             "base_rb1", "rb1_rb2", "rb2_rb3"
#             ],
#         saturation_effort=2.5,
#         velocity_limit=10.0,
#         stiffness=20.0,#80.0        
#         damping=0.8,#7.0          
#         friction=0.02,        
#         armature=0.005,#0.004269, # Sweet spot - jitters in place, no drift
#     ),
#     }
# )

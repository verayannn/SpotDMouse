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
# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     # Create a dictionary for easier lookup (if you prefer, otherwise list directly in order)
#     desired_joint_angles_dict = {
#         "base_lf1": 0.0,
#         "lf1_lf2": 0.785,
#         "lf2_lf3": -1.57,
#         "base_rf1": 0.0,
#         "rf1_rf2": 0.785,
#         "rf2_rf3": -1.57,
#         "base_lb1": 0.0,
#         "lb1_lb2": 0.785,
#         "lb2_lb3": -1.57,
#         "base_rb1": 0.0,
#         "rb1_rb2": 0.785,
#         "rb2_rb3": -1.57,
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

#     # Add test parameters for calf movement
#     test_amplitude = 0.5  # Start with 0.5 radians (~28 degrees)
#     test_frequency = 0.5  # 0.5 Hz = 2 seconds per cycle
#     lf2_lf3_index = scene["robot"].data.joint_names.index("lf2_lf3")
    
#     print(f"[INFO] Testing left front calf (lf2_lf3) at index {lf2_lf3_index}")
#     print(f"[INFO] Test amplitude: {test_amplitude} rad, frequency: {test_frequency} Hz")

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

#         # Create test target with sine wave on left front calf
#         test_joint_pos = default_joint_pos.clone()
        
#         # After initial settling (e.g., after 200 steps), apply sine wave to calf
#         if count > 200:
#             # Sine wave motion on left front calf
#             sine_offset = test_amplitude * torch.sin(torch.tensor(2 * np.pi * test_frequency * sim_time, device=sim.device))
#             test_joint_pos[lf2_lf3_index] = default_joint_pos[lf2_lf3_index] + sine_offset
            
#             # Print debug info every 50 steps
#             if count % 50 == 0:
#                 actual_pos = scene["robot"].data.joint_pos[0, lf2_lf3_index]
#                 target_pos = test_joint_pos[lf2_lf3_index]
#                 error = target_pos - actual_pos
#                 print(f"[TEST] Step {count}: LF Calf Target: {target_pos:.3f}, Actual: {actual_pos:.3f}, Error: {error:.3f}")
                
#         scene["robot"].set_joint_position_target(test_joint_pos)
        
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
# ...existing code...


#multilimb reference and direction test
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Create a dictionary for easier lookup
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
        # Fixed joints
        "lf3_foot": 0.0, "rf3_foot": 0.0, "lb3_foot": 0.0, "rb3_foot": 0.0,
        "lf1_plate": 0.0, "rf1_plate": 0.0, "lb1_plate": 0.0, "rb1_plate": 0.0,
        "lf2_plate": 0.0, "rf2_plate": 0.0, "lb2_plate": 0.0, "rb2_plate": 0.0,
        "base_lidar": 0.0, "imu_joint": 0.0
    }

    # Construct the target tensor
    target_standing_pos_tensor = torch.zeros(len(scene["robot"].data.joint_names), device=sim.device, dtype=torch.float)
    for i, joint_name in enumerate(scene["robot"].data.joint_names):
        if joint_name in desired_joint_angles_dict:
            target_standing_pos_tensor[i] = desired_joint_angles_dict[joint_name]

    default_joint_pos = target_standing_pos_tensor.clone()

    # Define the 12 actuated joints to test with cleaner names
    test_joints = [
        # Front Left
        ("base_lf1", "LF Hip (abduction/adduction)"),
        ("lf1_lf2", "LF Thigh (flexion/extension)"), 
        ("lf2_lf3", "LF Calf (flexion/extension)"),
        # Front Right
        ("base_rf1", "RF Hip (abduction/adduction)"),
        ("rf1_rf2", "RF Thigh (flexion/extension)"),
        ("rf2_rf3", "RF Calf (flexion/extension)"),
        # Back Left
        ("base_lb1", "LB Hip (abduction/adduction)"),
        ("lb1_lb2", "LB Thigh (flexion/extension)"),
        ("lb2_lb3", "LB Calf (flexion/extension)"),
        # Back Right
        ("base_rb1", "RB Hip (abduction/adduction)"),
        ("rb1_rb2", "RB Thigh (flexion/extension)"),
        ("rb2_rb3", "RB Calf (flexion/extension)")
    ]
    
    # Test parameters
    test_amplitude = 0.3  # 0.3 radians to match physical robot test
    test_duration = 2.0  # seconds to hold each position
    settling_time = 1.0  # seconds between tests
    
    # Convert to steps
    test_steps = int(test_duration / sim_dt)
    settle_steps = int(settling_time / sim_dt)
    
    # Results storage
    test_results = []
    current_joint_idx = 0
    test_phase = "settle"  # "settle", "positive", "negative", "return"
    phase_counter = 0
    
    print("\n" + "="*60)
    print("ISAAC SIM JOINT DIRECTION TEST")
    print("="*60)
    print(f"Test amplitude: ±{test_amplitude} radians (~{test_amplitude * 180/np.pi:.1f}°)")
    print(f"Hold duration: {test_duration}s per direction")
    print("\nExpected conventions:")
    print("- Hip: Positive = abduction (outward), Negative = adduction (inward)")
    print("- Thigh: Positive = flexion (forward), Negative = extension (backward)")
    print("- Calf: Positive = extension (straighten), Negative = flexion (bend)")
    print("="*60 + "\n")

    while simulation_app.is_running() and current_joint_idx < len(test_joints):
        # Reset robot if starting new joint test
        if test_phase == "settle" and phase_counter == 0:
            if current_joint_idx == 0:
                print("Starting joint direction tests...")
            else:
                # Print result from previous joint
                joint_name, desc = test_joints[current_joint_idx - 1]
                result = test_results[-1]
                print(f"\nRESULT for {joint_name}:")
                print(f"  Positive (+{test_amplitude} rad): {result['positive']}")
                print(f"  Negative (-{test_amplitude} rad): {result['negative']}")
                print("-" * 40)
            
            if current_joint_idx < len(test_joints):
                joint_name, description = test_joints[current_joint_idx]
                joint_idx = scene["robot"].data.joint_names.index(joint_name)
                
                print(f"\nTesting joint #{current_joint_idx + 1}/12: {joint_name}")
                print(f"Description: {description}")
                print(f"Default position: {default_joint_pos[joint_idx]:.3f} rad")
                
                # Initialize result storage
                test_results.append({
                    'joint': joint_name,
                    'description': description,
                    'default': default_joint_pos[joint_idx].item(),
                    'positive': "",
                    'negative': ""
                })
                
                # Reset robot
                root_state = scene["robot"].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene["robot"].write_root_pose_to_sim(root_state[:, :7])
                scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
                scene["robot"].write_joint_state_to_sim(default_joint_pos, scene["robot"].data.default_joint_vel.clone())
                scene.reset()

        # Get current joint info
        if current_joint_idx < len(test_joints):
            joint_name, _ = test_joints[current_joint_idx]
            joint_idx = scene["robot"].data.joint_names.index(joint_name)
            
            # Set target position based on test phase
            target_joint_pos = default_joint_pos.clone()
            
            if test_phase == "positive":
                target_joint_pos[joint_idx] = default_joint_pos[joint_idx] + test_amplitude
                if phase_counter == 0:
                    print(f"\n  Testing POSITIVE direction (+{test_amplitude} rad)...")
                elif phase_counter == test_steps // 2:
                    # Observe and record movement
                    actual_pos = scene["robot"].data.joint_pos[0, joint_idx].item()
                    diff = actual_pos - test_results[-1]['default']
                    
                    # Determine movement direction based on joint type
                    if "base_" in joint_name:  # Hip joints
                        if diff > 0:
                            movement = "Leg moves OUTWARD (abduction)"
                        else:
                            movement = "Leg moves INWARD (adduction)"
                    elif "_lf2" in joint_name or "_rf2" in joint_name or "_lb2" in joint_name or "_rb2" in joint_name:  # Thigh joints
                        if diff > 0:
                            movement = "Thigh moves FORWARD (flexion)"
                        else:
                            movement = "Thigh moves BACKWARD (extension)"
                    else:  # Calf joints
                        if diff > 0:
                            movement = "Calf EXTENDS (straightens)"
                        else:
                            movement = "Calf FLEXES (bends)"
                    
                    test_results[-1]['positive'] = movement
                    print(f"    → {movement}")
                    print(f"    Actual position: {actual_pos:.3f} rad (Δ = {diff:+.3f})")
                    
            elif test_phase == "negative":
                target_joint_pos[joint_idx] = default_joint_pos[joint_idx] - test_amplitude
                if phase_counter == 0:
                    print(f"\n  Testing NEGATIVE direction (-{test_amplitude} rad)...")
                elif phase_counter == test_steps // 2:
                    # Observe and record movement
                    actual_pos = scene["robot"].data.joint_pos[0, joint_idx].item()
                    diff = actual_pos - test_results[-1]['default']
                    
                    # Determine movement direction based on joint type
                    if "base_" in joint_name:  # Hip joints
                        if diff < 0:
                            movement = "Leg moves INWARD (adduction)"
                        else:
                            movement = "Leg moves OUTWARD (abduction)"
                    elif "_lf2" in joint_name or "_rf2" in joint_name or "_lb2" in joint_name or "_rb2" in joint_name:  # Thigh joints
                        if diff < 0:
                            movement = "Thigh moves BACKWARD (extension)"
                        else:
                            movement = "Thigh moves FORWARD (flexion)"
                    else:  # Calf joints
                        if diff < 0:
                            movement = "Calf FLEXES (bends)"
                        else:
                            movement = "Calf EXTENDS (straightens)"
                    
                    test_results[-1]['negative'] = movement
                    print(f"    → {movement}")
                    print(f"    Actual position: {actual_pos:.3f} rad (Δ = {diff:+.3f})")
            
            # Apply target position
            scene["robot"].set_joint_position_target(target_joint_pos)
        
        # Phase management
        phase_counter += 1
        
        if test_phase == "settle" and phase_counter >= settle_steps:
            test_phase = "positive"
            phase_counter = 0
        elif test_phase == "positive" and phase_counter >= test_steps:
            test_phase = "negative"
            phase_counter = 0
        elif test_phase == "negative" and phase_counter >= test_steps:
            test_phase = "return"
            phase_counter = 0
        elif test_phase == "return" and phase_counter >= settle_steps:
            # Move to next joint
            current_joint_idx += 1
            test_phase = "settle"
            phase_counter = 0
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        
    # Final summary
    if current_joint_idx >= len(test_joints) and test_results:
        # Print last result
        joint_name, desc = test_joints[-1]
        result = test_results[-1]
        print(f"\nRESULT for {joint_name}:")
        print(f"  Positive (+{test_amplitude} rad): {result['positive']}")
        print(f"  Negative (-{test_amplitude} rad): {result['negative']}")
        
        print("\n" + "="*60)
        print("JOINT DIRECTION TEST COMPLETE - SUMMARY")
        print("="*60)
        
        for result in test_results:
            print(f"\n{result['joint']} ({result['description']}):")
            print(f"  Default: {result['default']:.3f} rad")
            print(f"  Positive: {result['positive']}")
            print(f"  Negative: {result['negative']}")
        
        print("\n" + "="*60)
        print("Test complete! This confirms the Isaac Sim joint conventions.")
        print("="*60)

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

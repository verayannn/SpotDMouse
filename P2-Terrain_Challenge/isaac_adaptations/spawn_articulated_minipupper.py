# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(
#     description="This script demonstrates adding a custom robot to an Isaac Lab environment."
# )
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import numpy as np
# import torch

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets import AssetBaseCfg
# from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
# from isaaclab.assets.articulation import ArticulationCfg
# import os
# from math import pi

# # cfg_robot = ArticulationCfg(
# #     spawn=sim_utils.UsdFileCfg(usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd"),
# #     actuators={"leg_actuators": ImplicitActuatorCfg(joint_names_expr=[".*"],damping=1.0,stiffness=0.0)},
# # )

# # cfg_robot = ArticulationCfg(
# #     spawn=sim_utils.UsdFileCfg(usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd"),
# #     actuators={"leg_actuators": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=0.006, stiffness=10.0)},
# #     init_state=ArticulationCfg.InitialStateCfg(
# #         joint_pos={
# #             "base_lf1": -0.1181,
# #             "lf1_lf2": 0.8360,
# #             "lf2_lf3": -1.6081,
# #             "base_rf1": 0.1066,
# #             "rf1_rf2": 0.8202,
# #             "rf2_rf3": -1.6161,
# #             "base_lb1": -0.0522,
# #             "lb1_lb2": 0.8198,
# #             "lb2_lb3": -1.6220,
# #             "base_rb1": 0.0663,
# #             "rb1_rb2": 0.7983,
# #             "rb2_rb3": -1.6382,
# #         },
# #         pos=(0.0, 0.0, 0.11),  # This base height is fine to prevent collisions on load
# #     ),
# # )

# cfg_robot = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.1,
#             angular_damping=0.1,
#             max_linear_velocity=1.0,
#             max_angular_velocity=3.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=1,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.11),
#         joint_pos={
#             "base_lf1": -0.1181,
#             "lf1_lf2": 0.8360,
#             "lf2_lf3": -1.6081,
#             "base_rf1": 0.1066,
#             "rf1_rf2": 0.8202,
#             "rf2_rf3": -1.6161,
#             "base_lb1": -0.0522,
#             "lb1_lb2": 0.8198,
#             "lb2_lb3": -1.6220,
#             "base_rb1": 0.0663,
#             "rb1_rb2": 0.7983,
#             "rb2_rb3": -1.6382,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "leg_actuators": DCMotorCfg(
#             joint_names_expr=[".*"],
#             effort_limit=0.6,       # 4x higher effort
#             saturation_effort=0.6,
#             velocity_limit=0.2,     # Slightly faster corrections
#             stiffness=1.6,          # 4x higher stiffness
#             damping=1.6,            # Critical damping matches stiffness
#             friction=0.1,
#         )
#     },
# )


# class NewRobotsSceneCfg(InteractiveSceneCfg):
#     """Designs the scene."""

#     # Ground-plane
#     ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

#     # lights
#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     )

#     # robot
#     robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Spot")

# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     # Cache standing joint position target
#     default_joint_pos = scene["robot"].data.default_joint_pos.clone()

#     while simulation_app.is_running():
#         # Periodic reset
#         if count % 500 == 0:
#             count = 0
#             print("[INFO]: Resetting Mini Pupper state...")

#             # Reset root pose and velocity
#             root_state = scene["robot"].data.default_root_state.clone()
#             root_state[:, :3] += scene.env_origins
#             scene["robot"].write_root_pose_to_sim(root_state[:, :7])
#             scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])

#             # Reset joints
#             scene["robot"].write_joint_state_to_sim(default_joint_pos, scene["robot"].data.default_joint_vel.clone())

#             # Clear internal buffers
#             scene.reset()

#         # Continuously apply standing joint targets to hold position
#         scene["robot"].set_joint_position_target(default_joint_pos)

#         # Step sim
#         scene.write_data_to_sim()
#         sim.step()
#         sim_time += sim_dt
#         count += 1
#         scene.update(sim_dt)


# def main():
#     """Main function."""
#     # Initialize the simulation context
#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim = sim_utils.SimulationContext(sim_cfg)

#     sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
#     # design scene
#     scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)
#     # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # Run the simulator
#     run_simulator(sim, scene)


# if __name__ == "__main__":
#     main()
#     simulation_app.close()


# ###
# ###
# ###


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

def check_groups(scene, cfg):
    live = list(scene["robot"].data.joint_names)
    for gname, act in cfg.actuators.items():
        live_subset = [n for n in live if any(re.fullmatch(expr, n)
                                              for expr in act.joint_names_expr)]
        expected_subset = _expand_expr(act.joint_names_expr, live)
        print(f"{gname:13s}:",
              "✔" if live_subset == expected_subset else "❌",
              live_subset)

def _expand_expr(expr_list, full_name_list):
    """Return the subset of full_name_list that matches the (ordered) regex list."""
    out = []
    for expr in expr_list:
        pattern = re.compile(expr)
        for name in full_name_list:
            if pattern.fullmatch(name) and name not in out:
                out.append(name)
    return out

def check_joint_order(scene, cfg):
    # 1. Live DOF order --------------------------------------------------------
    live_names = list(scene["robot"].data.joint_names)

    # 2. Expected order per actuator group -------------------------------------
    expected_by_group = {}
    for group_name, act_cfg in cfg.actuators.items():
        expr = act_cfg.joint_names_expr
        expected_by_group[group_name] = _expand_expr(expr, live_names)

    expected_flat = [j for group in expected_by_group.values() for j in group]

    # 3. Diff ------------------------------------------------------------------
    print("\n\n===== JOINT‑ORDER CHECK =====")
    if live_names == expected_flat:
        print("✔ Joint ordering matches YAML exactly!")
    else:
        print("❌ Ordering mismatch detected\n")
        width = max(len(j) for j in live_names) + 2
        print(" idx | live".ljust(width + 6) + "| expected")
        print("-" * (width * 2 + 9))
        for i, (l, e) in enumerate(zip(live_names, expected_flat)):
            mark = " " if l == e else "<"
            print(f"{i:4d} | {l.ljust(width)}{mark}| {e}")
        print("\nFirst mismatch at index "
              f"{next(i for i,(l,e) in enumerate(zip(live_names, expected_flat)) if l!=e)}")
    print("================================\n")
    # (optional) return bool
    return live_names == expected_flat

# ACCURATE for real MiniPupper: 560g total weight
cfg_robot = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper/fixed_mini_pupper/fixed_mini_pupper.usd",
        activate_contact_sensors=True,
        # Realistic mass distribution for 560g MiniPupper
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.45,  # ~450g in base (80% of total weight: battery, Pi, PCBs)
                       # Leaves ~110g for all 12 leg segments (carbon fiber)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.005,   # Very low for small, lightweight robot
            angular_damping=0.005,  # Very low - carbon fiber has minimal drag
            max_linear_velocity=15.0,  # Small robots can be quite fast
            max_angular_velocity=25.0, # High agility for lightweight design
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=6,   # Lower for lightweight robot
            solver_velocity_iteration_count=1,   # Standard
            sleep_threshold=0.01,               # Appropriate for 560g robot
            stabilization_threshold=0.002,      # Good balance
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
        # In your init_state, try slightly straighter legs:
        joint_pos={
            # Less bent legs (30° instead of 45°)
            "base_lf1": 0.0,      
            "lf1_lf2": 0.52,      # ~30° (π/6 radians)
            "lf2_lf3": -1.05,     # ~60° to make foot flat
            
            "base_rf1": 0.0,      
            "rf1_rf2": 0.52,      
            "rf2_rf3": -1.05,     
            
            "base_lb1": 0.0,      
            "lb1_lb2": 0.52,      
            "lb2_lb3": -1.05,     
            
            "base_rb1": 0.0,      
            "rb1_rb2": 0.52,      
            "rb2_rb3": -1.05,     
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
    # Main leg joints (the ones that do the real work)
    "leg_joints": DCMotorCfg(
        joint_names_expr=[
        "base_lb1", "base_lf1", "base_rb1", "base_rf1",
        "lb1_lb2", "lf1_lf2", "rb1_rb2", "rf1_rf2",
        "lb2_lb3", "lf2_lf3", "rb2_rb3", "rf2_rf3",
            ],
        saturation_effort=3.0,
        velocity_limit=5,
        stiffness=35.0,#50.0        
        damping=0.7,#7.0          
        friction=0.05,        
        armature=0.001,      
    ),
        # saturation_effort=2.5,
        # velocity_limit=1.5,
        # stiffness=50.0,#35.0        
        # damping=7.0,#7.0          
        # friction=0.05,        
        # armature=0.001,
    # Foot joints (very tight, almost fixed)
    "foot_joints": DCMotorCfg(
        joint_names_expr=["lb3_foot", "lf3_foot", "rb3_foot", "rf3_foot"],
        saturation_effort=1000.0,  # Very high to keep rigid
        velocity_limit=0.1,        # Very slow movement
        stiffness=1000.0,          # Very stiff - almost fixed
        damping=100.0,             # Heavy damping
        friction=0.4,              
        armature=0.001,       
    ),
    
    # Plate joints (decorative, should be rigid)
    "plate_joints": DCMotorCfg(
        joint_names_expr=[
        "lb1_plate", "lf1_plate", "rb1_plate", "rf1_plate",
        "lb2_plate", "lf2_plate", "rb2_plate", "rf2_plate",
            ],
        saturation_effort=1000.0,  # Very high to keep rigid
        velocity_limit=0.1,        # Very slow movement
        stiffness=1000.0,          # Very stiff - almost fixed
        damping=100.0,             # Heavy damping
        friction=0.1,              # Lower friction for internal joints
        armature=0.001,       
    ),
    
    # Sensor joints (should be completely rigid)
    "sensor_joints": DCMotorCfg(
        joint_names_expr=["base_lidar", "imu_joint"],
        saturation_effort=1000.0,  # Very high to keep rigid
        velocity_limit=0.01,       # Almost no movement
        stiffness=2000.0,          # Extremely stiff
        damping=200.0,             # Very heavy damping
        friction=0.1,              
        armature=0.001,       
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

# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0
    
#     # Cache standing joint position target
#     default_joint_pos = scene["robot"].data.default_joint_pos.clone()
    
#     # Add small settling period after spawn
#     settling_steps = 100
    
#     print(f"[DEBUG] Sim Dt Value:{sim_dt}")
#     print(f"[DEBUG] Joint targets: {default_joint_pos[0]}")
#     print(f"[DEBUG] Actual joints: {scene['robot'].data.joint_pos[0]}")
#     print(f"[DEBUG] Joint errors: {default_joint_pos[0] - scene['robot'].data.joint_pos[0]}")
#     while simulation_app.is_running():
#         # Periodic reset
#         if count % 1000 == 0:  # Increased reset interval
#             count = 0
#             print("[INFO]: Resetting Mini Pupper state...")
            
#             # Reset root pose and velocity
#             root_state = scene["robot"].data.default_root_state.clone()
#             root_state[:, :3] += scene.env_origins
#             # Add small random perturbation to test stability
#             root_state[:, 2] += torch.randn_like(root_state[:, 2]) * 0.01
            
#             scene["robot"].write_root_pose_to_sim(root_state[:, :7])
#             scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            
#             # Reset joints with small random perturbation
#             joint_pos_reset = default_joint_pos.clone()
#             # joint_pos_reset += torch.randn_like(joint_pos_reset) * 0.05
#             scene["robot"].write_joint_state_to_sim(joint_pos_reset, scene["robot"].data.default_joint_vel.clone())
            
#             # Clear internal buffers
#             scene.reset()
        
#         # Apply standing joint targets with gentle settling
#         if count < settling_steps:
#             # Gradual settling - interpolate to target position
#             alpha = count / settling_steps
#             current_pos = scene["robot"].data.joint_pos
#             target_pos = alpha * default_joint_pos + (1 - alpha) * current_pos
#             scene["robot"].set_joint_position_target(target_pos)
#         else:
#             # Normal position control
#             scene["robot"].set_joint_position_target(default_joint_pos)
        
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
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # --- START OF MODIFIED SECTION ---
    # 1. Manually define the desired initial joint positions
    #    This MUST match the order of `scene["robot"].data.joint_names`
    #    Refer to your "[DEBUG]: Joint Names:" output for the exact order.
    #    The provided order is:
    #    ['base_lb1', 'base_lf1', 'base_lidar', 'base_rb1', 'base_rf1', 'imu_joint',
    #     'lb1_lb2', 'lb1_plate', 'lf1_lf2', 'lf1_plate', 'rb1_plate', 'rb1_rb2',
    #     'rf1_plate', 'rf1_rf2', 'lb2_lb3', 'lb2_plate', 'lf2_lf3', 'lf2_plate',
    #     'rb2_plate', 'rb2_rb3', 'rf2_plate', 'rf2_rf3', 'lb3_foot', 'lf3_foot',
    #     'rb3_foot', 'rf3_foot']

    # Create a dictionary for easier lookup (if you prefer, otherwise list directly in order)
    desired_joint_angles_dict = {
        "base_lf1": 0.0,
        "lf1_lf2": 0.52,
        "lf2_lf3": -1.05,
        "base_rf1": 0.0,
        "rf1_rf2": 0.52,
        "rf2_rf3": -1.05,
        "base_lb1": 0.0,
        "lb1_lb2": 0.52,
        "lb2_lb3": -1.05,
        "base_rb1": 0.0,
        "rb1_rb2": 0.52,
        "rb2_rb3": -1.05,
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

    # Add small settling period after spawn
    settling_steps = 100
    
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
            check_joint_order(scene, cfg_robot)
            check_groups(scene, cfg_robot)
            print("Leg order:", cfg_robot.actuators["leg_joints"].joint_names_expr)

                    
        # Apply standing joint targets with gentle settling
        if count < settling_steps:
            # Gradual settling - interpolate to target position
            alpha = count / settling_steps
            current_pos = scene["robot"].data.joint_pos[0] # Get current actual joint positions
            # Interpolate from current_pos to default_joint_pos (your desired bent pose)
            target_pos = alpha * default_joint_pos + (1 - alpha) * current_pos
            scene["robot"].set_joint_position_target(target_pos)
        else:
            # Normal position control
            scene["robot"].set_joint_position_target(default_joint_pos)
        
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

# ###
# ###
# ###


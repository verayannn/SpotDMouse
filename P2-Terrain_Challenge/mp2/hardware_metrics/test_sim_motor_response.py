#!/usr/bin/env python3
"""
Run inside Isaac Sim: python test_sim_motor_response.py
Sends same freq sweep signals as test_real_motor_response.py
and records the simulated motor response for comparison.
"""
import argparse

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import csv

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# --- Robot config (same as working stand script) ---
cfg_robot = ArticulationCfg(
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
                "base_rb1", "rb1_rb2", "rb2_rb3",
            ],
            saturation_effort=0.35,
            velocity_limit=10.5,
            stiffness=80.0,
            damping=2.5,
            friction=0.03,
            armature=0.005,
        ),
    },
)

class TestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

SIM_DT = 0.002
CONTROL_DT = 0.02

# Joints to test, by name (thigh joints)
TEST_JOINTS = [
    ("lf1_lf2", "LF_thigh"),
    ("rf1_rf2", "RF_thigh"),
    ("lb1_lb2", "LB_thigh"),
    ("rb1_rb2", "RB_thigh"),
]

TEST_FREQS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
AMPLITUDE = 0.3
DURATION_PER_FREQ = 3.0
SQUARE = False


def run_test(sim, scene):
    sim_dt = sim.get_physics_dt()
    decimation = int(CONTROL_DT / SIM_DT)

    # Build default joint pos tensor from joint names (same pattern as working script)
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
    }

    joint_names = scene["robot"].data.joint_names
    print(f"[DEBUG] All joint names: {list(enumerate(joint_names))}")

    default_pos = torch.zeros(len(joint_names), device=sim.device, dtype=torch.float)
    for i, name in enumerate(joint_names):
        if name in desired_joint_angles_dict:
            default_pos[i] = desired_joint_angles_dict[name]

    # Resolve test joint indices by name
    test_motor_indices = []
    for joint_name, friendly_name in TEST_JOINTS:
        idx = joint_names.index(joint_name)
        test_motor_indices.append((idx, joint_name, friendly_name))
        print(f"[DEBUG] {friendly_name} ({joint_name}) -> index {idx}")

    print("=" * 60)
    print("SIM MOTOR RESPONSE CHARACTERIZATION")
    print(f"Actuator: DCMotorCfg saturation effort= 0.35, velocity limit = 10.5, stiffness=80, damping=2.5, friction=0.03, armature=0.005")
    print(f"Motors: {[f[2] for f in test_motor_indices]}")
    print(f"Freqs: {TEST_FREQS} Hz, Amp: {AMPLITUDE} rad")
    print("=" * 60)

    for motor_idx, joint_name, friendly_name in test_motor_indices:
        all_rows = []

        for freq in TEST_FREQS:
            print(f"  {friendly_name} ({joint_name}, idx {motor_idx}) @ {freq}Hz...", end=" ", flush=True)

            # Reset to standing
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene["robot"].write_joint_state_to_sim(default_pos, scene["robot"].data.default_joint_vel.clone())
            scene.reset()

            # Settle
            for _ in range(50):
                scene["robot"].set_joint_position_target(default_pos)
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)

            phase = 0.0
            elapsed = 0.0
            count = 0

            while elapsed < DURATION_PER_FREQ:
                phase += freq * 2 * np.pi * CONTROL_DT

                if SQUARE:
                    signal = np.sign(np.sin(phase))
                else:
                    signal = np.sin(phase)
                command = signal * AMPLITUDE

                target = default_pos.clone()
                target[motor_idx] += command

                for _ in range(decimation):
                    scene["robot"].set_joint_position_target(target)
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)

                actual_pos = scene["robot"].data.joint_pos[0, motor_idx].item()
                actual_vel = scene["robot"].data.joint_vel[0, motor_idx].item()
                default_val = default_pos[motor_idx].item()
                actual_rel = actual_pos - default_val

                all_rows.append([elapsed, phase, AMPLITUDE, freq, motor_idx, signal, command, actual_rel, actual_vel])

                elapsed += CONTROL_DT
                count += 1

            print(f"{count} samples")

        fname = f"sim_motor_{motor_idx}_{friendly_name}_amp{AMPLITUDE}.csv"
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_s', 'phase', 'amp_rad', 'freq_hz', 'motor_idx', 'signal', 'command_rad', 'actual_rad', 'actual_vel'])
            w.writerows(all_rows)
        print(f"  Saved: {fname}")


def main():
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=SIM_DT,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            enable_stabilization=True,
            bounce_threshold_velocity=0.05,
            friction_offset_threshold=0.02,
            friction_correlation_distance=0.01,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.3])

    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO]: Setup complete...")
    run_test(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\nDone. Compare these CSVs against real motor output.")
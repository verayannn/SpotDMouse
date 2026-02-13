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

# --- All other imports AFTER AppLauncher ---
import torch
import numpy as np
import csv

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import SimulationContext

# --- Define robot config inline (same as your working stand script) ---
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

# --- Rest of the script unchanged ---
SIM_DT = 0.002
CONTROL_DT = 0.02

sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, device="cpu")
sim = SimulationContext(sim_cfg)

sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot_cfg = CUSTOM_QUAD_CFG.copy()
robot_cfg.prim_path = "/World/Robot"
robot = Articulation(cfg=robot_cfg)

sim.reset()
robot.reset()

default_pos = robot.data.default_joint_pos.clone()
num_joints = default_pos.shape[1]
decimation = int(CONTROL_DT / SIM_DT)

NAMES = ['LF_hip','LF_thigh','LF_calf','RF_hip','RF_thigh','RF_calf',
         'LB_hip','LB_thigh','LB_calf','RB_hip','RB_thigh','RB_calf']

TEST_MOTORS = [1, 4, 7, 10]
TEST_FREQS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
AMPLITUDE = 0.3
DURATION_PER_FREQ = 3.0
SQUARE = False

print("=" * 60)
print("SIM MOTOR RESPONSE CHARACTERIZATION")
print(f"Actuator: DCMotorCfg stiffness=80, damping=2.5, friction=0.03, armature=0.005")
print(f"Motors: {[NAMES[m] for m in TEST_MOTORS]}")
print(f"Freqs: {TEST_FREQS} Hz, Amp: {AMPLITUDE} rad")
print("=" * 60)

for motor_idx in TEST_MOTORS:
    all_rows = []

    for freq in TEST_FREQS:
        print(f"  {NAMES[motor_idx]} @ {freq}Hz...", end=" ", flush=True)

        robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos))
        robot.reset()
        for _ in range(50):
            robot.set_joint_position_target(default_pos)
            robot.write_data_to_sim()
            sim.step()
            robot.update(SIM_DT)

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
            target[0, motor_idx] += command

            for _ in range(decimation):
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
                sim.step()
                robot.update(SIM_DT)

            actual_pos = robot.data.joint_pos[0, motor_idx].item()
            actual_vel = robot.data.joint_vel[0, motor_idx].item()
            default_val = default_pos[0, motor_idx].item()
            actual_rel = actual_pos - default_val

            all_rows.append([elapsed, phase, AMPLITUDE, freq, motor_idx, signal, command, actual_rel, actual_vel])

            elapsed += CONTROL_DT
            count += 1

        print(f"{count} samples")

    fname = f"sim_motor_{motor_idx}_{NAMES[motor_idx]}_amp{AMPLITUDE}.csv"
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s','phase','amp_rad','freq_hz','motor_idx','signal','command_rad','actual_rad','actual_vel'])
        w.writerows(all_rows)
    print(f"  Saved: {fname}")

simulation_app.close()
print("\nDone. Compare these CSVs against real motor output.")

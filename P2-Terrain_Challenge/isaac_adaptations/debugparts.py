# Create a debug script to check all available body names and their properties
# filepath: /workspace/debug_bodies.py

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug body names")
parser.add_argument("--headless", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.custom_quadruped.custom_quad import CUSTOM_QUAD_CFG

class DebugSceneCfg(InteractiveSceneCfg):
    robot = CUSTOM_QUAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=0.01))
    scene_cfg = DebugSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    
    print("\n" + "="*60)
    print("ALL AVAILABLE BODY NAMES:")
    print("="*60)
    
    # Check contact forces sensor
    if "contact_forces" in scene:
        contact_sensor = scene["contact_forces"]
        print(f"\nContact sensor body names: {contact_sensor.data.body_names}")
        
        # Check body indices
        print("\nBody indices mapping:")
        for i, name in enumerate(contact_sensor.data.body_names):
            print(f"  Index {i}: {name}")
    
    # Check robot bodies
    robot = scene["robot"]
    print(f"\nRobot body names: {robot.data.body_names}")
    
    # Check for any bodies with "3" in name (potential foot segments)
    print("\nBodies ending with '3' (potential foot segments):")
    for name in robot.data.body_names:
        if name.endswith("3"):
            print(f"  - {name}")

if __name__ == "__main__":
    main()
    simulation_app.close()
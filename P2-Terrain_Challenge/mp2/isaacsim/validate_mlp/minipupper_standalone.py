from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from minipupper_isaaclab_policy import MiniPupperIsaacLabPolicy
from isaacsim.storage.native import get_assets_root_path

# Create world with same physics settings as training
my_world = World(
    stage_units_in_meters=1.0,
    physics_dt=0.002,  # Match your training: 500Hz sim, but 0.002 dt
    rendering_dt=1/50
)

# Add ground plane
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

ground_prim = define_prim("/World/Ground", "Xform")
ground_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground_prim.GetReferences().AddReference(ground_path)

# Create robot
minipupper = MiniPupperIsaacLabPolicy(
    prim_path="/World/MiniPupper",
    name="MiniPupper",
    position=np.array([0, 0, 0.15])
)

# Reset world and initialize robot
my_world.reset()
minipupper.initialize(my_world)

# Command tracking
base_command = np.zeros(3)
step_count = 0

# Main simulation loop
while simulation_app.is_running():
    if my_world.is_playing():
        # Update commands
        if step_count < 1000:
            base_command = np.array([0.3, 0, 0])  # forward
        elif step_count < 1500:
            base_command = np.array([0.2, 0, 0.2])  # turn
        elif step_count < 2000:
            base_command = np.array([0, 0.2, 0])  # sideways
        else:
            step_count = 0
            
        # Step robot controller
        minipupper.forward(my_world.physics_dt, base_command)
        step_count += 1
    
    # Step world
    my_world.step(render=True)
    
    # Handle reset
    if my_world.is_stopped() and my_world.is_playing():
        my_world.stop()
        my_world.play()

simulation_app.close() 
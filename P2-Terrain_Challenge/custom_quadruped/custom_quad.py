import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

CUSTOM_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd",
        usd_path="/workspace/ros2_ws/src/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper/mini_pupper_description/mini_pupper_description.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
        # In your init_state, try slightly straighter legs:
        joint_pos={
            # 30° Legs 'harvardrun' # Need to try 45 degree legs since that is what the actual pupper has 45°
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
            # 45° 'harvardrun_45'
            # "base_lf1": 0.0,      
            # "lf1_lf2": 0.785,     # π/4 radians = 45°
            # "lf2_lf3": -1.57,     # -π/2 radians = -90° (to keep foot flat)

            # "base_rf1": 0.0,      
            # "rf1_rf2": 0.785,     # π/4 radians = 45°
            # "rf2_rf3": -1.57,     # -π/2 radians = -90°

            # "base_lb1": 0.0,      
            # "lb1_lb2": 0.785,     # π/4 radians = 45°
            # "lb2_lb3": -1.57,     # -π/2 radians = -90°

            # "base_rb1": 0.0,      
            # "rb1_rb2": 0.785,     # π/4 radians = 45°            # "rb2_rb3": -1.57,     # -π/2 radians = -90°
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
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
        saturation_effort=2.5,
        velocity_limit=10.0,
        stiffness=80.0,        
        damping=2.0,          
        friction=0.02,        
        armature=0.005,      
    ),
    }
)


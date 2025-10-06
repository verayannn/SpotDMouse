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
            # 30° Thigh-Calf 'harvardrun_30'
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
        stiffness=45.0,        
        damping=1.3,          
        friction=0.02,        
        armature=0.005,#0.004269, # Sweet spot - jitters in place, no drift      
    ),
    }
)

#settings
#ORIGINAL: stiffness = 80, damping = 2.0 --> only walking forward
#TRIAL 1: stiffness = 45, damping = 2.0 --> mostly standing in place
#TRIAL 2: stiffness = 20, damping = 2.0 --> mostly standing in place
#TRIAL 3: stiffness = 45, damping = 1.3 --> only walking forward
#TRIAL 4: stiffness = 20, damping = 0.8 --> mostly standing in place
#TRIAL 5: stiffness = 45, damping = 1.3, base_angular_velocity=10, base_linear_velocity=20--> moves in +x, -x, +y, -y does not rotate wrt to any angular command and the posture is good! I may need to make an angular rotawtional reward
#TRIAL 6: stiffness = 45, damping = 1.3, base_angular_velocity=10, base_linear_velocity=20, joint_torques_penalty=-1.0e-4--> moves in +x, -x, +y, -y does not rotate wrt to any angular command and the base_orientation is not flat enough and the limbs are a bit unlike the standing pose.
#TRIAL 7: stiffness = 45, damping = 1.3, base_angular_velocity=20, base_linear_velocity=20--> Follows commands, but the legs are splayed out too far for it to be a  aesthticvally  pleasing, and engergitcally favorable method of gait.
#test categories:lin_vel_x=(0.2, 0.2),lin_vel_y=(0.0, 0.0),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(-0.2, -0.2),lin_vel_y=(0.0, 0.0),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(0.2, 0.2),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(-0.2, -0.2),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(0.0, 0.0),ang_vel_z=(0.2, 0.2) 
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(0.0, 0.0),ang_vel_z=(-0.2, -0.2) 


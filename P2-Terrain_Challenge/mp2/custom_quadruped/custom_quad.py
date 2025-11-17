import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

CUSTOM_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd",
        #Verify path
        usd_path="/workspace/ros2_ws/src/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper_2/mini_pupper_description/mini_pupper_description.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
        # In your init_state, try slightly straighter legs:
        joint_pos={
            # 45° 'harvardrun_45'
            "base_lf1": 0.0,      
            "lf1_lf2": 0.785,     # π/4 radians = 45°
            "lf2_lf3": -1.57,     # -π/2 radians = -90° (to keep foot flat)

            "base_rf1": 0.0,      
            "rf1_rf2": 0.785,     # π/4 radians = 45°
            "rf2_rf3": -1.57,     # -π/2 radians = -90°

            "base_lb1": 0.0,      
            "lb1_lb2": 0.785,     # π/4 radians = 45°
            "lb2_lb3": -1.57,     # -π/2 radians = -90°

            "base_rb1": 0.0,      
            "rb1_rb2": 0.785,     # π/4 radians = 45°            
            "rb2_rb3": -1.57,     # -π/2 radians = -90°
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
        # saturation_effort=2.5,
        # velocity_limit=10.0,
        # stiffness=45.0,        
        # damping=1.3,          
        # friction=0.02,        
        # armature=0.005,#0.004269, # Sweet spot - jitters in place, no drift
        # Accurate specs from https://www.robotshop.com/products/mangdang-high-performance-35kg-cm-robot-digital-servo?qd=cc36ca2653f9fea65ad13bd91c459f1c
        saturation_effort=0.35, # 3.5 kg·cm converted to N·m
        velocity_limit=10.5, # 0.1s/60° = 10.47 rad/s
        stiffness=80.0,#80.0 Official/Final: 45.0       
        damping=2.5,#2.0 Official/Final: 1.3     
        friction=0.03,        
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
#TRIAL 8: stiffness = 45, damping = 1.3, base_angular_velocity=20, base_linear_velocity=20, slip_penalty=-2.0 --> Best in show. Follows, commands the angular commands results in a donut walk (aerial) not rotating in place
#TRIAL 9: saturation_effort=0.35, velocity=10.5, stiffness = 80, damping = 2.5, friction=0.03,armature=0.005, base_angular_velocity=20, base_linear_velocity=20, slip_penalty=-2.0 --> Worked almost perfectly, the forard command gives a slightly north east/north west bias, but that can be recovered by supplementing with y or angular commands.
#test categories:lin_vel_x=(0.2, 0.2),lin_vel_y=(0.0, 0.0),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(-0.2, -0.2),lin_vel_y=(0.0, 0.0),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(0.2, 0.2),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(-0.2, -0.2),ang_vel_z=(0.0, 0.0)
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(0.0, 0.0),ang_vel_z=(0.2, 0.2) 
#test categories:lin_vel_x=(0.0, 0.0),lin_vel_y=(0.0, 0.0),ang_vel_z=(-0.2, -0.2) 


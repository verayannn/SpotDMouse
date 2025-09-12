import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

CUSTOM_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd",
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
            solver_position_iteration_count=8,   # Lower for lightweight robot
            solver_velocity_iteration_count=2,   # Standard
            sleep_threshold=0.005,               # Appropriate for 560g robot
            stabilization_threshold=0.001,      # Good balance
            fix_root_link=False,
        ),
    ),

    ###
    ### for RSL RL
    ###
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
    #     # In your init_state, try slightly straighter legs:
    #     joint_pos={
    #         # Less bent legs (30° instead of 45°)
    #         "base_lf1": 0.0,      
    #         "lf1_lf2": 0.52,      # ~30° (π/6 radians)
    #         "lf2_lf3": -1.05,     # ~60° to make foot flat
            
    #         "base_rf1": 0.0,      
    #         "rf1_rf2": 0.52,      
    #         "rf2_rf3": -1.05,     
            
    #         "base_lb1": 0.0,      
    #         "lb1_lb2": 0.52,      
    #         "lb2_lb3": -1.05,     
            
    #         "base_rb1": 0.0,      
    #         "rb1_rb2": 0.52,      
    #         "rb2_rb3": -1.05,     
    #     },
    #     joint_vel={".*": 0.0},
    # ),
    # soft_joint_pos_limit_factor=0.95,
    ###
    ### for RSL RL
    ###

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
        # In your init_state, try slightly straighter legs:
        joint_pos={
            # Less bent legs (30° instead of 45°)
            "base_lf1": -0.214,      
            "lf1_lf2": -1.940,      # ~30° (π/6 radians)
            "lf2_lf3": 1.993,     # ~60° to make foot flat
            
            "base_rf1": -0.260,      
            "rf1_rf2": -0.891,      
            "rf2_rf3": 1.047,     
            
            "base_lb1": 0.387,      
            "lb1_lb2": -0.233,      
            "lb2_lb3":  0.907,     
            
            "base_rb1": -0.038,      
            "rb1_rb2": -1.247,      
            "rb2_rb3": 1.945,     
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,


    actuators={

    ###
    ### for RSL RL
    ###
    # Main leg joints (the ones that do the real work)
    # "leg_joints": DCMotorCfg(
    #     joint_names_expr=[
    #         # LF leg (front-left)
    #         "base_lf1", "lf1_lf2", "lf2_lf3",
    #         # RF leg (front-right)  
    #         "base_rf1", "rf1_rf2", "rf2_rf3",
    #         # LB leg (back-left)
    #         "base_lb1", "lb1_lb2", "lb2_lb3",
    #         # RB leg (back-right)
    #         "base_rb1", "rb1_rb2", "rb2_rb3"
    #     ],
    #     saturation_effort=2.5,
    #     velocity_limit=1.5,
    #     stiffness=50.0,#35.0        
    #     damping=7.0,#7.0          
    #     friction=0.05,        
    #     armature=0.001,      
    # ),

    ###
    ### for RSL RL
    ###

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
        saturation_effort=1.5,
        velocity_limit=2.0,
        stiffness=40.0,#35.0        
        damping=5.0,#7.0          
        friction=0.05,        
        armature=0.001,      
    ),


    
    # Foot joints (very tight, almost fixed)
    "foot_joints": DCMotorCfg(
        joint_names_expr=["lf3_foot", "rf3_foot", "lb3_foot", "rb3_foot"],
        saturation_effort=1000.0,  # Very high to keep rigid
        velocity_limit=0.1,        # Very slow movement
        stiffness=1000.0,          # Very stiff - almost fixed
        damping=100.0,             # Heavy damping
        friction=0.4,              
        armature=0.001,       
    ),
    
    # Plate joints (decorative, should be rigid)
    "plate_joints": DCMotorCfg(
        joint_names_expr=["lf1_plate", "rf1_plate", "lb1_plate", "rb1_plate",
                          "lf2_plate", "rf2_plate", "lb2_plate", "rb2_plate"],
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


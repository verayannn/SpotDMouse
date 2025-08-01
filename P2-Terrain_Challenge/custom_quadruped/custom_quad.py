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
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.08),  # Conservative standing height
    #     joint_pos={
    #         "base_lf1": -0.1181,
    #         "lf1_lf2": 0.8360,
    #         "lf2_lf3": -1.6081,
    #         "base_rf1": 0.1066,
    #         "rf1_rf2": 0.8202,
    #         "rf2_rf3": -1.6161,
    #         "base_lb1": -0.0522,
    #         "lb1_lb2": 0.8198,
    #         "lb2_lb3": -1.6220,
    #         "base_rb1": 0.0663,
    #         "rb1_rb2": 0.7983,
    #         "rb2_rb3": -1.6382,
    #     },
        
    #     joint_vel={".*": 0.0},
    # ),
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

    # actuators={
    #     "leg_actuators": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=100.0,        # Was 60.0 - much stiffer!
    #         damping=20.0,          # Was 12.0 - stronger damping
    #         effort_limit=6.0,      # Was 4.0 - more torque available
    #         velocity_limit=3.0,    # Was 2.0 - allow faster movements
    #     )
    # }
    
    # actuators={
    #     "leg_actuators": DCMotorCfg(
    #         joint_names_expr=[".*"],
    #         saturation_effort=2.5,
    #         velocity_limit=1.5,
    #         stiffness=50.0,#35.0        
    #         damping=7.0,#7.0          
    #         friction=0.05,        
    #         armature=0.001,       
    #     )
    # }

    # actuators={
    #     "leg_actuators": DCMotorCfg(
    #         joint_names_expr=[".*"],
    #         saturation_effort=3.0,
    #         velocity_limit=1.5,
    #         stiffness=50.0,        # Lower for more realistic servo response
    #         damping=10.0,          # Lower for more realistic dynamics
    #         friction=0.05,        # Servo friction
    #         armature=0.001,       # Small servo inertia
    #     )
    # }
        # soft_joint_pos_limit_factor=0.95,
    actuators={
    # Main leg joints (the ones that do the real work)
    "leg_joints": DCMotorCfg(
        joint_names_expr=["base_lf1", "base_rf1", "base_lb1", "base_rb1", 
                        "lf1_lf2", "rf1_rf2", "lb1_lb2", "rb1_rb2",
                        "lf2_lf3", "rf2_rf3", "lb2_lb3", "rb2_rb3"],
        saturation_effort=2.5,
        velocity_limit=1.5,
        stiffness=50.0,#35.0        
        damping=7.0,#7.0          
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


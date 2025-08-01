"""
Minimal gait addition to your existing standing script.
This adds just the essential gait logic using CHAMP knowledge.
"""

import argparse
import torch
import numpy as np
import math

from isaaclab.app import AppLauncher

# Your existing argument parsing (unchanged)
parser = argparse.ArgumentParser(description="MiniPupper with minimal gait")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--enable_gait", action="store_true", help="Enable simple gait")
parser.add_argument("--gait_speed", type=float, default=0.1, help="Gait speed (m/s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Your existing imports (unchanged)
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Your existing robot config (unchanged)
cfg_robot = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper/fixed_mini_pupper/fixed_mini_pupper.usd",
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.45),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.005,
            angular_damping=0.005,
            max_linear_velocity=15.0,
            max_angular_velocity=25.0,
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.01,
            stabilization_threshold=0.002,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.09),  # CHAMP height: 20cm
        joint_pos={
            "base_lf1": 0.0, "lf1_lf2": 0.52, "lf2_lf3": -1.05,
            "base_rf1": 0.0, "rf1_rf2": 0.52, "rf2_rf3": -1.05,
            "base_lb1": 0.0, "lb1_lb2": 0.52, "lb2_lb3": -1.05,
            "base_rb1": 0.0, "rb1_rb2": 0.52, "rb2_rb3": -1.05,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "leg_joints": DCMotorCfg(
            joint_names_expr=["base_lf1", "base_rf1", "base_lb1", "base_rb1", 
                            "lf1_lf2", "rf1_rf2", "lb1_lb2", "rb1_rb2",
                            "lf2_lf3", "rf2_rf3", "lb2_lb3", "rb2_rb3"],
            saturation_effort=8.0, velocity_limit=2.0, stiffness=150.0,
            damping=4.0, friction=0.05, armature=0.001,
        ),
        "foot_joints": DCMotorCfg(
            joint_names_expr=["lf3_foot", "rf3_foot", "lb3_foot", "rb3_foot"],
            saturation_effort=1000.0, velocity_limit=0.1, stiffness=1000.0,
            damping=100.0, friction=0.4, armature=0.001,
        ),
        "plate_joints": DCMotorCfg(
            joint_names_expr=["lf1_plate", "rf1_plate", "lb1_plate", "rb1_plate",
                              "lf2_plate", "rf2_plate", "lb2_plate", "rb2_plate"],
            saturation_effort=1000.0, velocity_limit=0.1, stiffness=1000.0,
            damping=100.0, friction=0.1, armature=0.001,
        ),
        "sensor_joints": DCMotorCfg(
            joint_names_expr=["base_lidar", "imu_joint"],
            saturation_effort=1000.0, velocity_limit=0.01, stiffness=2000.0,
            damping=200.0, friction=0.1, armature=0.001,
        ),
    }
)

# Your existing scene config (unchanged)
class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Spot")

# MINIMAL GAIT CLASS - just the essentials!
class SimpleGait:
    """
    Minimal gait implementation using CHAMP knowledge.
    Only implements the core trot pattern.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        
        # CHAMP gait parameters (from your gait.yaml)
        self.stance_duration = 0.25  # seconds
        self.cycle_time = 0.5        # full cycle (2 Hz)
        self.swing_height = 0.04     # 4cm
        self.nominal_height = 0.09   # 20cm
        
        # Trot phase offsets (diagonal pairs)
        self.phase_offsets = {"lf": 0.0, "rf": 0.5, "lb": 0.5, "rb": 0.0}
        
        # Timing
        self.time = 0.0
        
        # Standing pose (your working values)
        self.standing_angles = {
            "base_lf1": 0.0, "lf1_lf2": 0.52, "lf2_lf3": -1.05,
            "base_rf1": 0.0, "rf1_rf2": 0.52, "rf2_rf3": -1.05,
            "base_lb1": 0.0, "lb1_lb2": 0.52, "lb2_lb3": -1.05,
            "base_rb1": 0.0, "rb1_rb2": 0.52, "rb2_rb3": -1.05,
        }
        
    def update(self, dt: float, forward_speed: float = 0.0):
        """Update gait timing and return joint targets."""
        self.time += dt
        
        if forward_speed < 0.01:
            # Standing - return your working standing pose
            return self._get_standing_targets()
        else:
            # Walking - apply simple gait
            return self._get_walking_targets(forward_speed)
    
    def _get_standing_targets(self):
        """Return standing pose (your proven working pose)."""
        return self.standing_angles.copy()
        
    def _get_walking_targets(self, speed: float):
        """Generate walking gait using CHAMP logic."""
        targets = self.standing_angles.copy()
        
        # Calculate stride length based on speed (like CHAMP)
        stride_length = speed * self.cycle_time  # Distance per cycle
        
        # DEBUG: Track what's happening
        global_phase = (self.time / self.cycle_time) % 1.0
        duty_cycle = self.stance_duration / self.cycle_time
        
        # DEBUG: Print every few updates
        if int(self.time * 100) % 50 == 0:  # Every 0.5 seconds
            print(f"[GAIT DEBUG] Time: {self.time:.2f}s, Phase: {global_phase:.2f}, Speed: {speed:.2f}")
        
        changes_made = False
        
        for leg in ["lf", "rf", "lb", "rb"]:
            # Get leg phase
            leg_phase = (global_phase + self.phase_offsets[leg]) % 1.0
            
            # Determine if in stance or swing
            if leg_phase < duty_cycle:
                # STANCE: foot on ground
                if int(self.time * 100) % 50 == 0:
                    print(f"  {leg}: STANCE (phase {leg_phase:.2f})")
            else:
                # SWING: lift the leg
                swing_progress = (leg_phase - duty_cycle) / (1.0 - duty_cycle)
                changes_made = True
                
                if int(self.time * 100) % 50 == 0:
                    print(f"  {leg}: SWING (phase {leg_phase:.2f}, progress {swing_progress:.2f})")
                
                # Make the swing more dramatic for debugging
                swing_amplitude = 0.5  # Increased from 0.2/0.3
                
                if leg == "lf" or leg == "rf":
                    # Front legs: lift by bending knee more
                    knee_offset = swing_amplitude * math.sin(swing_progress * math.pi)
                    ankle_offset = -swing_amplitude * 1.5 * math.sin(swing_progress * math.pi)
                    
                    targets[f"{leg}1_{leg}2"] += knee_offset
                    targets[f"{leg}2_{leg}3"] += ankle_offset
                    
                    if int(self.time * 100) % 50 == 0:
                        print(f"    {leg} offsets: knee +{knee_offset:.3f}, ankle +{ankle_offset:.3f}")
                else:
                    # Back legs: similar pattern
                    knee_offset = swing_amplitude * math.sin(swing_progress * math.pi)
                    ankle_offset = -swing_amplitude * 1.5 * math.sin(swing_progress * math.pi)
                    
                    targets[f"{leg}1_{leg}2"] += knee_offset
                    targets[f"{leg}2_{leg}3"] += ankle_offset
        
        if int(self.time * 100) % 50 == 0:
            print(f"[GAIT DEBUG] Changes made: {changes_made}")
            
        return targets

class AggressiveTrotGait:
    """
    RAMBUNCTIOUS gait that actually moves the robot!
    No more timid leg lifting - this pupper TROTS!
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        
        # More aggressive timing for visible motion
        self.stance_duration = 0.3   # Longer stance for power
        self.cycle_time = 0.8        # Slower for debugging (1.25 Hz)
        self.swing_height = 0.08     # DOUBLE the swing height!
        
        # Trot pattern: LF+RB together, RF+LB together  
        self.phase_offsets = {
            "lf": 0.0,   # left front
            "rb": 0.0,   # right back (diagonal pair)
            "rf": 0.5,   # right front  
            "lb": 0.5    # left back (diagonal pair)
        }
        
        self.time = 0.0
        
        # Base standing pose - but we'll modify this aggressively
        self.base_angles = {
            "base_lf1": 0.0, "base_rf1": 0.0, "base_lb1": 0.0, "base_rb1": 0.0,
            "lf1_lf2": 0.52, "rf1_rf2": 0.52, "lb1_lb2": 0.52, "rb1_rb2": 0.52,
            "lf2_lf3": -1.05, "rf2_rf3": -1.05, "lb2_lb3": -1.05, "rb2_rb3": -1.05,
            "lf3_foot": 0.0, "rf3_foot": 0.0, "lb3_foot": 0.0, "rb3_foot": 0.0,
            "lf1_plate": 0.0, "rf1_plate": 0.0, "lb1_plate": 0.0, "rb1_plate": 0.0,
            "lf2_plate": 0.0, "rf2_plate": 0.0, "lb2_plate": 0.0, "rb2_plate": 0.0,
            "base_lidar": 0.0, "imu_joint": 0.0
        }
        
    def update(self, dt: float, forward_speed: float = 0.0):
        self.time += dt
        
        if forward_speed < 0.01:
            return self._get_standing_targets()
        else:
            return self._get_aggressive_trot(forward_speed)
    
    def _get_standing_targets(self):
        return self.base_angles.copy()
    
    def _get_aggressive_trot(self, speed: float):
        """AGGRESSIVE trot that actually moves the robot!"""
        targets = self.base_angles.copy()
        
        # Calculate more aggressive stride
        stride_length = max(0.1, speed * self.cycle_time)  # Minimum stride
        
        global_phase = (self.time / self.cycle_time) % 1.0
        duty_cycle = self.stance_duration / self.cycle_time
        
        # Debug every second
        if int(self.time * 10) % 10 == 0:
            print(f"[AGGRESSIVE TROT] Time: {self.time:.1f}s, Phase: {global_phase:.2f}, Stride: {stride_length:.2f}m")
        
        swing_legs = []
        stance_legs = []
        
        for leg in ["lf", "rf", "lb", "rb"]:
            leg_phase = (global_phase + self.phase_offsets[leg]) % 1.0
            
            if leg_phase < duty_cycle:
                # STANCE PHASE - PUSH THE GROUND!
                stance_legs.append(leg)
                
                # Add forward lean and power during stance
                stance_progress = leg_phase / duty_cycle
                
                # Hip: sweep backward during stance (propulsion!)
                hip_sweep = -0.3 * stride_length * (stance_progress - 0.5)
                targets[f"base_{leg}1"] += hip_sweep
                
                # Knee: extend more during mid-stance for power
                knee_extension = -0.2 * math.sin(stance_progress * math.pi)
                targets[f"{leg}1_{leg}2"] += knee_extension
                
                # Ankle: push harder during stance
                ankle_push = 0.3 * math.sin(stance_progress * math.pi)
                targets[f"{leg}2_{leg}3"] += ankle_push
                
            else:
                # SWING PHASE - LIFT THAT LEG HIGH!
                swing_legs.append(leg)
                swing_progress = (leg_phase - duty_cycle) / (1.0 - duty_cycle)
                
                # AGGRESSIVE swing motion
                swing_factor = math.sin(swing_progress * math.pi)
                
                # Hip: swing forward during swing phase
                hip_reach = 0.4 * stride_length * (swing_progress - 0.5)
                targets[f"base_{leg}1"] += hip_reach
                
                # Knee: REALLY bend that knee during swing
                knee_lift = 0.8 * swing_factor  # Much more aggressive!
                targets[f"{leg}1_{leg}2"] += knee_lift
                
                # Ankle: Clear the ground dramatically
                ankle_clear = -1.0 * swing_factor  # VERY aggressive clearance
                targets[f"{leg}2_{leg}3"] += ankle_clear
        
        # Add body dynamics for even more movement
        body_sway = 0.1 * math.sin(global_phase * 2 * math.pi)  # Body sway
        
        # Lean forward slightly during locomotion
        forward_lean = 0.15
        for leg in ["lf", "rf", "lb", "rb"]:
            targets[f"{leg}1_{leg}2"] -= forward_lean
        
        # Add lateral weight shift (rock side to side)
        if "lf" in swing_legs or "lb" in swing_legs:  # Left side swinging
            # Shift weight to right
            targets["base_rf1"] -= 0.1
            targets["base_rb1"] -= 0.1
        else:  # Right side swinging
            # Shift weight to left  
            targets["base_lf1"] -= 0.1
            targets["base_lb1"] -= 0.1
        
        if int(self.time * 10) % 10 == 0:
            print(f"  Swing legs: {swing_legs}, Stance legs: {stance_legs}")
            
        return targets

class WorkingGait:
    """Final gait implementation with correct movement directions."""
    
    def __init__(self, device="cuda"):
        self.device = device
        
        # Conservative gait timing for stability
        self.stance_duration = 0.6   # Longer stance for stability
        self.cycle_time = 1.2        # Slower cycle (50% duty cycle)
        self.time = 0.0
        
        # Trot pattern: LF+RB together, RF+LB together
        self.phase_offsets = {
            "lf": 0.0,   # left front
            "rf": 0.5,   # right front  
            "lb": 0.5,   # left back
            "rb": 0.0    # right back
        }
        
        # Standing pose
        self.standing_angles = {
            "base_lb1": 0.0, "base_lf1": 0.0, "base_rb1": 0.0, "base_rf1": 0.0,
            "lb1_lb2": 0.52, "lf1_lf2": 0.52, "rb1_rb2": 0.52, "rf1_rf2": 0.52,
            "lb2_lb3": -1.05, "lf2_lf3": -1.05, "rb2_rb3": -1.05, "rf2_rf3": -1.05,
            "base_lidar": 0.0, "imu_joint": 0.0,
            "lb1_plate": 0.0, "lf1_plate": 0.0, "rb1_plate": 0.0, "rf1_plate": 0.0,
            "lb2_plate": 0.0, "lf2_plate": 0.0, "rb2_plate": 0.0, "rf2_plate": 0.0,
            "lb3_foot": 0.0, "lf3_foot": 0.0, "rb3_foot": 0.0, "rf3_foot": 0.0,
        }
        
    def update(self, dt: float, forward_speed: float = 0.0):
        self.time += dt
        
        if forward_speed < 0.01:
            return self._get_standing_targets()
        else:
            return self._get_walking_targets(forward_speed)
    
    def _get_standing_targets(self):
        return self.standing_angles.copy()
    
    def _get_walking_targets(self, speed: float):
        targets = self.standing_angles.copy()
        
        global_phase = (self.time / self.cycle_time) % 1.0
        duty_cycle = self.stance_duration / self.cycle_time
        
        # Debug every 0.4 seconds
        if int(self.time * 2.5) % 1 == 0:
            print(f"\n[WORKING GAIT] Time: {self.time:.1f}s, Phase: {global_phase:.2f}")
        
        legs_in_swing = []
        
        for leg in ["lf", "rf", "lb", "rb"]:
            leg_phase = (global_phase + self.phase_offsets[leg]) % 1.0
            
            if leg_phase < duty_cycle:
                # STANCE: Keep normal standing pose
                if int(self.time * 2.5) % 1 == 0:
                    print(f"  {leg.upper()}: STANCE")
            else:
                # SWING: Lift the leg with CORRECT directions
                swing_progress = (leg_phase - duty_cycle) / (1.0 - duty_cycle)
                swing_factor = math.sin(swing_progress * math.pi)
                
                legs_in_swing.append(leg.upper())
                
                # UNIFIED APPROACH: Same direction for all legs, but conservative amounts
                if leg in ["lf", "rf"]:  # FRONT LEGS
                    # Front legs: moderate movement (we know this works)
                    knee_lift = 0.4 * swing_factor      # Conservative lift
                    ankle_compensate = 0.2 * swing_factor   # Small ankle
                    
                    targets[f"{leg}1_{leg}2"] += knee_lift      
                    targets[f"{leg}2_{leg}3"] += ankle_compensate  
                    
                else:  # HIND LEGS (lb, rb)
                    # HIND LEGS: SAME DIRECTION but more conservative to avoid "going too far back"
                    knee_lift = 0.25 * swing_factor     # SMALLER knee lift to avoid extreme positions
                    ankle_compensate = 0.1 * swing_factor   # SMALLER ankle adjustment
                    
                    targets[f"{leg}1_{leg}2"] += knee_lift      # Same direction as front
                    targets[f"{leg}2_{leg}3"] += ankle_compensate  # Same direction as front
                
                if int(self.time * 2.5) % 1 == 0:
                    knee_val = targets[f"{leg}1_{leg}2"]
                    ankle_val = targets[f"{leg}2_{leg}3"]
                    print(f"  {leg.upper()}: SWING (progress {swing_progress:.2f}) - Knee {knee_val:.2f}, Ankle {ankle_val:.2f}")
        
        if int(self.time * 2.5) % 1 == 0:
            print(f"  --> Legs in swing: {legs_in_swing}")
            
        return targets
# First, let's add a comprehensive joint debugging function

# FIXED debug function that doesn't crash on tensor formatting

def debug_all_robot_info(scene):
    """Complete robot debug info - FIXED to handle PyTorch tensors properly."""
    print("\n" + "="*60)
    print("COMPLETE ROBOT DEBUG INFO")
    print("="*60)
    
    robot = scene["robot"]
    
    # Joint names and positions
    joint_names = robot.data.joint_names
    current_pos = robot.data.joint_pos[0]
    
    print(f"\nJoint Names ({len(joint_names)}):")
    for i, name in enumerate(joint_names):
        print(f"  {i:2d}: {name}")
    
    print(f"\nCurrent Joint Positions:")
    for i, (name, pos) in enumerate(zip(joint_names, current_pos)):
        # FIXED: Convert tensor to float before formatting
        pos_val = float(pos.item()) if hasattr(pos, 'item') else float(pos)
        pos_deg = pos_val * 180.0 / 3.14159
        print(f"  {i:2d}: {name:20s} = {pos_val:7.3f} rad ({pos_deg:6.1f}°)")
    
    # DOF names (if available)
    try:
        if hasattr(robot.data, 'joint_names'):
            print(f"\nDOF Names:")
            # DOF names are usually the same as joint names in Isaac Lab
            for i, name in enumerate(robot.data.joint_names):
                print(f"  {i:2d}: {name}")
    except Exception as e:
        print(f"\nDOF Names: Not available ({e})")
    
    # Body/Link names
    try:
        if hasattr(robot, 'data') and hasattr(robot.data, 'body_names'):
            body_names = robot.data.body_names
            print(f"\nBody/Link Names:")
            for i, name in enumerate(body_names):
                print(f"  {i:2d}: {name}")
    except Exception as e:
        print(f"\nBody/Link Names: Not available ({e})")
    
    # Joint limits - FIXED to handle tensors properly
    try:
        if hasattr(robot.data, 'soft_joint_pos_limits'):
            limits = robot.data.soft_joint_pos_limits[0]  # Get first environment
            print(f"\nJoint Position Limits:")
            for i, name in enumerate(joint_names):
                if i < len(limits):
                    # FIXED: Convert tensor elements to float properly
                    min_val = float(limits[i][0].item()) if hasattr(limits[i][0], 'item') else float(limits[i][0])
                    max_val = float(limits[i][1].item()) if hasattr(limits[i][1], 'item') else float(limits[i][1])
                    print(f"  {name:20s}: [{min_val:6.2f}, {max_val:6.2f}]")
    except Exception as e:
        print(f"\nJoint Position Limits: Not available ({e})")
    
    # Joint velocities
    try:
        if hasattr(robot.data, 'joint_vel'):
            current_vel = robot.data.joint_vel[0]
            print(f"\nCurrent Joint Velocities:")
            for i, (name, vel) in enumerate(zip(joint_names, current_vel)):
                # FIXED: Convert tensor to float
                vel_val = float(vel.item()) if hasattr(vel, 'item') else float(vel)
                print(f"  {i:2d}: {name:20s} = {vel_val:7.3f} rad/s")
    except Exception as e:
        print(f"\nJoint Velocities: Not available ({e})")
    
    # Root state
    try:
        if hasattr(robot.data, 'root_pos_w'):
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0] if hasattr(robot.data, 'root_quat_w') else None
            
            print(f"\nRoot State:")
            pos_x = float(root_pos[0].item()) if hasattr(root_pos[0], 'item') else float(root_pos[0])
            pos_y = float(root_pos[1].item()) if hasattr(root_pos[1], 'item') else float(root_pos[1])
            pos_z = float(root_pos[2].item()) if hasattr(root_pos[2], 'item') else float(root_pos[2])
            print(f"  Position: ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
            
            if root_quat is not None:
                qw = float(root_quat[0].item()) if hasattr(root_quat[0], 'item') else float(root_quat[0])
                qx = float(root_quat[1].item()) if hasattr(root_quat[1], 'item') else float(root_quat[1])
                qy = float(root_quat[2].item()) if hasattr(root_quat[2], 'item') else float(root_quat[2])
                qz = float(root_quat[3].item()) if hasattr(root_quat[3], 'item') else float(root_quat[3])
                print(f"  Orientation: ({qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f})")
    except Exception as e:
        print(f"\nRoot State: Not available ({e})")
    
    print("\n" + "="*60)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Modified simulator with optional gait."""
    print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Initialize gait (even if not used)
    gait = SimpleGait(device=sim.device)
    
    # Your existing standing pose setup
    desired_joint_angles_dict = {
        "base_lf1": 0.0, "lf1_lf2": 0.52, "lf2_lf3": -1.05,
        "base_rf1": 0.0, "rf1_rf2": 0.52, "rf2_rf3": -1.05,
        "base_lb1": 0.0, "lb1_lb2": 0.52, "lb2_lb3": -1.05,
        "base_rb1": 0.0, "rb1_rb2": 0.52, "rb2_rb3": -1.05,
        "lf3_foot": 0.0, "rf3_foot": 0.0, "lb3_foot": 0.0, "rb3_foot": 0.0,
        "lf1_plate": 0.0, "rf1_plate": 0.0, "lb1_plate": 0.0, "rb1_plate": 0.0,
        "lf2_plate": 0.0, "rf2_plate": 0.0, "lb2_plate": 0.0, "rb2_plate": 0.0,
        "base_lidar": 0.0, "imu_joint": 0.0
    }
    
    def create_joint_tensor(angle_dict):
        """Convert angle dictionary to joint tensor."""
        target_tensor = torch.zeros(len(scene["robot"].data.joint_names), device=sim.device)
        for i, joint_name in enumerate(scene["robot"].data.joint_names):
            if joint_name in angle_dict:
                target_tensor[i] = angle_dict[joint_name]
        return target_tensor
    
    def create_joint_tensor_with_boost(angle_dict, scene, boost_factor=1.0):
        """Convert angle dictionary to joint tensor with movement boost."""
        target_tensor = torch.zeros(len(scene["robot"].data.joint_names), device=scene.device)
        
        for i, joint_name in enumerate(scene["robot"].data.joint_names):
            if joint_name in angle_dict:
                base_angle = angle_dict[joint_name]
                
                # Boost leg movements for more dramatic motion
                if any(x in joint_name for x in ["lf1_lf2", "rf1_rf2", "lb1_lb2", "rb1_rb2", 
                                            "lf2_lf3", "rf2_rf3", "lb2_lb3", "rb2_rb3"]):
                    target_tensor[i] = base_angle * boost_factor
                else:
                    target_tensor[i] = base_angle
                    
        return target_tensor    
    
    default_joint_pos = create_joint_tensor(desired_joint_angles_dict)
    settling_steps = 100
    
    print(f"[INFO] Gait enabled: {args_cli.enable_gait}")
    print(f"[INFO] Gait speed: {args_cli.gait_speed} m/s")
    
    while simulation_app.is_running():
        # Periodic reset (your existing logic)
        if count % 1000 == 0:
            count = 0
            print("[INFO]: Resetting Mini Pupper state...")
            
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 2] = 0.09  # CHAMP height
            
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene["robot"].write_joint_state_to_sim(default_joint_pos, scene["robot"].data.default_joint_vel.clone())
            scene.reset()
            
            # Reset gait timing
            gait.time = 0.0
        
        # MAIN CHANGE: Choose between standing and gait
        # if args_cli.enable_gait and count > settling_steps:
        #     # Use gait
        #     current_speed = args_cli.gait_speed
        #     joint_targets_dict = gait.update(sim_dt, current_speed)
        #     target_pos = create_joint_tensor(joint_targets_dict).unsqueeze(0)
        # else:
        #     # Use standing (your original logic)
        #     if count < settling_steps:
        #         alpha = count / settling_steps
        #         current_pos = scene["robot"].data.joint_pos[0]
        #         target_pos = alpha * default_joint_pos + (1 - alpha) * current_pos
        #     else:
        #         target_pos = default_joint_pos
        
        # # Apply targets (your existing logic)
        # scene["robot"].set_joint_position_target(target_pos)
        if args_cli.enable_gait and count > settling_steps:
            # Use gait
            current_speed = args_cli.gait_speed
            joint_targets_dict = gait.update(sim_dt, current_speed)
            target_pos = create_joint_tensor_with_boost(joint_targets_dict, scene, boost_factor=5.0)
            # target_pos = create_joint_tensor(joint_targets_dict)  # Remove .unsqueeze(0)
        else:
            # Use standing (your original logic)
            if count < settling_steps:
                alpha = count / settling_steps
                current_pos = scene["robot"].data.joint_pos[0]
                target_pos = alpha * default_joint_pos + (1 - alpha) * current_pos
            else:
                target_pos = default_joint_pos

        # Keep target_pos as 1D, then expand when needed:
        scene["robot"].set_joint_position_target(target_pos.unsqueeze(0))        
        # Step simulation (your existing logic)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Debug output
        if count % 200 == 0:  # Every 200 steps
            root_pos = scene["robot"].data.root_pos_w[0]
            if args_cli.enable_gait:
                phase = (gait.time / gait.cycle_time) % 1.0
                print(f"[DEBUG] Step {count}: Pos=({root_pos[0]:.2f},{root_pos[1]:.2f},{root_pos[2]:.2f}), Phase={phase:.2f}")
            else:
                print(f"[DEBUG] Step {count}: Standing at ({root_pos[0]:.2f},{root_pos[1]:.2f},{root_pos[2]:.2f})")
                
def run_final_gait_test(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Test the final working gait."""
    print("[FINAL GAIT] Testing working gait with correct directions...")
    
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Use the working gait
    gait = WorkingGait(device=sim.device)  # Or VeryConservativeGait for safer testing
    
    # Joint mapping
    joint_order = [
        'base_lb1', 'base_lf1', 'base_lidar', 'base_rb1', 'base_rf1', 'imu_joint',
        'lb1_lb2', 'lb1_plate', 'lf1_lf2', 'lf1_plate', 'rb1_plate', 'rb1_rb2', 
        'rf1_plate', 'rf1_rf2', 'lb2_lb3', 'lb2_plate', 'lf2_lf3', 'lf2_plate', 
        'rb2_plate', 'rb2_rb3', 'rf2_plate', 'rf2_rf3', 'lb3_foot', 'lf3_foot', 
        'rb3_foot', 'rf3_foot'
    ]
    
    def dict_to_tensor(angle_dict):
        tensor = torch.zeros(len(joint_order), device=sim.device)
        for i, joint_name in enumerate(joint_order):
            if joint_name in angle_dict:
                tensor[i] = angle_dict[joint_name]
        return tensor
    
    standing_dict = gait.standing_angles
    standing_tensor = dict_to_tensor(standing_dict)
    
    print(f"[FINAL GAIT] Conservative gait enabled")
    print(f"[FINAL GAIT] Cycle time: {gait.cycle_time}s, Duty cycle: {gait.stance_duration/gait.cycle_time:.1%}")
    
    settling_steps = 200
    
    while simulation_app.is_running():
        # Reset periodically
        if count % 2000 == 0:
            count = 0
            print(f"\n[RESET] Resetting robot")
            
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 2] = 0.09  
            
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene["robot"].write_joint_state_to_sim(
                standing_tensor.unsqueeze(0), 
                torch.zeros_like(standing_tensor).unsqueeze(0)
            )
            
            scene.reset()
            gait.time = 0.0
        
        # Control logic
        if count < settling_steps:
            # Settling phase
            alpha = count / settling_steps
            current_pos = scene["robot"].data.joint_pos[0]
            target_tensor = alpha * standing_tensor + (1 - alpha) * current_pos
                
        elif args_cli.enable_gait:
            # FINAL GAIT
            gait_targets_dict = gait.update(sim_dt, args_cli.gait_speed)
            target_tensor = dict_to_tensor(gait_targets_dict)
            
        else:
            # Standing mode
            target_tensor = standing_tensor
        
        # Apply targets
        scene["robot"].set_joint_position_target(target_tensor.unsqueeze(0))
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Status reporting
        if count % 300 == 0:  # Every 1.2 seconds
            root_pos = scene["robot"].data.root_pos_w[0]
            print(f"\n[STATUS] Step {count}: Pos ({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f})")
            
            if args_cli.enable_gait and count > settling_steps:
                phase = (gait.time / gait.cycle_time) % 1.0
                print(f"[STATUS] Gait phase: {phase:.2f}")
                
                # Expected active legs
                duty_cycle = gait.stance_duration / gait.cycle_time
                active_legs = []
                for leg in ["lf", "rf", "lb", "rb"]:
                    leg_phase = (phase + gait.phase_offsets[leg]) % 1.0
                    if leg_phase >= duty_cycle:
                        active_legs.append(leg.upper())
                print(f"[STATUS] Should be moving: {active_legs}")

# FIXED debug function that doesn't crash on tensor formatting

def debug_all_robot_info(scene):
    """Complete robot debug info - FIXED to handle PyTorch tensors properly."""
    print("\n" + "="*60)
    print("COMPLETE ROBOT DEBUG INFO")
    print("="*60)
    
    robot = scene["robot"]
    
    # Joint names and positions
    joint_names = robot.data.joint_names
    current_pos = robot.data.joint_pos[0]
    
    print(f"\nJoint Names ({len(joint_names)}):")
    for i, name in enumerate(joint_names):
        print(f"  {i:2d}: {name}")
    
    print(f"\nCurrent Joint Positions:")
    for i, (name, pos) in enumerate(zip(joint_names, current_pos)):
        # FIXED: Convert tensor to float before formatting
        pos_val = float(pos.item()) if hasattr(pos, 'item') else float(pos)
        pos_deg = pos_val * 180.0 / 3.14159
        print(f"  {i:2d}: {name:20s} = {pos_val:7.3f} rad ({pos_deg:6.1f}°)")
    
    # DOF names (if available)
    try:
        if hasattr(robot.data, 'joint_names'):
            print(f"\nDOF Names:")
            # DOF names are usually the same as joint names in Isaac Lab
            for i, name in enumerate(robot.data.joint_names):
                print(f"  {i:2d}: {name}")
    except Exception as e:
        print(f"\nDOF Names: Not available ({e})")
    
    # Body/Link names
    try:
        if hasattr(robot, 'data') and hasattr(robot.data, 'body_names'):
            body_names = robot.data.body_names
            print(f"\nBody/Link Names:")
            for i, name in enumerate(body_names):
                print(f"  {i:2d}: {name}")
    except Exception as e:
        print(f"\nBody/Link Names: Not available ({e})")
    
    # Joint limits - FIXED to handle tensors properly
    try:
        if hasattr(robot.data, 'soft_joint_pos_limits'):
            limits = robot.data.soft_joint_pos_limits[0]  # Get first environment
            print(f"\nJoint Position Limits:")
            for i, name in enumerate(joint_names):
                if i < len(limits):
                    # FIXED: Convert tensor elements to float properly
                    min_val = float(limits[i][0].item()) if hasattr(limits[i][0], 'item') else float(limits[i][0])
                    max_val = float(limits[i][1].item()) if hasattr(limits[i][1], 'item') else float(limits[i][1])
                    print(f"  {name:20s}: [{min_val:6.2f}, {max_val:6.2f}]")
    except Exception as e:
        print(f"\nJoint Position Limits: Not available ({e})")
    
    # Joint velocities
    try:
        if hasattr(robot.data, 'joint_vel'):
            current_vel = robot.data.joint_vel[0]
            print(f"\nCurrent Joint Velocities:")
            for i, (name, vel) in enumerate(zip(joint_names, current_vel)):
                # FIXED: Convert tensor to float
                vel_val = float(vel.item()) if hasattr(vel, 'item') else float(vel)
                print(f"  {i:2d}: {name:20s} = {vel_val:7.3f} rad/s")
    except Exception as e:
        print(f"\nJoint Velocities: Not available ({e})")
    
    # Root state
    try:
        if hasattr(robot.data, 'root_pos_w'):
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0] if hasattr(robot.data, 'root_quat_w') else None
            
            print(f"\nRoot State:")
            pos_x = float(root_pos[0].item()) if hasattr(root_pos[0], 'item') else float(root_pos[0])
            pos_y = float(root_pos[1].item()) if hasattr(root_pos[1], 'item') else float(root_pos[1])
            pos_z = float(root_pos[2].item()) if hasattr(root_pos[2], 'item') else float(root_pos[2])
            print(f"  Position: ({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
            
            if root_quat is not None:
                qw = float(root_quat[0].item()) if hasattr(root_quat[0], 'item') else float(root_quat[0])
                qx = float(root_quat[1].item()) if hasattr(root_quat[1], 'item') else float(root_quat[1])
                qy = float(root_quat[2].item()) if hasattr(root_quat[2], 'item') else float(root_quat[2])
                qz = float(root_quat[3].item()) if hasattr(root_quat[3], 'item') else float(root_quat[3])
                print(f"  Orientation: ({qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f})")
    except Exception as e:
        print(f"\nRoot State: Not available ({e})")
    
    print("\n" + "="*60)

def debug_actuator_config(scene):
    """Debug actuator configuration."""
    print("\n[ACTUATOR DEBUG] Checking actuator groups...")
    
    robot = scene["robot"]
    
    # Check if we can access actuator info
    try:
        if hasattr(robot, 'actuators'):
            print("Available actuator groups:")
            for name, actuator in robot.actuators.items():
                print(f"  {name}: {type(actuator)}")
                if hasattr(actuator, 'joint_names_expr'):
                    print(f"    Joint expression: {actuator.joint_names_expr}")
        else:
            print("No actuator info available")
            
    except Exception as e:
        print(f"Actuator debug failed: {e}")

def test_individual_legs(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Test each leg individually to diagnose the issue."""
    print("[LEG TEST] Testing individual leg control...")
    
    sim_dt = sim.get_physics_dt()
    count = 0
    test_phase = 0  # 0=settle, 1=LF, 2=RF, 3=LB, 4=RB
    phase_duration = 500  # steps per phase
    
    # Joint indices based on your debug output
    joint_indices = {
        'base_lb1': 0, 'base_lf1': 1, 'base_rb1': 3, 'base_rf1': 4,
        'lb1_lb2': 6, 'lf1_lf2': 8, 'rb1_rb2': 11, 'rf1_rf2': 13,
        'lb2_lb3': 14, 'lf2_lf3': 16, 'rb2_rb3': 19, 'rf2_rf3': 21
    }
    
    # Base standing pose (as tensor indices)
    base_pose = torch.zeros(26, device=sim.device)
    base_pose[6] = 0.52   # lb1_lb2 (LB knee)
    base_pose[8] = 0.52   # lf1_lf2 (LF knee)  
    base_pose[11] = 0.52  # rb1_rb2 (RB knee)
    base_pose[13] = 0.52  # rf1_rf2 (RF knee)
    base_pose[14] = -1.05 # lb2_lb3 (LB ankle)
    base_pose[16] = -1.05 # lf2_lf3 (LF ankle)
    base_pose[19] = -1.05 # rb2_rb3 (RB ankle)
    base_pose[21] = -1.05 # rf2_rf3 (RF ankle)
    
    while simulation_app.is_running() and count < 2500:  # 5 phases * 500 steps
        
        # Phase transitions
        if count % phase_duration == 0:
            test_phase = count // phase_duration
            print(f"\n[LEG TEST] === PHASE {test_phase} ===")
            
            # Reset robot
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 2] = 0.09
            
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene["robot"].write_joint_state_to_sim(
                base_pose.unsqueeze(0), 
                torch.zeros_like(base_pose).unsqueeze(0)
            )
            scene.reset()
        
        # Create target pose
        target_pose = base_pose.clone()
        
        if test_phase == 0:
            print(f"[LEG TEST] SETTLING - All legs in standing pose") if count % 100 == 0 else None
            # Keep base pose
            
        elif test_phase == 1:
            print(f"[LEG TEST] TESTING LEFT FRONT (LF) LEG") if count % 100 == 0 else None
            # Lift left front leg dramatically
            target_pose[8] += 0.8   # lf1_lf2 knee bend
            target_pose[16] += 0.4  # lf2_lf3 ankle compensate
            
        elif test_phase == 2:
            print(f"[LEG TEST] TESTING RIGHT FRONT (RF) LEG") if count % 100 == 0 else None
            # Lift right front leg
            target_pose[13] += 0.8  # rf1_rf2 knee bend
            target_pose[21] += 0.4  # rf2_rf3 ankle compensate
            
        elif test_phase == 3:
            print(f"[LEG TEST] TESTING LEFT BACK (LB) LEG") if count % 100 == 0 else None
            # Lift left back leg
            target_pose[6] += 0.8   # lb1_lb2 knee bend  *** KEY TEST ***
            target_pose[14] += 0.4  # lb2_lb3 ankle compensate
            
        elif test_phase == 4:
            print(f"[LEG TEST] TESTING RIGHT BACK (RB) LEG") if count % 100 == 0 else None
            # Lift right back leg
            target_pose[11] += 0.8  # rb1_rb2 knee bend  *** KEY TEST ***
            target_pose[19] += 0.4  # rb2_rb3 ankle compensate
        
        # Apply targets
        scene["robot"].set_joint_position_target(target_pose.unsqueeze(0))
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Detailed reporting
        if count % 100 == 0:
            root_pos = scene["robot"].data.root_pos_w[0]
            current_joints = scene["robot"].data.joint_pos[0]
            
            print(f"[LEG TEST] Step {count}: Height {root_pos[2]:.3f}m")
            
            # Show actual vs target for key joints
            key_joints = [
                ("LF_knee", 8, target_pose[8], current_joints[8]),
                ("RF_knee", 13, target_pose[13], current_joints[13]), 
                ("LB_knee", 6, target_pose[6], current_joints[6]),
                ("RB_knee", 11, target_pose[11], current_joints[11])
            ]
            
            for name, idx, target, actual in key_joints:
                error = abs(target - actual)
                status = "✅" if error < 0.1 else "❌"
                print(f"  {name}: target={target:.2f}, actual={actual:.2f}, error={error:.2f} {status}")
        
        # End of phase summary
        if (count + 1) % phase_duration == 0 and test_phase > 0:
            current_joints = scene["robot"].data.joint_pos[0]
            root_pos = scene["robot"].data.root_pos_w[0]
            
            print(f"\n[LEG TEST] === PHASE {test_phase} SUMMARY ===")
            print(f"Final height: {root_pos[2]:.3f}m")
            
            if test_phase == 1:  # LF test
                knee_reached = abs(current_joints[8] - (0.52 + 0.8)) < 0.2
                print(f"LF leg lift: {'SUCCESS' if knee_reached else 'FAILED'}")
                
            elif test_phase == 2:  # RF test  
                knee_reached = abs(current_joints[13] - (0.52 + 0.8)) < 0.2
                print(f"RF leg lift: {'SUCCESS' if knee_reached else 'FAILED'}")
                
            elif test_phase == 3:  # LB test
                knee_reached = abs(current_joints[6] - (0.52 + 0.8)) < 0.2
                print(f"LB leg lift: {'SUCCESS' if knee_reached else 'FAILED'}")
                if not knee_reached:
                    print("*** HIND LEG CONTROL ISSUE CONFIRMED ***")
                
            elif test_phase == 4:  # RB test
                knee_reached = abs(current_joints[11] - (0.52 + 0.8)) < 0.2  
                print(f"RB leg lift: {'SUCCESS' if knee_reached else 'FAILED'}")
                if not knee_reached:
                    print("*** HIND LEG CONTROL ISSUE CONFIRMED ***")
    
    print("\n[LEG TEST] Individual leg test complete!")
    print("If front legs worked but hind legs failed, check:")
    print("1. Joint name mapping for hind legs")
    print("2. Actuator configuration for hind leg joints") 
    print("3. Joint limits or motor saturation")

def test_hind_leg_directions(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Test different directions for hind leg movement."""
    print("[DIRECTION TEST] Testing hind leg movement directions...")
    
    sim_dt = sim.get_physics_dt()
    count = 0
    test_phase = 0  # 0=settle, 1=+knee, 2=-knee, 3=+ankle, 4=-ankle
    phase_duration = 300
    
    # Base standing pose
    base_pose = torch.zeros(26, device=sim.device)
    base_pose[6] = 0.52   # lb1_lb2
    base_pose[8] = 0.52   # lf1_lf2  
    base_pose[11] = 0.52  # rb1_rb2
    base_pose[13] = 0.52  # rf1_rf2
    base_pose[14] = -1.05 # lb2_lb3
    base_pose[16] = -1.05 # lf2_lf3
    base_pose[19] = -1.05 # rb2_rb3
    base_pose[21] = -1.05 # rf2_rf3
    
    while simulation_app.is_running() and count < 1500:  # 5 phases * 300 steps
        
        if count % phase_duration == 0:
            test_phase = count // phase_duration
            print(f"\n[DIRECTION TEST] === PHASE {test_phase} ===")
            
            # Reset robot
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 2] = 0.09
            
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene["robot"].write_joint_state_to_sim(
                base_pose.unsqueeze(0), 
                torch.zeros_like(base_pose).unsqueeze(0)
            )
            scene.reset()
        
        # Create target pose
        target_pose = base_pose.clone()
        
        if test_phase == 0:
            print(f"[DIRECTION TEST] SETTLING") if count % 100 == 0 else None
            
        elif test_phase == 1:
            print(f"[DIRECTION TEST] HIND LEGS: +0.5 KNEE BEND") if count % 100 == 0 else None
            target_pose[6] += 0.5   # lb1_lb2 +knee
            target_pose[11] += 0.5  # rb1_rb2 +knee
            
        elif test_phase == 2:
            print(f"[DIRECTION TEST] HIND LEGS: -0.5 KNEE BEND") if count % 100 == 0 else None
            target_pose[6] -= 0.5   # lb1_lb2 -knee
            target_pose[11] -= 0.5  # rb1_rb2 -knee
            
        elif test_phase == 3:
            print(f"[DIRECTION TEST] HIND LEGS: +0.5 ANKLE BEND") if count % 100 == 0 else None
            target_pose[14] += 0.5  # lb2_lb3 +ankle
            target_pose[19] += 0.5  # rb2_rb3 +ankle
            
        elif test_phase == 4:
            print(f"[DIRECTION TEST] HIND LEGS: -0.5 ANKLE BEND") if count % 100 == 0 else None
            target_pose[14] -= 0.5  # lb2_lb3 -ankle
            target_pose[19] -= 0.5  # rb2_rb3 -ankle
        
        # Apply targets
        scene["robot"].set_joint_position_target(target_pose.unsqueeze(0))
        
        # Step simulation
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Report
        if count % 100 == 0:
            root_pos = scene["robot"].data.root_pos_w[0]
            print(f"[DIRECTION TEST] Step {count}: Height {root_pos[2]:.3f}m")
            
            if test_phase > 0:
                current_joints = scene["robot"].data.joint_pos[0]
                lb_knee = current_joints[6]
                rb_knee = current_joints[11]
                lb_ankle = current_joints[14]
                rb_ankle = current_joints[19]
                
                print(f"  LB: knee={lb_knee:.2f}, ankle={lb_ankle:.2f}")
                print(f"  RB: knee={rb_knee:.2f}, ankle={rb_ankle:.2f}")
                
                # Visual assessment
                if test_phase in [1, 2]:  # Knee tests
                    print("  --> Do the hind legs look LIFTED or LOWERED?")
                elif test_phase in [3, 4]:  # Ankle tests  
                    print("  --> Do the hind feet point UP or DOWN?")
    
    print("\n[DIRECTION TEST] Complete!")
    print("Observe which direction actually LIFTS the hind legs off the ground.")

# Your existing main function (unchanged)
def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=0.004,  # 250Hz
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
    
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    
    print("[INFO]: Setup complete...")
    if args_cli.enable_gait:
        print(f"[INFO]: Simple gait enabled at {args_cli.gait_speed} m/s")
    else:
        print("[INFO]: Standing mode (use --enable_gait to test walking)")
    
    # run_simulator(sim, scene)
    # run_simulator_with_debug(sim, scene)
    # test_individual_legs(sim, scene)
    # debug_actuator_config(scene)
    # test_hind_leg_directions(sim, scene) 
    run_final_gait_test(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
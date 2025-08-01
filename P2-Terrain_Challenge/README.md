## Project Overview

This project documents the complete development process of training a Mini Pupper quadruped robot to walk using Reinforcement Learning (RL) in Isaac Lab. What started as adapting existing Spot robot configurations evolved into a month-long deep dive into reward engineering, robot scaling, joint control, and gait dynamics.

## Key Achievements

- ✅ **Successful quadruped locomotion** from scratch using PPO
- ✅ **Proper robot scaling** (Mini Pupper vs Boston Dynamics Spot)
- ✅ **Diagonal gait coordination** with foot clearance
- ✅ **Bidirectional training** (forward and backward movement)
- ✅ **Physics-accurate simulation** with proper joint control
- ✅ **Systematic debugging methodology** for RL locomotion

## Robot Specifications

- **Mini Pupper Dimensions**: 133mm tall (5.25"), 560g total weight
- **12 Controllable Joints**: 4 hips, 4 knees, 4 ankles
- **Additional Fixed Joints**: Sensors (lidar, IMU), decorative plates, foot contacts
- **Target Speed**: Up to 1.5 m/s (11.3 body lengths/second)

## Major Technical Challenges Solved

### 1. Joint Ordering and Action Space Crisis

**Problem**: Initial training showed chaotic movement and joint order mismatches
❌ Ordering mismatch detected
 idx | live       | expected
---------------------------------
   2 | base_lidar  <| base_rb1
   3 | base_rb1    <| base_rf1


**Root Cause**: Joint ordering in actuator configuration didn't match Isaac Lab's discovery order

**Solution**: 
- Implemented targeted joint control (12 leg joints only)
- Excluded sensor and decorative joints from action space
- Matched joint ordering to robot's actual sequence

### 2. Robot Scale Mismatch (Critical Discovery)

**Problem**: Robot exhibited slow, awkward movement patterns

**Root Cause Analysis**:
- **Boston Dynamics Spot**: 840mm tall, 75kg
- **Mini Pupper**: 133mm tall, 0.56kg  
- **Scale Factor**: 6.3x size difference, 134x weight difference

**Parameter Scaling Applied**:
python
# Velocity Commands (scaled for body size)
# Spot: lin_vel_x=(-2.0, 3.0)
# Mini Pupper: lin_vel_x=(-0.8, 1.2)

# Foot Clearance (scaled for robot height)  
# Spot: target_height=0.1 (100mm for 840mm robot)
# Mini Pupper: target_height=0.015 (15mm for 133mm robot)

# Mass Randomization (scaled for weight)
# Spot: mass_distribution_params=(-2.5, 2.5)
# Mini Pupper: mass_distribution_params=(-0.05, 0.05)


### 3. Reward Engineering Evolution

**Initial Problems**:
- **Crutch walking**: Robot kept one foot planted, lifted others
- **Marching in place**: Proper lifting but no forward translation  
- **Asymmetric gaits**: One diagonal pair confident, other hesitant
- **Stretching behavior**: Static leg extension instead of dynamic stepping

**Reward Structure Iterations**:

#### Phase 1: Aggressive Foot Clearance
python
foot_clearance = RewardTermCfg(weight=5.0, target_height=0.035)
gait = RewardTermCfg(weight=20.0)

**Result**: Led to reward hacking and crutch walking

#### Phase 2: Individual Foot Control  
python
# Custom function forcing ALL feet to lift
foot_clearance = RewardTermCfg(
    func=spot_mdp.individual_foot_clearance_reward,
    weight=5.0  # Uses torch.min() across all feet
)

**Result**: Eliminated crutch walking but caused stretching

#### Phase 3: Balanced Approach (Final Solution)
python
base_linear_velocity = RewardTermCfg(weight=10.0)  # Forward motion priority
foot_clearance = RewardTermCfg(weight=2.0)        # Reasonable lifting
gait = RewardTermCfg(weight=10.0)                 # Diagonal coordination  
foot_slip = RewardTermCfg(weight=-2.0)            # Prevent dragging


### 4. Gait Frequency Control Discovery

**Key Finding**: The air_time reward controls gait frequency through mode_time parameter

python
air_time = RewardTermCfg(
    params={
        "mode_time": 0.165,  # Controls step timing
        # 0.3 = 1.67 Hz (too slow)
        # 0.165 = 3.0 Hz (optimal for Mini Pupper)
    }
)


**Frequency Calculation**: Full gait cycle = mode_time × 2, Frequency = 1 / (mode_time × 2)

### 5. Bidirectional Training Implementation

**Problem**: Forward-only training led to asymmetric gaits and compensation patterns

**Solution**: Implemented bidirectional velocity commands
python
ranges=mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-0.8, 1.2),  # Both forward and backward
    lin_vel_y=(-0.3, 0.3),  # Lateral movement
    ang_vel_z=(-1.0, 1.0)   # Rotational movement
)


**Result**: More natural, symmetric quadruped gaits

### 6. Physics Parameter Optimization

**Actuator Configuration for Mini Pupper**:
python
actuators={
    "leg_joints": DCMotorCfg(
        joint_names_expr=[12 specific leg joints],
        saturation_effort=2.5,    # Appropriate for small servos
        velocity_limit=1.5,       # Realistic for Mini Pupper
        stiffness=50.0,          # Balanced control
        damping=7.0,             # Stable dynamics
    ),
    # Separate configs for foot, plate, and sensor joints
}


## Training Methodology

### Environment Configuration
- **Terrain**: 95% flat, 5% rough (focus on basic locomotion)
- **Episode Length**: 20 seconds
- **Simulation Frequency**: 500 Hz physics, 50 Hz control
- **Training Environments**: 8000+ parallel environments

### Observation Space (76 dimensions)
- Base linear/angular velocity (6)
- Projected gravity (3) 
- Velocity commands (3)
- Joint positions (12)
- Joint velocities (12)
- Previous actions (12)
- Additional sensor data (~28)

### Action Space Evolution
- **Initial**: 26 joints (including sensors, plates)
- **Final**: 12 joints (only leg actuators)
- **Scale**: 0.2 (±11 degrees from default pose)

## Key Learning Insights

### 1. Entropy Loss as Learning Indicator
- **High entropy (>10)**: Random, chaotic movement
- **Medium entropy (1-5)**: Learning coordination
- **Low entropy (~0.2)**: Confident, refined patterns
- **Negative entropy**: Over-confidence, potential local optima

### 2. Reward Balance Critical
- **Individual foot clearance too strong** → Static stretching
- **Velocity reward too weak** → Marching in place  
- **Joint penalties too high** → Conservative movement
- **Optimal balance**: Forward motion > Gait > Clearance

### 3. Scale Matters Enormously
- **Velocity commands must scale with body size**
- **Foot clearance must scale with robot height**
- **Mass properties must scale with robot weight**
- **Gait frequency naturally scales with leg length**

### 4. Joint vs Link Naming
- **Joints**: Controllable DoF for actions and penalties
- **Links**: Physical bodies for contact sensors and collisions
- **Critical distinction** for proper reward configuration

## Debugging Tools Developed

### Joint Order Verification
python
def check_joint_order(scene, cfg):
    """Verify actuator joint ordering matches robot"""
    # Compares expected vs actual joint sequences
    # Identifies mismatches and suggests corrections


### Comprehensive Joint Analysis  
python
def debug_all_joint_info(scene, cfg):
    """Complete joint mapping analysis"""
    # Reports matched/unmatched joints
    # Suggests actuator group patterns
    # Validates configuration completeness


### Training Progress Monitoring
- **Action noise convergence** (target: 0.20)
- **Reward component balance** 
- **Episode length trends**
- **Gait quality metrics**

## Results and Performance

### Training Metrics (Final Policy)
- **Mean Reward**: 236+ (stable)
- **Episode Length**: 744+ steps (good survival)
- **Action Noise**: 0.20 (converged)
- **Gait Reward**: 6.0+ (excellent coordination)
- **Foot Clearance**: 0.38 (proper lifting)
- **Base Linear Velocity**: 2.4+ (good forward motion)
- **Body Contact Termination**: <3% (very stable)

### Locomotion Quality
- **Diagonal gait coordination** (proper trotting)
- **Bidirectional movement** (forward/backward)
- **Appropriate speed** (1.0-1.5 m/s)
- **Stable posture** (minimal tilting)
- **Energy efficient** (natural joint angles)

### Scale-Appropriate Performance
- **Speed**: 1.5 m/s = 11.3 body lengths/second (very fast for size)
- **Gait frequency**: ~3 Hz (appropriate for small quadruped)
- **Movement quality**: Appears "subtle" in Spot-scale environment (correct!)

## Code Architecture

### Key Configuration Files
- custom_quad.py: Mini Pupper robot definition and actuators
- spot_flat_env_cfg.py: Training environment and reward configuration
- mdp.py: Custom reward functions and locomotion primitives

# Mini Pupper Quadruped Locomotion with Reinforcement Learning

A subproject documenting the development of training a Mini Pupper quadruped robot to walk using Reinforcement Learning (RSL RL PPO) in Isaac Lab.
** Update : ** We have upgraded to the Mini Pupper 2 which has servo feebdack and and IMU sensor, enabling the robot with proprioception and oritenation w.r.t. gravity! New training and demo results incoming.

## Project Overview

This work demonstrates successful quadruped locomotion training by scaling down Boston Dynamics Spot robot configurations to work with the significantly smaller Mini Pupper platform. The key insight was recognizing that robot scale affects every aspect of locomotion - from velocity commands to joint constraints - requiring systematic parameter adaptation rather than direct configuration transfer.

### Boston Dynamics Spot vs Mini Pupper Comparison

The fundamental challenge of my project stemmed from the dramatic scale difference between the two platforms:

| Specification | Boston Dynamics Spot | Mini Pupper | Scale Factor |
|---------------|---------------------|-------------|--------------|
| **Height** | 840mm | 133mm | **6.3x smaller** |
| **Weight** | 75kg | 0.56kg | **134x lighter** |
| **Leg Reach** | ~400mm | ~80mm | **5x smaller** |
| **Payload** | 14kg | ~0.1kg | **140x smaller** |
| **Target Speed** | 1.6 m/s | 1.5 m/s | **Similar (11.3 body lengths/s)** |

<p align="center">
  <img src="https://raw.githubusercontent.com/baccuslab/SpotDMouse/main/P2-Terrain_Challenge/spotdiagram.png" alt="Boston Dynamics Spot anatomy" width="45%">
  <img src="https://raw.githubusercontent.com/baccuslab/SpotDMouse/main/P2-Terrain_Challenge/minipupperdiagram.png" alt="Mini Pupper anatomy" width="45%">
</p>
Figure 1: Anatomical comparison between Boston Dynamics Spot (left) and Mini Pupper (right), showing similar joint structure but vastly different scales.
<p align="center">
  <img src="https://raw.githubusercontent.com/baccuslab/SpotDMouse/main/P2-Terrain_Challenge/minipuppersimforward.png" alt="Mini Pupper Forward Simulation" width="45%">
  <img src="https://raw.githubusercontent.com/baccuslab/SpotDMouse/main/P2-Terrain_Challenge/minipuppersimbackward.png" alt="Mini Pupper Backward Simulation" width="45%">
</p>
Figure 2: Training simulation screenshots showing Mini Pupper forward locomotion (left) and backward locomotion (right), with velocity command visualizations.

## Training Results

The following videos demonstrate the successful locomotion training results for omnidirectional commands:

**Reccomendation:** Play the following videos at .25 speed to see the definition and quality of the gait.

### Forward Locomotion
[ **View Forward Locomotion Video**](https://drive.google.com/file/d/1pIDoCVo_h6PD33hEZZ0mXxZnllQLlWeS/view?usp=sharing)

### Backward Locomotion  
[ **View Backward Locomotion Video**](https://drive.google.com/file/d/19WZgnwRTBf2TTEwuUKFu-YkcIJnNiMEI/view?usp=drive_link)

### Sideward Locomotion  
[ **View Leftward Locomotion Video**](https://drive.google.com/file/d/1F9FblJeUDIs2kOXx-gHd2dDTrICOh669/view?usp=drive_link)
[ **View Rightward Locomotion Video**](https://drive.google.com/file/d/1b70eLqoDxakTujM0RViBpFu_Bmsu8wXZ/view?usp=drive_link)

### Rotational Locomotion  
[ **View Clockwise Rotational Video**](https://drive.google.com/file/d/1JMoIIOFdj5rl8yU8qG_p6rYVvHUKUV1o/view?usp=drive_link)
[ **View ReverseClockwise Rotational Video**](https://drive.google.com/file/d/1Fuy9OrDv32MQmup0KJpmC2LzPPZyVnHg/view?usp=drive_link)

Think of reverse clockwise as following the clockwise trajectory with the back of the subject facing the center rotation.


**Video Legend:**
- **Green Arrow**: Commanded linear velocity direction (forward/backward)
- **Blue Circle with Arrow**: Commanded angular velocity (yaw rotation)
- **Robot Movement**: Demonstrates coordinated diagonal gait pattern

*Note: The movement appears subtle due to the robot's small size relative to the Spot-scale simulation environment - this is currenty being trained at a higher action scale to accommodate the size of the minipupper's servo capabilities.* Updated August 5th, 2025. It walks as expected in simulation.

### RL Trained MLP Deployed in Live Robot walking forward
[ **View RL Trained MLP Deployed in the Live Robot**](https://drive.google.com/file/d/1nv4__0dOafNf9OwID0kadZvP415--ZMJ/view?usp=sharing)

### RL Trained MLP Deployed adversarial tests during forward walking command
[ **View RL Trained MLP Deployed in the Live Robot**](https://drive.google.com/file/d/1V6eA2XB2u1r9KYyrS89x8FuwjQu7w_ac/view?usp=sharing)

*Note: This robot follows omnidirecitonal commands and is following a command to walk forward while being adversrially tested by the experimente

##  Critical Scaling Discoveries

### 1. Action Space Scaling

While most parameters required **downscaling** for the Mini Pupper, the action space required **upscaling** due to joint movement constraints.

We will train policies that use the native servo specifications to reduce the sim2real gap, minimizing the action simple, linear scaling.

```python
    "leg_joints": DCMotorCfg(
        joint_names_expr=[
            "base_lf1", "lf1_lf2", "lf2_lf3",  
            "base_rf1", "rf1_rf2", "rf2_rf3",
            "base_lb1", "lb1_lb2", "lb2_lb3",
            "base_rb1", "rb1_rb2", "rb2_rb3"
            ],
        saturation_effort=0.35, #Official 3.5 kg/cm
        velocity_limit=10.47, #Official 0.1s/60° = 10.47 rad/s
        stiffness=80.0, #Estimated for Carbon Fiber Joints       
        damping=2.5, #Scaled to stiffness        
        friction=0.03, #No ball bearing gears        
        armature=0.005,#Arrived at this value empirically testing standing and single limb movement      
```
#### Joint Action Mathematics

The `JointPositionActionCfg` scale parameter controls the range of joint movement:

```python
# Action range calculation:
joint_movement_range = scale × 2  # symmetric around default position
actual_joint_range = scale × 2 × (joint_limit_range)
```

**Scale Parameter Analysis:**

| Robot | Scale Parameter | Joint Range | Reasoning |
|-------|----------------|-------------|-----------|
| **Spot** | 0.2 | ±0.2 rad (±11.5°) | Large joints, small relative movements |
| **Mini Pupper** | 0.5 | ±0.5 rad (±28.75°) | Small joints need larger movemments realtive to their size |

```python
# Boston Dynamics Spot Configuration
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot", 
    joint_names=[".*"], 
    scale=0.2,  # Conservative for large, powerful joints
    use_default_offset=True
)

# Mini Pupper Configuration  
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "base_lf1", "lf1_lf2", "lf2_lf3", #Left Front Leg
        "base_rf1", "rf1_rf2", "rf2_rf3", #Right Front Leg
        "base_lb1", "lb1_lb2", "lb2_lb3", #Left Back Leg
        "base_rb1", "rb1_rb2", "rb2_rb3"  #Left Back Leg
    ],
    scale=0.5,
    use_default_offset=True
)
```

**Radians to Degrees Conversion:**
```
Mini Pupper Action Range = ±0.4 rad = ±22.9°
Spot Action Range = ±0.2 rad = ±11.5°
```

This counter-intuitive scaling occurs because:
1. **Small servos** have limited absolute torque
2. **Locomotion requires proportionally larger joint excursions** for effective ground clearance
3. **Gait coordination** needs sufficient joint range to achieve proper foot placement

### 2. My Systematic Parameter Downscaling

All other parameters required proportional downscaling based on robot dimensions:

#### Velocity Commands
```python
# Spot Configuration
ranges=mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-2.0, 3.0), m/s
    lin_vel_y=(-1.5, 1.5), m/s
    ang_vel_z=(-2.0, 2.0)  m/s
)

# Mini Pupper Configuration (scaled by ~0.4x for body size)
ranges=mdp.UniformVelocityCommandCfg.Ranges(
    lin_vel_x=(-0.35, 0.40), m/s 
    lin_vel_y=(-0.35, 0.35), m/s
    ang_vel_z=(-0.30, 0.30), m/s
)
```

#### Foot Clearance Target
```python
# Spot: 100mm clearance for 840mm tall robot (11.9% of height)
foot_clearance = RewardTermCfg(
    params={"target_height": 0.1}  # 100mm
)

# Mini Pupper: 15.8mm clearance for 133mm tall robot (11.9% of height)  
foot_clearance = RewardTermCfg(
    params={"target_height": 0.02}  # 20.0mm
)
```

#### Mass Randomization
```python
# Spot: ±2.5kg variation on 75kg robot (±3.3%)
add_base_mass = EventTerm(
    params={"mass_distribution_params": (-2.5, 2.5)}
)

# Mini Pupper: ±0.05kg variation on 0.56kg robot (±8.9%)
add_base_mass = EventTerm(
    params={"mass_distribution_params": (-0.05, 0.05)}
)
```

## Training Architecture

### Massive Parallel Simulation
My project leverages the computational approaches outlined in recent quadruped locomotion research from ETH-Zurich to achieve efficient training through massive parallelization:

- **8,098 parallel Mini Pupper environments** training simultaneously
- **Actor-Critic Policy Network**: Three-layer MLP with 12-dimensional action output (one per joint)
- **NVIDIA RTX 4090 GPU acceleration** for physics simulation
- **PPO (Proximal Policy Optimization)** for stable policy learning

#### Neural Network Architecture
The actor-critic framework employs a teacher-student paradigm where:
- **Actor Network**: Outputs 12-dimensional action vector corresponding to the 3 joint segments per leg (hip, knee, ankle) × 4 legs
- **Critic Network**: Evaluates state values for policy optimization
- **Observation Space**: 76-dimensional proprioceptive input including joint states, base motion, and command vectors

### Research Foundation

My work builds upon three key research contributions:

1. **Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning** (Rudin et al., 2022) [¹](#references)
   - Established the foundation for massive parallel simulation in quadruped locomotion
   - Demonstrated that thousands of parallel environments enable rapid policy learning
   - Introduced domain randomization techniques for sim-to-real transfer

2. **Dream to Control: Learning Behaviors by Latent Imagination** (Schwarke et al., 2023) [²](#references)  
   - Advanced model-based RL approaches for locomotion control
   - Contributed to understanding of reward shaping in continuous control tasks
   - Provided insights into gait pattern emergence through learned representations

3. **Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control** (IEEE, 2024) [³](#references)
   - Extended RL locomotion principles to challenging dynamic scenarios
   - Demonstrated robustness techniques applicable to quadruped systems
   - Influenced reward engineering strategies for stable gait development

The combination of these approaches enabled me to train a Mini Pupper locomotion policy from scratch in approximately **4-6 hours** using parallel simulation, compared to weeks or months that would be required with traditional single-environment training.

## Robot Specifications

| Component | Specification |
|-----------|---------------|
| **Dimensions** | 133mm × 220mm × 133mm (H×L×W) |
| **Weight** | 560g total system weight |
| **Joints** | 12 controllable (4 hips, 4 knees, 4 ankles) |
| **Actuators** | Digital servos with position feedback |
| **Sensors** | IMU, optional LiDAR (excluded from action space) |
| **Computational Target** | Raspberry Pi 4 deployment capability |

## Key Technical Achievements

### 1. Joint Control Discovery & Actor Network Design
**Problem**: My initial configuration included all 26+ joints (sensors, decorative plates, etc.)
**Solution**: I isolated the 12 locomotion-critical joints for precise control, directly matching the actor network's 12-dimensional output

The actor-critic architecture required careful consideration of the action space:

```python
# Targeted joint control (Mini Pupper specific)
# Maps directly to 12-dimensional actor network output
joint_names=[
        "base_lf1", "lf1_lf2", "lf2_lf3", #Left Front Leg
        "base_rf1", "rf1_rf2", "rf2_rf3", #Right Front Leg
        "base_lb1", "lb1_lb2", "lb2_lb3", #Left Back Leg
        "base_rb1", "rb1_rb2", "rb2_rb3"  #Left Back Leg
]
# Total: 12 joints = 3 segments per leg × 4 legs
```

This design ensures the actor network's 12-dimensional output vector directly corresponds to the physical joint actuators, enabling efficient policy learning without action space mismatch. The leg output order must match the output of the Ros2 /joint_veliocity order.

### 2. My Reward Engineering Evolution

My reward structure underwent multiple iterations to address specific locomotion problems:

#### Final Reward Configuration
```python
# Primary locomotion objectives
base_linear_velocity = RewardTermCfg(weight=20.0)  # Forward motion priority
gait = RewardTermCfg(weight=20.0)                  # Diagonal coordination
foot_clearance = RewardTermCfg(weight=0.5)         # Appropriate lifting
air_time = RewardTermCfg(weight=5.0)               # Gait timing

# Stability and efficiency penalties  
foot_slip = RewardTermCfg(weight=-2.0)             # Prevent dragging and encourage gait behavior
base_orientation = RewardTermCfg(weight=-3.0)      # Maintain upright posture
joint_torques = RewardTermCfg(weight=-5.0e-4)      # Energy efficiency
```

The actor network learns to output coordinated 12-dimensional action sequences that produce this optimal gait frequency, with the teacher-student paradigm ensuring the policy generalizes across different velocity commands.

## Training Results

### Final Performance Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Reward** | 900+ | Stable, high-performance policy |
| **Episode Length** | ~1000 | Robust locomotion at the maximum possible episode length |
| **Gait Reward** | ~9.0 | Excellent diagonal coordination |
| **Base Linear Velocity** | ~20 | Effective forward motion |
| **Base Angular Velocity** | ~20 | Effective angular motion |
| **Body Contact Rate** | ~0% | Very stable, rare falls |
| **Action Noise** | 0.01 | Converged execution |

### Locomotion Quality Assessment
- **Diagonal trot gait** with proper phase relationships
- **Bidirectional movement perdirection command** (omnidirecitonal capability)
- **Speed**: 0.3 m/s (Max Speed)
- **Stable posture** with minimal body contact terminations

## Technical Implementation

### Environment Configuration
```python
# Simulation Parameters
decimation = 10              # 50 Hz control frequency in simulation, 200Hz on the live robot
episode_length_s = 20.0      # 20-second episodes
sim.dt = 0.002              # 500 Hz physics simulation
num_envs = 4098             # Massive parallelization. These benefits plataeu around 4K envs.

# Terrain Configuration  
terrain_mix = {
    "flat": 50%,            # Distrubte locomotion learning across varied terrain
    "random_rough": 50%     
}
```

### Acttor and Critic Observation Space (48 dimensions)
- Base linear (3D) (Command input)
- Angular velocity (3D) (Command input)
- Projected gravity vector (3D) (0,,0, -9.81m/s^2)
- Velocity commands (3D)
- Joint positions (12D) #Legs and the primary output of the MLP
- Joint Velocities (12D) #Legs and the primary output of the MLP
- Actions (12 D) #Legs and the primary output of the MLP

## Getting Started

### Prerequisites
- Isaac Lab installation
- NVIDIA GPU with CUDA support (RTX 4090 recommended)
- Python 3.8+

### Training Command
```bash
python isaaclab/scripts/reinforcemment_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Custom-Quad-v0 \
    --num_envs=4098 \
    --headless
```

### Configuration Files
- `custom_quad.py`: Mini Pupper robot definition and actuator configuration
- `spot_flat_env_cfg.py`: Training environment with scaled parameters
- `mdp.py`: Custom reward functions and locomotion primitives

## References

1. Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to walk in minutes using massively parallel deep reinforcement learning. *Proceedings of Machine Learning Research*, 164, 91-104. [Link](https://proceedings.mlr.press/v164/rudin22a/rudin22a.pdf)

2. Schwarke, J., Jankowski, J., & Martius, G. (2023). Dream to control: Learning behaviors by latent imagination. *Proceedings of Machine Learning Research*, 229, 1-15. [Link](https://proceedings.mlr.press/v229/schwarke23a/schwarke23a.pdf)

3. Kumar, A., et al. (2024). Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control. *IEEE Transactions on Robotics*, 40, 1234-1250. [Link](https://ieeexplore.ieee.org/document/10611493)

---

# Sim2Real Transfer for Quadruped Locomotion: MLP Controller

This repository documents the complete sim2real transfer process for deploying a trained MLP (Multi-Layer Perceptron) locomotion policy from Isaac Gym simulation to a real Mini Pupper quadruped robot.

## Overview

The project successfully transfers a walking policy trained in Isaac Gym simulation to real hardware, overcoming significant challenges in coordinate frame mismatches, joint direction differences, and action scaling issues that are common in sim2real robotics deployments.

## Key Achievements

- **Successful sim2real transfer** of MLP locomotion policy
- **Coordinated quadruped gait** with proper leg phasing
- **Robust joint-specific corrections** for hardware differences
- **Real-time control** at 50Hz with ROS2 integration
- **Observation normalization** preservation from training

## Architecture

### Model Details
- **Architecture**: 4-layer MLP (48 → 512 → 256 → 128 → 12)
- **Activation**: ELU (Exponential Linear Units)
- **Input**: 48-dimensional observation vector
- **Output**: 12 joint position actions (3 joints × 4 legs)
- **Control Frequency**: 50Hz

### Observation Vector (48D)
```python
obs = [
    base_lin_vel,           # 3D - robot linear velocity
    base_ang_vel,           # 3D - robot angular velocity  
    projected_gravity,      # 3D - gravity direction
    velocity_commands,      # 3D - cmd_vel input
    joint_positions,        # 12D - current joint angles (relative to default)
    joint_velocities,       # 12D - joint angular velocities
    last_action            # 12D - previous action for temporal consistency
]
```

## Major Sim2Real Challenges & Solutions

### 1. Joint Direction Mismatches

**Problem**: Several joints had opposite positive directions between simulation and reality, causing incorrect leg movements and instability.

**Root Cause**: Isaac Gym simulation used different joint axis conventions than the real robot hardware.

**Solutions**:

#### RB (Right-Back) Thigh Joint - Critical Issue
- **Simulation Behavior**: Positive action moved thigh inward
- **Reality Behavior**: Positive action moved thigh outward  
- **Impact**: RB leg consistently lagged behind or moved incorrectly
- **Solution**: Applied constant offset correction
```python
scaled_action[10] += 0.45  # RB thigh offset correction
```

#### RB Hip Joint - Secondary Issue
- **Problem**: Consistently over-active compared to LB hip, causing asymmetric gait
- **Solution**: Applied negative offset to reduce excessive motion
```python
scaled_action[9] -= 0.10   # RB hip correction
```

**Key Insight**: **Offset corrections worked better than multiplicative scaling** because they provide consistent directional bias regardless of action magnitude. Multiplication can cause instability with small actions (becoming too small) or large actions (becoming excessive).

### 2. Hip Action Scaling Evolution

**Problem**: Hip joints were too aggressive, causing instability and unnatural leg spreading.

**Iterative tuning process**:
- **Initial**: `hip_scale = 0.15` (too restrictive, robot couldn't lift legs properly)
- **Intermediate**: `hip_scale = 0.20` (better but still limited range of motion)
- **Final**: `hip_scale = 0.25` (optimal balance of stability and mobility)

```python
hip_indices = [0, 3, 6, 9]  # All hip joints
hip_scale = 0.25
for idx in hip_indices:
    scaled_action[idx] *= hip_scale
```

**Learning**: Hip joints required **uniform scaling** rather than individual corrections, as they control the fundamental stance width and stability across all legs.

### 3. Action Smoothing & Filtering

**Problem**: Raw MLP outputs caused jerky, unstable motions on real hardware due to high-frequency noise and control discontinuities.

**Solution**: Implemented exponential moving average smoothing:
```python
alpha = 0.3  # 30% new action, 70% previous
self.filtered_action = alpha * scaled_action + (1 - alpha) * self.filtered_action
```

**Tuning Process**:
- `alpha = 0.1`: Too sluggish, poor responsiveness to commands
- `alpha = 0.5`: Too jerky, unstable at high speeds
- `alpha = 0.3`: **Optimal balance** of smoothness and responsiveness

### 4. Action Scaling Calibration

**Problem**: Actions trained in simulation were too large for real hardware, causing dangerous joint movements and potential hardware damage.

**Multi-stage scaling evolution**:
1. **Initial**: `action_scale = 0.15` (too aggressive)  
2. **Intermediate**: `action_scale = 0.13` (better but still too high)
3. **Final**: `action_scale = 0.10` (safe and effective)

**Multi-stage scaling process**:
1. **Standard deviation scaling**: `action * action_std` (from training)
2. **Global scaling**: `action * action_scale` 
3. **Joint-specific scaling**: Hip joints reduced by 75%
4. **Individual corrections**: Offsets for problematic joints

### 5. Observation Normalization - Critical for Policy Transfer

**Problem**: Training normalization must be preserved exactly, but real-world sensor data has different characteristics than simulation.

**Solution**: Selective normalization with key signal preservation:
```python
# Apply training normalization
obs_normalized = (obs - obs_mean) / sqrt(obs_var + eps)

# CRITICAL: Preserve key signals in their expected ranges
obs_normalized[9:12] = velocity_commands    # Raw cmd_vel (MLP expects -1 to 1 range)
obs_normalized[6:9] = projected_gravity     # Unit gravity vector (MLP expects normalized)
```

**Key Insight**: Velocity commands and gravity vectors must remain in their original training ranges, as the MLP learned specific response patterns to these value ranges.

### 6. Joint Velocity Estimation

**Problem**: Real robot doesn't provide clean joint velocity feedback, requiring numerical differentiation of noisy position data.

**Solution**: Implemented heavily filtered numerical differentiation:
```python
# Calculate velocity from position differences
instantaneous_vel = (current_pos - prev_pos) / dt

# Apply aggressive filtering (90% old, 10% new)
alpha = 0.1  
joint_velocities = alpha * instantaneous_vel + (1 - alpha) * joint_velocities

# Hard clamp to prevent unrealistic values
joint_velocities = np.clip(joint_velocities, -5.0, 5.0)
```

### 7. Startup Sequence & Safety System

**Problem**: Sudden activation from standing to full walking caused instability and potential hardware damage.

**Solution**: Graduated constraint system with time-based restrictions:

```python
if walking_duration < 1.0:
    # STARTUP PHASE: Very restrictive limits
    joint_limits_low = [-0.15, -0.2, -2.36]  # Especially tight hip limits
    joint_limits_high = [0.15, 1.2, -0.5]
else:
    # NORMAL OPERATION: Standard limits
    joint_limits_low = [-0.3, -0.2, -2.36]
    joint_limits_high = [0.3, 1.2, -0.5]
```

**Standing Mode**: Gradual decay to default positions instead of abrupt stops:
```python
if is_standing:
    self.filtered_action *= 0.95  # 5% decay per timestep
    if np.linalg.norm(self.filtered_action) < 0.01:
        self.filtered_action = np.zeros(12)  # Final stop
```

### 8. Velocity Command Range Matching

**Problem**: Training used specific velocity command ranges that must be preserved for proper policy behavior.

**Solution**: Exact range matching with deadzone filtering:
```python
# Match exact training ranges
self.velocity_commands[0] = np.clip(msg.linear.x, -0.8, 3.5)    # Forward/back
self.velocity_commands[1] = np.clip(msg.linear.y, -0.3, 0.3)    # Left/right  
self.velocity_commands[2] = np.clip(msg.angular.z, -1.0, 1.0)   # Rotation

# Apply deadzone to prevent micro-movements
if abs(msg.linear.x) < 0.05:
    self.velocity_commands[0] = 0.0
```

## Implementation Details

### Control Loop Architecture
```python
def control_loop(self):    # 50Hz execution
    obs = get_observation()           # 48D observation vector
    obs_normalized = apply_training_normalization(obs)
    raw_action = model(obs_normalized)    # MLP inference  
    action_with_std = raw_action * action_std
    scaled_action = action_with_std * action_scale
    apply_hip_scaling(scaled_action)      # Uniform hip reduction
    apply_joint_corrections(scaled_action) # Individual joint fixes
    filtered_action = exponential_smoothing(scaled_action)
    target_pos = default_pos + filtered_action
    target_pos = apply_safety_limits(target_pos)
    publish_joint_commands(target_pos)
```

### Joint Mapping & Coordinate System
```python
joint_mapping = {
    # FRONT LEGS (Isaac indices 0-5)
    0: 'base_lf1',    # LF hip    
    1: 'lf1_lf2',     # LF thigh  
    2: 'lf2_lf3',     # LF calf   
    3: 'base_rf1',    # RF hip    
    4: 'rf1_rf2',     # RF thigh  
    5: 'rf2_rf3',     # RF calf   
    
    # BACK LEGS (Isaac indices 6-11) 
    6: 'base_lb1',    # LB hip    
    7: 'lb1_lb2',     # LB thigh  
    8: 'lb2_lb3',     # LB calf   
    9: 'base_rb1',    # RB hip    ← Required correction
    10: 'rb1_rb2',    # RB thigh  ← Required major correction  
    11: 'rb2_rb3'     # RB calf   
}
```

### Default Standing Positions
Conservative stance optimized for stability:
```python
default_positions = [
    0.0, 0.52, -1.05,  # LF: hip=0°, thigh=30°, calf=-60°  
    0.0, 0.52, -1.05,  # RF: symmetric to LF
    0.0, 0.52, -1.05,  # LB: same stable pose
    0.0, 0.52, -1.05   # RB: before corrections applied
]
```

## Key Insights for Sim2Real Transfer

### 1. **Offset vs Multiplication Strategy**
- **Offsets**: Best for direction mismatches (joint axis flipped)
  - Provide consistent bias regardless of action magnitude
  - Stable across full range of movements
  - Example: `scaled_action[10] += 0.45`

- **Multiplication**: Best for magnitude scaling (joint too sensitive/insensitive)
  - Proportional adjustment maintains relative dynamics
  - Can amplify noise if not carefully applied
  - Example: `scaled_action[hip_indices] *= 0.25`

### 2. **Joint-Specific vs Systematic Corrections**
- **Systematic issues**: Hip joints all needed uniform 75% reduction
- **Individual issues**: Only RB thigh and RB hip needed specific corrections
- **Learning**: Hardware differences manifest both as joint-type patterns and individual joint quirks

### 3. **Temporal Dynamics Matter**
- **Smoothing is critical**: Raw MLP outputs too high-frequency for hardware
- **Startup sequences prevent damage**: Gradual constraint relaxation
- **Standing transitions**: Smooth decay prevents jarring stops

### 4. **Observation Fidelity**
- **Training normalization must be preserved exactly**
- **But key signals need special handling**: cmd_vel and gravity vectors
- **Joint velocities need heavy filtering**: Real sensors much noisier than simulation

### 5. **Safety-First Development**
- **Conservative limits initially**: Gradually expand as confidence grows
- **Extensive logging**: Essential for identifying specific problem joints
- **Comparison metrics**: Left vs right leg analysis revealed asymmetries

### 6. **Iterative Refinement Process**
The successful deployment required systematic iteration:

1. **Coarse scaling** → Get basic movement without damage
2. **Hip tuning** → Establish stable stance and basic gait
3. **Individual joint fixes** → Address specific asymmetries  
4. **Fine-tuning smoothing** → Optimize responsiveness vs stability
5. **Safety system refinement** → Smooth transitions and reliable stops

## Common Failure Modes & Debugging

### Diagnostic Logging Implementation
```python
if self.step_count % 50 == 0:
    # Monitor action ranges
    self.get_logger().info(f'Scaled action: [{scaled_action.min():.3f}, {scaled_action.max():.3f}]')
    
    # Compare symmetric joints
    lb_actions = position_action[6:9]   # LB: hip, thigh, calf  
    rb_actions = position_action[9:12]  # RB: hip, thigh, calf
    self.get_logger().info(f'Back legs - LB: {lb_actions.round(3)}, RB: {rb_actions.round(3)}')
    
    # Flag unusual joint behavior
    if abs(rb_actions[0]) > abs(lb_actions[0]) * 1.5:
        self.get_logger().warn(f'RB hip unusually large: {rb_actions[0]:.3f} vs LB: {lb_actions[0]:.3f}')
```

### Warning Signs to Watch For
- **Action range explosions**: `[raw_action.min(), raw_action.max()]` suddenly very large
- **Joint asymmetries**: Left vs right legs showing very different action patterns
- **Velocity noise**: Joint velocities exceeding reasonable bounds (±5 rad/s)
- **Standing instability**: Robot unable to maintain stable stance
- **Gait irregularities**: Legs not coordinating properly, limping, or leg dragging

## Usage Instructions

### Setup Commands
```bash
# Build the ROS2 package
cd ~/ros2_ws
colcon build --packages-select mlp_controller_pkg
source install/setup.bash

# Start the controller
ros2 run mlp_controller_pkg mlpcontroller

# Send walking commands
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Emergency stop
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

### Testing Sequence
1. **Start stationary** - Verify stable standing
2. **Small forward command** - `x: 0.1` to test basic forward motion
3. **Increase gradually** - Build up to `x: 0.3` for full walking
4. **Test directions** - Try backward (`x: -0.2`), side-to-side (`y: ±0.1`), rotation (`z: ±0.3`)

## Future Improvements

- **Adaptive joint corrections**: Learn individual joint offsets during operation
- **Enhanced safety system**: IMU-based stability monitoring  
- **Gait analysis**: Automated detection of irregular leg coordination
- **Terrain adaptation**: Modify action scaling based on surface detection
- **Model fine-tuning**: Collect real-world data for sim2real domain adaptation

## Conclusion

This sim2real transfer demonstrates that careful, systematic correction of joint-level differences can successfully bridge the simulation-reality gap for complex locomotion policies. The key insights around offset vs multiplication strategies, temporal smoothing, and observation fidelity provide a roadmap for similar deployments on other quadruped platforms.

The final system achieves stable, coordinated quadruped locomotion with the ability to walk forward, backward, strafe, and turn in place - a successful translation of simulation-trained behaviors to real-world hardware.

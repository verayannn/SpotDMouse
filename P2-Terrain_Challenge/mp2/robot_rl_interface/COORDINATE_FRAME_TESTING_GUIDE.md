# Coordinate Frame Testing Guide for v4

## Problem Summary

**Test results showed:**
- ✅ Simulation actions replay perfectly on hardware (smooth, coordinated motion)
- ❌ But directions are wrong:
  - Sim X+ (forward) → Real rotates clockwise
  - Sim Y+ (left strafe) → Real strafes right
  - Sim Z+ (turn left) → Real mostly still
  - Sim Z- (turn right) → Real turns left

**Root cause:** Coordinate frame mismatch between simulation and hardware

## v4 Changes

**New in v4:**
1. **Velocity command transformation** - `_sim_velocity_to_hw_velocity()`
2. **IMU frame transformation** - `_hw_imu_to_sim_imu()`
3. **Multiple remapping modes** - Can test different hypotheses
4. **ACTION_SCALE back to 0.5** - Since we're fixing the coordinate issue

## Remapping Modes to Test

### Velocity Command Remapping (`cmd_remap_mode`)

**Mode 1** (default): Y-flip + 90° rotation
```python
vx_hw = -vy_sim   # Forward → Right (then rotated becomes forward)
vy_hw = -vx_sim   # Left → Backward
vyaw_hw = -vyaw_sim  # Flip yaw
```

**Mode 2**: Y-axis flip only
```python
vx_hw = vx_sim    # Forward stays forward
vy_hw = -vy_sim   # Left → Right
vyaw_hw = vyaw_sim   # Yaw unchanged
```

**Mode 3**: 90° rotation only
```python
vx_hw = -vy_sim   # Forward → Right
vy_hw = vx_sim    # Left → Forward
vyaw_hw = vyaw_sim   # Yaw unchanged
```

**Mode 0**: No remapping (for comparison)

### IMU Remapping (`imu_remap_mode`)

**Mode 1** (default): Matches velocity remapping
```python
gyro_sim = [-gyro_hw[1], -gyro_hw[0], -gyro_hw[2]]
accel_sim = [-accel_hw[1], -accel_hw[0], accel_hw[2]]
```

**Mode 2**: Y-flip only
**Mode 3**: 90° rotation
**Mode 0**: No remapping

## Testing Protocol

### Step 1: Test Each Remap Mode with Live Policy

For each mode (1, 2, 3, 0):

```bash
python mlp_controller_v4.py

> set cmd_remap 1
> set imu_remap 1
> w              # Try forward at 0.10 m/s
# Observe: Does robot walk forward?

> set cmd_remap 2
> set imu_remap 2
> w
# Observe again

> set cmd_remap 3
> set imu_remap 3
> w

> set cmd_remap 0  # No remapping (baseline)
> set imu_remap 0
> w
```

### Step 2: Record Results

For each mode, note:
- Does robot walk **forward** when you press `w`?
- Does robot strafe **left** when you press `a`?
- Does robot turn **left (CCW)** when you press `q`?
- Does it walk smoothly or thrash around?
- What are the `base_ang_vel` and `gravity` values?

### Step 3: Find the Winning Mode

The correct mode will:
- ✅ Robot walks forward on `w` command
- ✅ Robot strafes left on `a` command
- ✅ Robot turns CCW on `q` command
- ✅ `base_ang_vel` stays under 0.5 rad/s
- ✅ `gravity[2]` stays below -0.95
- ✅ Actions don't hit ±1.0 limits

## Quick Test Commands

```bash
# Test forward walking with mode 1
python mlp_controller_v4.py
> w

# Test forward walking with mode 2
>
> set cmd_remap 2
> set imu_remap 2
> w

# Test forward walking with mode 3
>
> set cmd_remap 3
> set imu_remap 3
> w
```

## Expected Outcome

**If you find the correct remapping mode:**
- Robot will walk in expected directions
- Policy will work because observations now match simulation
- base_ang_vel and gravity will be in healthy ranges

**If NO mode works perfectly:**
- We may need a custom transformation matrix
- We can measure the exact rotation angle needed
- We can add more sophisticated frame transformations

## Debugging Tips

1. **Start with `debug`** to see standing observations
2. **Use low speed** (0.10) for initial tests
3. **Stop immediately** if robot tilts significantly
4. **Compare** ang_vel and gravity across modes
5. **Test simulation replay** with `test` to verify actions still work

## Next Steps After Finding Correct Mode

Once you identify the working `cmd_remap` and `imu_remap` modes:

1. Update the default in v4 code
2. Re-run `compare_observations.py` to verify observation errors are reduced
3. Test with higher speeds (0.15, 0.20, 0.30)
4. Test all directions (w/s/a/d/q/e)
5. Celebrate! 🎉

## Fallback Plan

If modes 1-3 don't work, we can:
- Add mode 4 with custom transformation
- Measure exact hardware frame orientation
- Use quaternion rotations for precision
- Add observation logging to diagnose remaining issues

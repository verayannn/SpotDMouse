# Difference: existing `deploy_network.py` vs proposed `lstm_servo_policy.py`

## Existing code: `deploy_network.py` (deployment runtime)

Purpose: run an already-trained TorchScript policy on the physical robot.

- **Scope**: inference + hardware control loop + operator CLI.
- **Policy call**: single-step `policy(obs)` at runtime.
- **State source**:
  - Real IMU (gyro/accel) via ESP32 + complementary filter.
  - Joint channels are mostly **synthetic** from an internal delayed PD simulator.
- **Delay handling**: fixed-size action delay queue (`pd_delay_steps`) in PD simulator.
- **Outputs**: servo target positions sent to hardware.
- **Training**: none (no optimizer/loss/backprop).

## Proposed code: `lstm_servo_policy.py` (trainable architecture module)

Purpose: define a clean two-layer model that can be trained for delay-robustness.

- **Scope**: neural architecture definition only (no hardware I/O loop).
- **Layer 1**: `TemporalPolicyLSTM`
  - Inputs: IMU angular velocity (3), projected gravity (3), self-actions (12).
  - Output: multi-step command sequence over horizon `H`.
- **Layer 2**: `ServoModelMLP`
  - Inputs: command sequence `(H x 12)` + current real position `(12)`.
  - Output: predicted joint position sequence `(H x 12)`.
- **Delay handling**: via curriculum stage parameters (min/max delay, noise, terrain).
- **Outputs**: tensors for rollout/supervision in training.
- **Training**: designed to plug into PPO + auxiliary servo prediction losses.

## Practical relationship

- `deploy_network.py` is a **runtime controller** for execution on robot.
- `lstm_servo_policy.py` is a **modeling/training building block** for learning.
- They are complementary: train with the new stack, export policy, then deploy with a runtime controller (which may later be updated to consume richer feedback).

# deploy_network.py explained

`deploy_network.py` is an interactive runtime controller for Mini Pupper 2 that deploys a TorchScript locomotion policy (LSTM or MLP) to real hardware while synthesizing part of the observation with an internal delayed PD simulator.

## High-level behavior

1. Loads a policy preset (`lstm`, `lstm_25hz`, `mlp`, `mlp_25hz`) with controller gains/scales.
2. Reads real IMU from ESP32 and estimates angular velocity + gravity via a complementary filter.
3. Maintains an **open-loop delayed PD joint simulation** to generate joint position/velocity observation channels expected by the policy.
4. Builds a 60D observation vector that mixes:
   - zeros for linear velocity and effort,
   - real IMU,
   - command input,
   - synthetic joint states from PD simulator,
   - previous actions.
5. Runs policy inference, clips/smooths actions, and sends scaled targets to hardware servos.
6. Provides an interactive CLI (`w/s/a/d/q/e`, `stance`, `preset`, `set`, etc.) for operation and tuning.

## Why the PD simulation exists

The policy was trained with delayed PD actuator dynamics; deployment reconstructs those actuator-state observation channels using a local PD model (`_step_pd`) rather than direct servo-state feedback, so policy input shape/semantics remain close to training.

## Key runtime details

- **Control frequency**: preset-dependent (25 or 50 Hz).
- **Delay model**: `pd_delay_steps` queue delays policy action before PD dynamics.
- **Smoothing**: Butterworth low-pass filter over actions.
- **Safety-ish bounds**: action clipping, joint limit clipping, startup fade-in, stance interpolation.
- **Logging**: optional CSV capture of obs/raw/final actions for ~600 steps.

## Main caveat

The stack is still largely open-loop for joints (real IMU + synthetic joints), so mismatch between simulated PD state and physical servo state can accumulate.

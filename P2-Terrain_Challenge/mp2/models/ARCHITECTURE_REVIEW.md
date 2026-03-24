# MP2 Architecture Review + Upgrade Path (LSTM Policy + Curriculum + Servo Model)

## What is in the current code

From `mp2` deployment/analysis scripts, there is already partial support for recurrent policies and delay modeling:

- LSTM and MLP TorchScript policies are both deployed.
- Delay is currently modeled mostly with fixed delay buffers (`pd_delay_steps`) in open-loop PD wrappers.
- Hardware prediction and observer work is split across analysis scripts, but not encapsulated as a reusable, train-time two-layer module.

This means the code already has the ingredients for your target architecture, but not yet as a single, clean policy stack that can be trained end-to-end with explicit curriculum stages.

## Implemented in this patch

A reusable module was added at `mp2/models/lstm_servo_policy.py` with exactly the requested split:

1. **Layer 1 policy (LSTM):**
   - Input: IMU angular velocity (3), projected gravity (3), self-actions (12)
   - Output: command sequence over a 5-step horizon
2. **Layer 2 servo model (MLP):**
   - Input: command sequence + real position at time *t*
   - Output: predicted joint position sequence
3. **Curriculum helper:**
   - Staged delay/noise/terrain progression suitable for variable-delay robustness training.

## How to integrate

1. Replace direct `policy(obs)` calls in your training loop with:
   - `command_seq, pred_seq, hidden = stack(obs_lstm, real_pos_t, hidden)`
2. Roll out actuator delay/servo effects using `pred_seq` (or use it as auxiliary supervision).
3. At each PPO update, compute stage from progress and set environment randomization from `DelayCurriculumStage`.

## Suggested losses

- `L_policy`: PPO objective for the LSTM policy output.
- `L_servo`: MSE(predicted_pos_seq, measured_or_simulated_pos_seq).
- `L_smooth`: temporal smoothness on command sequence.
- `L_total = L_policy + λ1*L_servo + λ2*L_smooth`.

## Notes for variable delays

- During training, sample delay each episode (or step) in `[min_delay_steps, max_delay_steps]` from current stage.
- Keep action history in observations (already included as self-actions) so LSTM can infer latent delay state.
- Increase delay range only when policy meets stability threshold at current stage.

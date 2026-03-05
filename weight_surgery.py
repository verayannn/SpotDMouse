"""
Weight surgery: expand a 33-dim checkpoint to 69-dim for action history observations.

Old layout (33 dims):
  [0:3]   base_ang_vel
  [3:6]   projected_gravity
  [6:9]   velocity_commands
  [9:21]  joint_pos_rel (12)
  [21:33] last_action (12)

New layout (69 dims):
  [0:3]   base_ang_vel
  [3:6]   projected_gravity
  [6:9]   velocity_commands
  [9:21]  joint_pos_rel (12)
  [21:69] action_history (4 x 12 = 48)
          [21:33] = a_{t-4}
          [33:45] = a_{t-3}
          [45:57] = a_{t-2}
          [57:69] = a_{t-1}  <-- old last_action weights go here

Usage:
  python weight_surgery.py /path/to/model_12950.pt
  python weight_surgery.py /path/to/model_12950.pt /path/to/output.pt
"""

import torch
import sys
import os

OLD_DIM = 33
NEW_DIM = 69
SHARED_PREFIX = 21       # ang_vel + gravity + cmd + joint_pos
OLD_ACTION_SLICE = (21, 33)   # old last_action location
NEW_T1_SLICE = (57, 69)       # new a_{t-1} location (last 12 of action_history)


def surgery(checkpoint_path, output_path=None):
    if output_path is None:
        base, ext = os.path.splitext(checkpoint_path)
        output_path = f"{base}_69dim{ext}"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # RSL-RL stores model weights under 'model_state_dict'
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    print("All keys and shapes:")
    for k, v in state_dict.items():
        print(f"  {k}: {v.shape}")

    # Expand input layers (any weight matrix with 33 columns)
    modified = []
    for key, tensor in state_dict.items():
        if tensor.dim() == 2 and tensor.shape[1] == OLD_DIM:
            out_features = tensor.shape[0]
            new_weight = torch.zeros(out_features, NEW_DIM)

            # Copy shared prefix weights (dims 0-20: ang_vel, gravity, cmd, joint_pos)
            new_weight[:, :SHARED_PREFIX] = tensor[:, :SHARED_PREFIX]

            # Copy old last_action weights to the a_{t-1} slot (last 12 of history)
            new_weight[:, NEW_T1_SLICE[0]:NEW_T1_SLICE[1]] = tensor[:, OLD_ACTION_SLICE[0]:OLD_ACTION_SLICE[1]]

            # Dims 21-57 (a_{t-4}, a_{t-3}, a_{t-2}) stay zero-initialized
            state_dict[key] = new_weight
            modified.append(f"  {key}: ({out_features}, {OLD_DIM}) -> ({out_features}, {NEW_DIM})")

    if not modified:
        print(f"\nERROR: No layers found with input dim {OLD_DIM}. Check checkpoint format.")
        sys.exit(1)

    print(f"\nExpanded {len(modified)} layers:")
    for m in modified:
        print(m)

    # Clear optimizer state — shapes changed, must reinitialize
    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = state_dict
    else:
        checkpoint = state_dict

    if "optimizer_state_dict" in checkpoint:
        # Keep the key (RSL-RL expects it) but clear per-parameter state
        # (momentum buffers have wrong shapes). Optimizer will rebuild them on first step.
        old_optim = checkpoint["optimizer_state_dict"]
        checkpoint["optimizer_state_dict"] = {
            "state": {},
            "param_groups": old_optim["param_groups"],
        }
        print("\nReset optimizer state (kept param_groups, cleared momentum buffers)")

    torch.save(checkpoint, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python weight_surgery.py <checkpoint.pt> [output.pt]")
        sys.exit(1)

    surgery(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

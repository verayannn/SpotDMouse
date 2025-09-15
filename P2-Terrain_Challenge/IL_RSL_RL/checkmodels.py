import torch
import numpy as np
import IPython

IL_MLP_FILE = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt"
RL_MLP_FILE = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/birthdayrun/2025-08-07_19-17-44/model_9999_with_stats.pt"

IL_MODEL = torch.load(IL_MLP_FILE, weights_only=False)
RL_MODEL = torch.load(RL_MLP_FILE, weights_only=False)

il_keys = IL_MODEL.keys()
rl_keys = RL_MODEL.keys()

print("=== IL Model Keys ===")
print(f"Total keys: {len(il_keys)}")
for key in sorted(il_keys):
    print(f"  - {key}")

print("\n=== RL Model Keys ===")
print(f"Total keys: {len(rl_keys)}")
for key in sorted(rl_keys):
    print(f"  - {key}")

print("\n=== Key Comparison ===")
common_keys = set(il_keys) & set(rl_keys)
il_only_keys = set(il_keys) - set(rl_keys)
rl_only_keys = set(rl_keys) - set(il_keys)

print(f"Common keys: {len(common_keys)}")
print(f"Keys only in IL model: {len(il_only_keys)}")
if il_only_keys:
    for key in sorted(il_only_keys):
        print(f"  - {key}")

print(f"\nKeys only in RL model: {len(rl_only_keys)}")
if rl_only_keys:
    for key in sorted(rl_only_keys):
        print(f"  - {key}")

print("INFOS in RL model", RL_MODEL['infos'])
print("ITER in RL model", RL_MODEL['iter'])
print("NUM_ACTIONS in RL model", RL_MODEL['num_actions'])
print("OBS RMS MEAN shape in RL model", RL_MODEL['obs_rms_mean'].shape)
print("OBS RMS VAR shape in RL model", RL_MODEL['obs_rms_var'].shape)

print("ACTION MEAN in IL model", IL_MODEL['action_mean'])
print("ACTION STD in IL model", IL_MODEL['action_std'])
print("ACTION EPOCH in IL model", IL_MODEL['epoch'])
print("OBS MEAN in IL model", IL_MODEL['obs_mean'])
print("OBS STD in IL model", IL_MODEL['obs_std'])

print(IL_MODEL['model_state_dict'].keys())
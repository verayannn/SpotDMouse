import torch
import numpy as np
import h5py as h5
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys 

# Ensure correct backend for plotting if needed
import matplotlib
# matplotlib.use("Qt5Agg") # Keep this commented out unless running on a display server

device = torch.device("cpu")

# --- MODEL PATHS ---
rsl_model_path = os.path.expanduser("~/rsl_rl_trainedmodels/45degree_mlp.pt") 
il_model_path = os.path.expanduser("~/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt")
il_demonstrations_path = os.path.expanduser("~/mini_pupper_demos_20250914_233847.hdf5")

# --- ACTOR-CRITIC CLASS ---
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # ... (Actor and Critic definitions are omitted for brevity, assumed correct)
        self.actor = nn.Sequential(
                nn.Linear(48, 512, bias=True), nn.ELU(),
                nn.Linear(512, 256, bias=True), nn.ELU(),
                nn.Linear(256, 128, bias=True), nn.ELU(),
                nn.Linear(128, 12, bias=True)
                )
        self.critic = nn.Sequential(
                nn.Linear(48, 512, bias=True), nn.ELU(),
                nn.Linear(512, 256, bias=True), nn.ELU(),
                nn.Linear(256, 128, bias=True), nn.ELU(),
                nn.Linear(128, 1, bias=True)
                )
    def forward(self,x):
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

# --- OBSERVATION REMAPPING FUNCTION (CRITICAL CHANGE) ---

def remap_il_to_rsl_obs(il_obs_np):
    """
    Transforms a single 48-dim observation vector from the IL training order 
    to the standard RSL-RL order.

    IL Order: [Cmd(3), Q_rel(12), dQ(12), Action_prev(12), Grav(3), LinVel(3), AngVel(3)]
    RSL Order: [LinVel(3), AngVel(3), Grav(3), Cmd(3), Q_rel(12), dQ(12), Action_prev(12)]
    """
    if il_obs_np.shape[-1] != 48:
        raise ValueError(f"Input observation must be 48 elements, got {il_obs_np.shape[-1]}")

    # Extract features from the IL vector
    cmd_vel       = il_obs_np[..., 0:3]
    joint_pos_rel = il_obs_np[..., 3:15]
    joint_vel     = il_obs_np[..., 15:27]
    last_action   = il_obs_np[..., 27:39]
    proj_gravity  = il_obs_np[..., 39:42]
    base_lin_vel  = il_obs_np[..., 42:45]
    base_ang_vel  = il_obs_np[..., 45:48]

    # Assemble into the RSL-RL vector
    rsl_obs = np.concatenate([
        base_lin_vel,
        base_ang_vel,
        proj_gravity,
        cmd_vel,
        joint_pos_rel,
        joint_vel,
        last_action
    ], axis=-1)
    
    return rsl_obs

# --- MODEL LOADING (NO CHANGE) ---
chckpt_rsl_model = torch.load(rsl_model_path, weights_only=False, map_location=device)
chckpt_il_model = torch.load(il_model_path, weights_only=False, map_location=device)

rsl_model = ActorCritic()
rsl_state_dict = chckpt_rsl_model['model_state_dict']
rsl_model.load_state_dict(rsl_state_dict, strict=False)
rsl_model.eval()

il_model = ActorCritic()
il_state_dict = chckpt_il_model['model_state_dict']
il_model.load_state_dict(il_state_dict, strict=False)
il_model.eval()

# --- DEMONSTRATION DATA PROCESSING ---
demo = h5.File(il_demonstrations_path, 'r')
obs_demo_il_order = np.array(demo['data/demo_2/obs'], dtype=np.float32)
obs_demo_tensor = torch.Tensor(obs_demo_il_order)

il_outputs = []
rsl_outputs = []

for i, o_il in enumerate(obs_demo_il_order):
    
    # 1. IL Model: Feed raw data (correct order for IL)
    o_il_tensor = torch.tensor(o_il, dtype=torch.float32)
    il_output , _ = il_model(o_il_tensor)
    il_outputs.append(il_output)

    # 2. RSL Model: Remap data (correct order for RSL-RL)
    o_rsl_np = remap_il_to_rsl_obs(o_il)
    o_rsl_tensor = torch.tensor(o_rsl_np, dtype=torch.float32)
    rsl_output, _ = rsl_model(o_rsl_tensor)
    rsl_outputs.append(rsl_output)

il_outputs = torch.stack(il_outputs,dim=0)
rsl_outputs = torch.stack(rsl_outputs,dim=0)

print(il_outputs.shape) # should be ~3K,12
print(rsl_outputs.shape)

print(il_outputs.detach().numpy().T.shape) #should be 12,~3K
print(rsl_outputs.detach().numpy().T.shape)

body_parts_dict = {
    0: "LEFT_FRONT_HIP",
    1: "LEFT_FRONT_THIGH",
    2: "LEFT_FRONT_CALF",
    3: "RIGHT_FRONT_HIP",
    4: "RIGHT_FRONT_THIGH",
    5: "RIGHT_FRONT_CALF",
    6: "LEFT_BACK_HIP",
    7: "LEFT_BACK_THIGH",
    8: "LEFT_BACK_CALF",
    9: "RIGHT_BACK_HIP",
    10: "RIGHT_BACK_THIGH",
    11: "RIGHT_BACK_CALF"
}

il_np = il_outputs.detach().numpy()
rsl_np = rsl_outputs.detach().numpy()

# Total number of joints (12)
J = len(body_parts_dict) 

# Set up the subplot grid (4 rows x 3 columns)
N_ROWS = 4
N_COLS = 3
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(15, 12), sharex=True, sharey=True)
fig.suptitle('Robot Joint Outputs: Comparison of IL and RSL Models', fontsize=16)

for i in range(J):
    # Calculate the position in the subplot grid
    row = i // N_COLS
    col = i % N_COLS

    ax = axes[row, col]

    ax.plot(il_np[:, i], label='IL Output', color='blue', alpha=0.7)
    ax.plot(rsl_np[:, i], label='RSL Output', color='red', alpha=0.7)

    ax.set_title(body_parts_dict[i], fontsize=10)

    if i == 0:
        ax.legend(loc='upper right')

    if row == N_ROWS - 1:
        ax.set_xlabel('Time Steps', fontsize=10)

    if col == 0:
        ax.set_ylabel('Joint Command Value', fontsize=10)

fig.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
plt.savefig('joint_outputs_comparison.png')

scale = 20

rsl_t = rsl_np.T
il_t = il_np.T

reg = rsl_t[0][1200:1500]
scaled = rsl_t[0][1200:1500]/scale

plt.plot(reg, label="reg")
plt.plot(scaled, label="scaled")
plt.plot(il_t[0][1200:1500], label="ref")

plt.legend()
plt.grid(True)
plt.show()

new_scale = int(input(f"Adjust Scale? Current Scale {scale}"))

reg = rsl_t[0][1200:1500]
scaled = rsl_t[0][1200:1500]/new_scale

plt.plot(reg, label="reg")
plt.plot(scaled, label="scaled")
plt.plot(il_np.T[0][1200:1500], label="ref")

plt.legend()
plt.grid(True)
plt.show()
plt.close()

try:
    scale = 20.0
    
    scale_input = input(f"Adjust Scale? Current Scale {scale}")
    if scale_input: # Only update if the user entered something
        new_scale_float = float(scale_input)
        if new_scale_float == 0:
             print("Scale cannot be zero. Using default scale (20.0).")
             new_scale_float = scale
        scale = new_scale_float
        
except ValueError:
    print("Invalid input. Scale must be a number. Using current scale.")
    
TIME_SLICE = slice(1200, 1500)
N_ROWS = 4
N_COLS = 3
J = len(body_parts_dict) # J should be 12

scaled_rsl_np = rsl_np / scale

fig2, axes2 = plt.subplots(N_ROWS, N_COLS, figsize=(15, 12), sharex=True, sharey=False)
fig2.suptitle(f'Scaled RSL Output Comparison (Scale Factor: {scale})', fontsize=16)

for i in range(J):
    row = i // N_COLS
    col = i % N_COLS
    ax = axes2[row, col]

        
    ax.plot(rsl_np[TIME_SLICE, i], 
            label='Original RSL', 
            color='red', 
            alpha=0.5, 
            linestyle='--')
            
    ax.plot(scaled_rsl_np[TIME_SLICE, i], 
            label='Scaled RSL', 
            color='green', 
            alpha=0.9, 
            linewidth=2)

    ax.plot(il_np[TIME_SLICE, i], 
            label='IL Reference', 
            color='blue', 
            alpha=0.7)
    
    ax.set_title(body_parts_dict[i], fontsize=10)
    
    if i == 0:
        ax.legend(loc='lower left')
        
    if row == N_ROWS - 1:
        # Show time steps relative to the slice start (0 to 300)
        ax.set_xlabel('Time Steps (Relative)', fontsize=10) 
        
    if col == 0:
        ax.set_ylabel('Joint Command Value', fontsize=10)
        
    ax.grid(True, linestyle=':', alpha=0.6)

fig2.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
plt.savefig('scaled_joint_outputs_comparison.png')
print("Saved scaled joint comparison plot to 'scaled_joint_outputs_comparison.png'")

plt.close(fig2)



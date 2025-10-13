import torch
import numpy as np
import IPython
import torch.nn as nn
from collections import  OrderedDict
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import h5py as h5
import os
import sys 

device = torch.device("cpu")

rsl_model_path = os.path.expanduser("~/rsl_rl_trainedmodels/45degree_mlp.pt") #"/home/ubuntu/rsl_rl_trainedmodels/30degree_mlp.pt"
il_model_path = os.path.expanduser("~/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt")

il_demonstrations_path = os.path.expanduser("~/mini_pupper_demos_20250914_233847.hdf5")

chckpt_rsl_model = torch.load(rsl_model_path, weights_only=False, map_location=device)
chckpt_il_model = torch.load(il_model_path, weights_only=False, map_location=device)

rsl_state_dict = chckpt_rsl_model['model_state_dict']
il_state_dict = chckpt_il_model['model_state_dict']

for key in rsl_state_dict.keys():
    print("rsl",key)
for key in il_state_dict.keys():
    print('il',key)

if rsl_state_dict.keys() == il_state_dict.keys():
    print(True)
else:
    print(False)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                nn.Linear(48, 512, bias=True),
                nn.ELU(),
                nn.Linear(512,256, bias=True),
                nn.ELU(),
                nn.Linear(256, 128, bias=True),
                nn.ELU(),
                nn.Linear(128,12, bias=True)
                )
        self.critic = nn.Sequential(
                nn.Linear(48,512, bias=True),
                nn.ELU(),
                nn.Linear(512,256, bias=True),
                nn.ELU(),
                nn.Linear(256,128, bias=True),
                nn.ELU(),
                nn.Linear(128,1, bias=True)
                )
    def forward(self,x):
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

rsl_model = ActorCritic()
rsl_model.load_state_dict(rsl_state_dict, strict=False)#strict=False since I do not have an 'std' parameter in the plain instatiation of the model.
rsl_model.eval()

il_model = ActorCritic()
il_model.load_state_dict(il_state_dict, strict=False)
il_model.eval()

GRAVITY = [0.0, 0.0, -9.81]
DEFAULT_JOINT_POS = [ #30 thigh-to-calf angle
        0.0, 0.785 , -1.57,
        0.0, 0.785 , -1.57,
        0.0, 0.785 , -1.57,
        0.0, 0.785 , -1.57,
        ]

def build_observation(joint_state_msg, cmd_vel_msg, prev_actions, prev_joint_actions=None):
    obs = np.zeros()

    obs[0:3] = [0.0,0.0,0.0]# IMU: N/A
    obs[3:6] = [0.0, 0.0, -9.81] #Projected Gravity, assume upright
    obs[6:9] = [cmd_vel_msg.linear.x, cmd_vel_msg.linear.y, cmd_vel_msg.angular.z]

    current_positions = np.array(joint_state_msg.position)
    obs[9:21] = current_positions - DEFAULT_JOINT_POS

    if len(joint_state_msg.velocity) > 0:
        obs[21:33] = joint_state_msg.velocity
    elif prev_joint_pos is not None:
        dt = 0.02 # given that the pi reports at 200hz, we may be undersampling
        obs[21:33] = (current_positions - prev_joint_pos) / dt
    else:
        obs[21:33] = np.zeros(12)

    obs[33:45] = prev_actions

    obs[45:48] = [0.0, 0.0, 0.0]

    return obs

#make fake observaitions and compare the aciton output scales
obs = np.zeros(48)
obs[0:3] = [0.0,0.0,0.0]
obs[3:6] = [0.0, 0.0, -9.81]
obs[6:9] = [0.2, 0.0, 0.0]
obs[9:21] = DEFAULT_JOINT_POS

obs = torch.tensor(obs, dtype=torch.float32)

print(obs.shape)

rsl_output, _ = rsl_model(obs)
il_output, _ = il_model(obs)

print(rsl_output.shape, il_output.shape)

plt.plot(rsl_output.detach().cpu().numpy())
plt.plot(il_output.detach().cpu().numpy())
plt.savefig("compare_output.png")
plt.close()

demo = h5.File(il_demonstrations_path, 'r')

IPython.embed()
sys.exit()

obs_demo = demo['data/demo_2/obs']
obs_demo = torch.Tensor(obs_demo)

il_outputs = []
rsl_outputs = []
for i, o in enumerate(obs_demo):
    rsl_output, _ = rsl_model(o)
    il_output , _ = il_model(o)

    il_outputs.append(il_output)
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

        
    # ax.plot(rsl_np[TIME_SLICE, i], 
    #         label='Original RSL', 
    #         color='red', 
    #         alpha=0.5, 
    #         linestyle='--')
            
    # ax.plot(scaled_rsl_np[TIME_SLICE, i], 
    #         label='Scaled RSL', 
    #         color='green', 
    #         alpha=0.9, 
    #         linewidth=2)

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



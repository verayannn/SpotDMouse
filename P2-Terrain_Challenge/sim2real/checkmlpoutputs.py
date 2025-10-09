import torch
import numpy as np
import IPython
import torch.nn as nn
from collections import  OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py as h5
import os

device = torch.device("cpu")

rsl_model_path = os.path.expanduser("~/rsl_rl_trainedmodels/30degree_mlp.pt") #"/home/ubuntu/rsl_rl_trainedmodels/30degree_mlp.pt"
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

demo = h5.File(il_demonstrations_path, 'r')

obs demo['data/demo_1/obs']

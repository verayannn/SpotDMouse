import torch
import numpy as np
import IPython
import torch.nn as nn
from collections import  OrderedDict

device = torch.device("cpu")

rsl_model_path = "/home/ubuntu/rsl_rl_trainedmodels/30degree_mlp.pt"
il_model_path = "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"

chckpt_rsl_model = torch.load(rsl_model_path, weights_only=False, map_location=device)
chckpt_il_model = torch.load(il_model_path, weights_only=False, map_location=device)

rsl_model = chckpt_rsl_model['model_state_dict']
il_model = chckpt_il_model['model_state_dict']

for key in rsl_model.keys():
    print("rsl",key)
for key in il_model.keys():
    print('il',key)

if rsl_model.keys() == il_model.keys():
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

model = ActorCritic()
model.load_state_dict(rsl_model, strict=False)#strict=False since I do not have an 'std' parameter in the plain instatiation of the model.
model.eval()

test_input = torch.rand(1,48).to(device)

actor, _ = model(test_input)

print(actor.shape) # should be torch.size[1,12]



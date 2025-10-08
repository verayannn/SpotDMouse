import torch
import numpy as np
import IPython

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



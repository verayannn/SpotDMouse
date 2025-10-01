import torch
import numpy as np
import voltron
import tqdm
import IPython
import bopt
import bscope

goldroger_pth = "/home/grandline/cortical/goldroger_epoch_95.pt"
charmander_pth = "/home/grandline/cortical/best_charmander.pt"

device = torch.device("cuda:1")

goldroger_model = torch.load(goldroger_pth, weights_only=False, map_location=device)
charmander_model = torch.load(charmander_pth, weights_only=False, map_location=device)

print(goldroger_model)
print(charmander_model)



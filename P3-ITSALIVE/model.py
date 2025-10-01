import torch
import numpy as np
import voltron
import tqdm

goldroger_pth = "/Users/javierweddington/cortical/goldroger_epoch_95.pt"
charmander_pth = "/Users/javierweddington/cortical/best_charmander.pt"

device = torch.device("cpu")

golroger_model = torch.load(goldroger_pth, weights_only=False, map_location=device)
charmander_model = torch.load(charmander_pth, weights_only=False, device=device)

print(goldroger_model)


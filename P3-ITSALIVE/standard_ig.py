import numpy as np
import torch

def integrated_gradients(inp, baseline, label, steps=range(50)):
    t_input = torch.tensor(inp) 
    t_prediction = torch.tensor(label)

    t_gradients =  torch.autograd.grad(t_prediction, t_input) # check the output size/shape

    path_inputs = [baselne]

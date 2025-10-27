import numpy as np
import torch
import models
import IPython

DEVICE = torch.device('cpu')
goldroger_pth = "/Users/javierweddington/cortical/goldroger_epoch_95.pt"
goldroger_model = torch.load(goldroger_pth, map_location=DEVICE, weights_only=False)

def preds_and_grads(model, inp, target_class=None):

    if model == None:
        model = goldroger_model
    
    inp.requires_grad_(True)

    output = model(inp)

    target = output[:,target_class]

    target.backward(retain_graph=True)

    gradients = inp.grads

    return target, gradients

def integrated_gradients(inp, baseline, predictions_and_gradients, steps=range(50)):
    
    if baseline is None:
        baseline = 0*inp
    assert(baseline.shape == inp.shape)
   
    scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]

    predictions, gradients = predictions_and_gradients(inp=scaled_inputs ,target_class=1)

    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    integrated_gradients = (inp-baseline)*avg_grads  # shape: <inp.shape>

    return integrated_gradients

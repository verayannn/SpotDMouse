import numpy as np
import torch

#Computing the gradients should be the same
def compute_gradients(model, inputs, target_class=None):
   
    inputs = inputs.requires_grad_(True)
    output = model(inputs)

    if target_class is None:
        target_class = outputs.argmax(dim=1)
    
    if isinstance(target_class, int):
        target = output[:,target_class]
    else:
        target = output.gather(1,target_class.view(-1,1)).squeeze()

    model.zero_grad()
    target.sum().backward(retain_graph=True)

    gradients = inputs.grad

    return output, inputs.grad.detach()

def spatiotemporal_integrated_gradients(model, input_tensor, interval=5, baseline=None, target_class=None, steps=50):
    
    #Initialize X'
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    assert baseline.shape == input_tensor.shape, "Should be [batch,history,height,width]"#Should we be integrating through the history or the batch?
     
    #Def Partial Input States
    T = input_tensor.size(1)
    partial = T / interval
    
    X_prime_par_t = [seg for partial in baseline] #Baselines
    alphas = torch.linspace(t-1/T, t/T, partial + X_part_t, device=input_tensor.device)

    accumulated_gradients = torch.zeros_like(input_tensor)
    
    for t in range(1, T):
        betas = torch.dot(alphas, T)-(t-1) #constrained by the alphas
        interpolation = X_par_t + betas * (X_part_t - (X_par_t -1))
        
        X_par_t = [seg for partial in input_tensor]

        _ , gradients = compute_gradients(model, input_tensor, target_class)


        #where does the image get revelaed
        #generate s interpolation steps (b_1, b_s),  where Bj = j/s
        reveal = 
        

    return integrated_grads

def spatiotemporal_integrated_gradients_batch(model, input_tensor, baseline=None, target_class=None,
                              steps=50, batch_size=32):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    alphas = torch.linspace(0, 1, steps + 1, device=input_tensor.device)

    scaled_inputs = []
    for alpha in alphas:
        interpolated = baseline + alpha * (input_tensor - baseline)
        scaled_inputs.append(interpolated)

    scaled_inputs = torch.cat(scaled_inputs, dim=0)

    all_gradients = []
    for i in range(0, scaled_inputs.shape[0], batch_size):
        batch = scaled_inputs[i:i + batch_size]
        _, grads = compute_gradients(model, batch, target_class)
        all_gradients.append(grads)

    all_gradients = torch.cat(all_gradients, dim=0)
    all_gradients = all_gradients.view(steps + 1, *input_tensor.shape)

    weights = torch.ones(steps + 1, device=input_tensor.device)
    weights[0] = weights[-1] = 0.5
    weights = weights.view(-1, *([1] * len(input_tensor.shape)))

    avg_gradients = (all_gradients * weights).sum(dim=0) / steps
    integrated_grads = (input_tensor - baseline) * avg_gradients

    return integrated_grads


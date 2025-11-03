import numpy as np
import torch

def compute_gradients(model, inputs, target_class=None):
    inputs = inputs.requires_grad_(True)
    output = model(inputs)
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    if isinstance(target_class, int):
        target = output[:, target_class]
    else:
        target = output.gather(1, target_class.view(-1, 1)).squeeze()
    
    model.zero_grad()
    target.sum().backward(retain_graph=True)
    gradients = inputs.grad.detach().clone()
    
    return output.detach(), gradients

def spatiotemporal_integrated_gradients(model, input_tensor, baseline=None, 
                                       target_class=None, steps_per_segment=20):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    assert baseline.shape == input_tensor.shape, "baseline, input shapes mismatch"
    
    device = input_tensor.device
    T = input_tensor.size(1)
    sig_attributions = torch.zeros_like(input_tensor)
    M = steps_per_segment - 1
    
    for t in range(1, T + 1):
        betas = torch.linspace(0, 1, steps_per_segment, device=device)
        segment_gradients = []
        
        for beta in betas:
            X_partial = baseline.clone()
            
            if t > 1:
                X_partial[:, :t-1] = input_tensor[:, :t-1]
            
            current_frame_idx = t - 1
            X_partial[:, current_frame_idx] = baseline[:, current_frame_idx] + beta * \
                                              (input_tensor[:, current_frame_idx] - baseline[:, current_frame_idx])
            
            X_partial.requires_grad_(True)
            _, grads = compute_gradients(model, X_partial, target_class)
            grad_t = grads[:, current_frame_idx].clone()
            segment_gradients.append(grad_t)
        
        segment_gradients = torch.stack(segment_gradients, dim=0)
        IG_t = torch.zeros_like(segment_gradients[0])
        
        for j in range(M):
            IG_t += 0.5 * (segment_gradients[j] + segment_gradients[j + 1])
        
        IG_t = IG_t / M
        frame_diff = input_tensor[:, current_frame_idx] - baseline[:, current_frame_idx]
        SIG_t = frame_diff * IG_t
        sig_attributions[:, current_frame_idx] = SIG_t
    
    return sig_attributions

#The following methods (batched and blocked) are implemented for speeding up wall clock time

def spaitotemporal_integrated_gradients_batched(model, input_tensor, baseline=None, target_class=None,steps_per_segment=50, block_size=1):

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    assert baseline.shape == input_tensor.shape, "input and baseline mismatch"

    N, T, H, W  = input_tensor.shape

    device=input_tensor.device

    K = block_size
    T_blocks = T // K
    steps_per_segment = int(steps_per_segment)
    M = steps_per_segment - 1

    sig_attributions = torch.zeros_like(input_tensor)

    betas = torch.linspace(0, 1, steps_per_segment, device=device)

    for k in range(T_blocks):

        start_idx = k*K
        end_idx = (k+1) * K

        interpolated_inputs = []

        for beta in betas:

            X_partial = baseline.clone()

            if k > 0:
                X_partial[:, :start_idx, :, :] = input_tensor[:, :start_idx, :, :]

            current_block_input = input_tensor[:, start_idx:end_idx, :, :]
            current_block_baseline = baseline[:, start_idx:end_idx, :, :]

            X_partial[:, start_idx:end_idx, :, :] = current_block_baseline + beta * (current_block_input - current_block_baseline)

            interpolated_inputs.append(X_partial)

        input_batch = torch.cat(interpolated_inputs, dim=0)

        _, grads_batch = compute_gradients(model, input_tensor, target_class)

        grads_reshaped = grads_batch.view(steps_per_segment, N, T, H, W)

        segment_gradients = grads_reshaped[:, :, start_idx:end_idx, :, :]

        IG_k_sum = torch.zeros_like(segment_gradients[0])
       
        for j in range(M):
            IG_k_sum += 0.5 * (segment_gradients[j] + segment_gradients[j + 1])
            
            IG_k = IG_k_sum / M

        block_diff = current_block_input - current_block_baseline
        SIG_k = block_diff * IG_k

        sig_attributions[:, start_idx:end_idx, :,:] = SIG_k

    return sig_attributions

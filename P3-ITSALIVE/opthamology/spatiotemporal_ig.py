import torch

def compute_gradients(model, inputs, target_class=None):
    # This function remains the same as your existing, correct implementation
    inputs = inputs.requires_grad_(True)
    output = model(inputs)
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    if isinstance(target_class, int):
        target = output[:, target_class]
    else:
        target = output.gather(1, target_class.view(-1, 1)).squeeze()
    
    model.zero_grad()
    # We use retain_graph=True in case the batch size for scaled inputs is > 1
    # and the loop attempts to backprop through the same model state multiple times
    target.sum().backward(retain_graph=True)
    
    return output.detach(), inputs.grad.detach()

def stig_path_interpolation(input_tensor, baseline, alpha):
    """
    Constructs the STIG piecewise path r*(alpha) for a given alpha.
    """
    B, T, D = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2:]
    device = input_tensor.device
    
    # 1. Determine the current segment t
    # T is the total number of time steps (frames)
    T_float = float(T)
    
    # Calculate t, the current segment index (1-indexed)
    # t = ceil(alpha * T)
    t = torch.ceil(alpha * T_float).long().clip(min=1, max=T)
    
    # Handle the start (alpha=0) separately, where t=1
    if alpha.item() == 0.0:
        t = torch.tensor(1).to(device)
    
    # 2. Calculate the scaled parameter beta
    # beta = alpha * T - (t - 1)
    t_minus_1 = t - 1
    beta = alpha * T_float - t_minus_1.float()

    # X_t^partial: frames 1 to t are input, t+1 to T are baseline
    X_t_partial = baseline.clone()
    if t.item() > 0:
        X_t_partial[:, :t.item()] = input_tensor[:, :t.item()]
    
    # X_{t-1}^partial: frames 1 to t-1 are input, t to T are baseline
    X_t_minus_1_partial = baseline.clone()
    if t_minus_1.item() > 0:
        X_t_minus_1_partial[:, :t_minus_1.item()] = input_tensor[:, :t_minus_1.item()]
    
    # 3. Construct the path r*(alpha)
    # r*(alpha) = X_{t-1}^partial + beta * (X_t^partial - X_{t-1}^partial)
    path_r_alpha = X_t_minus_1_partial + beta * (X_t_partial - X_t_minus_1_partial)

    return path_r_alpha, t.item()

def spatiotemporal_integrated_gradients_corrected(model, input_tensor, baseline=None, 
                                                  target_class=None, steps=50):
    
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    assert baseline.shape == input_tensor.shape, "Baseline and input shapes must match"
    
    device = input_tensor.device
    T = input_tensor.size(1) # Number of time steps (frames)
    
    # Use steps + 1 for trapezoidal rule
    alphas = torch.linspace(0, 1, steps + 1, device=device)
    
    # Accumulate gradients multiplied by a weight (for trapezoidal sum)
    accumulated_weighted_gradients = torch.zeros_like(input_tensor)
    
    # Store the difference for the final multiplication
    diff = input_tensor - baseline
    
    for i, alpha in enumerate(alphas):
        # Path and segment t
        interpolated_input, t = stig_path_interpolation(input_tensor, baseline, alpha)
        
        # Compute gradient on the path
        _, gradients = compute_gradients(model, interpolated_input, target_class)
        
        # Apply trapezoidal rule weights
        weight = 1.0
        if i == 0 or i == steps:
            weight = 0.5
            
        accumulated_weighted_gradients += gradients * weight
    
    # The integral term is the accumulated weighted gradient sum divided by the number of steps (M)
    # The integration factor T * d_alpha / T cancels out, leaving us with a standard averaging/scaling factor.
    avg_gradients = accumulated_weighted_gradients / steps
    
    # The final STIG is frame-specific, using the integral and the frame difference
    # SIG_i(x) = (x_i - x_i') * integral_term_for_x_i
    stig_attributions = diff * avg_gradients

    return stig_attributions
# import numpy as np
# import torch

# def compute_gradients(model, inputs, target_class=None):
#     inputs = inputs.requires_grad_(True)
#     output = model(inputs)
    
#     if target_class is None:
#         target_class = output.argmax(dim=1)
    
#     if isinstance(target_class, int):
#         target = output[:, target_class]
#     else:
#         target = output.gather(1, target_class.view(-1, 1)).squeeze()
    
#     model.zero_grad()
#     target.sum().backward(retain_graph=True)
#     gradients = inputs.grad.detach().clone()
    
#     return output.detach(), gradients

# def spatiotemporal_integrated_gradients(model, input_tensor, baseline=None, 
#                                        target_class=None, steps_per_segment=20):
#     if baseline is None:
#         baseline = torch.zeros_like(input_tensor)
    
#     assert baseline.shape == input_tensor.shape, "baseline, input shapes mismatch"
    
#     device = input_tensor.device
#     T = input_tensor.size(1)
#     sig_attributions = torch.zeros_like(input_tensor)
#     M = steps_per_segment - 1
    
#     for t in range(1, T + 1):
#         betas = torch.linspace(0, 1, steps_per_segment, device=device)
#         segment_gradients = []
        
#         for beta in betas:
#             X_partial = baseline.clone()
            
#             if t > 1:
#                 X_partial[:, :t-1] = input_tensor[:, :t-1]
            
#             current_frame_idx = t - 1
#             X_partial[:, current_frame_idx] = baseline[:, current_frame_idx] + beta * \
#                                               (input_tensor[:, current_frame_idx] - baseline[:, current_frame_idx])
            
#             X_partial.requires_grad_(True)
#             _, grads = compute_gradients(model, X_partial, target_class)
#             grad_t = grads[:, current_frame_idx].clone()
#             segment_gradients.append(grad_t)
        
#         segment_gradients = torch.stack(segment_gradients, dim=0)
#         IG_t = torch.zeros_like(segment_gradients[0])
        
#         for j in range(M):
#             IG_t += 0.5 * (segment_gradients[j] + segment_gradients[j + 1])
        
#         IG_t = IG_t / M
#         frame_diff = input_tensor[:, current_frame_idx] - baseline[:, current_frame_idx]
#         SIG_t = frame_diff * IG_t
#         sig_attributions[:, current_frame_idx] = SIG_t
    
#     return sig_attributions

# #The following methods (batched and blocked) are implemented for speeding up wall clock time

# def spaitotemporal_integrated_gradients_batched(model, input_tensor, baseline=None, target_class=None,steps_per_segment=50, block_size=1):

#     if baseline is None:
#         baseline = torch.zeros_like(input_tensor)

#     assert baseline.shape == input_tensor.shape, "input and baseline mismatch"

#     N, T, H, W  = input_tensor.shape

#     device=input_tensor.device

#     K = block_size
#     T_blocks = T // K
#     steps_per_segment = int(steps_per_segment)
#     M = steps_per_segment - 1

#     sig_attributions = torch.zeros_like(input_tensor)

#     betas = torch.linspace(0, 1, steps_per_segment, device=device)

#     for k in range(T_blocks):

#         start_idx = k*K
#         end_idx = (k+1) * K

#         interpolated_inputs = []

#         for beta in betas:

#             X_partial = baseline.clone()

#             if k > 0:
#                 X_partial[:, :start_idx, :, :] = input_tensor[:, :start_idx, :, :]

#             current_block_input = input_tensor[:, start_idx:end_idx, :, :]
#             current_block_baseline = baseline[:, start_idx:end_idx, :, :]

#             X_partial[:, start_idx:end_idx, :, :] = current_block_baseline + beta * (current_block_input - current_block_baseline)

#             interpolated_inputs.append(X_partial)

#         input_batch = torch.cat(interpolated_inputs, dim=0)

#         _, grads_batch = compute_gradients(model, input_tensor, target_class)

#         grads_reshaped = grads_batch.view(steps_per_segment, N, T, H, W)

#         segment_gradients = grads_reshaped[:, :, start_idx:end_idx, :, :]

#         IG_k_sum = torch.zeros_like(segment_gradients[0])
       
#         for j in range(M):
#             IG_k_sum += 0.5 * (segment_gradients[j] + segment_gradients[j + 1])
            
#             IG_k = IG_k_sum / M

#         block_diff = current_block_input - current_block_baseline
#         SIG_k = block_diff * IG_k

#         sig_attributions[:, start_idx:end_idx, :,:] = SIG_k

#     return sig_attributions
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
                                                  target_class=None, steps=20):
    
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

import torch

def modified_stig_with_temporal_masking(model, input_tensor, baseline=None, 
                                       target_class=None, steps=50):
    """
    Modified STIG that truly accounts for temporal dependencies
    by masking future information during interpolation
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    batch_size, T, features = input_tensor.shape
    attributions = torch.zeros_like(input_tensor)
    
    # For each timestep, compute its marginal contribution
    for t in range(T):
        accumulated_grads = torch.zeros(batch_size, features).to(input_tensor.device)
        
        for i in range(steps + 1):
            alpha = i / steps
            
            # Create interpolated input with temporal masking
            interpolated = baseline.clone()
            
            # Past timesteps are at full input value (history)
            if t > 0:
                interpolated[:, :t, :] = input_tensor[:, :t, :]
            
            # Current timestep is interpolated
            interpolated[:, t, :] = baseline[:, t, :] + alpha * (input_tensor[:, t, :] - baseline[:, t, :])
            
            # Future timesteps remain at baseline (no peeking)
            # This is the KEY DIFFERENCE from standard STIG
            
            interpolated.requires_grad_(True)
            output = model(interpolated)
            
            if target_class is None:
                target = output.sum()
            else:
                target = output[:, target_class].sum()
            
            model.zero_grad()
            target.backward()
            
            # Get gradient only for current timestep
            grad_t = interpolated.grad[:, t, :]
            
            # Trapezoidal rule
            weight = 0.5 if i == 0 or i == steps else 1.0
            accumulated_grads += grad_t * weight
        
        avg_grads = accumulated_grads / steps
        
        # Marginal attribution for timestep t given history
        attributions[:, t, :] = (input_tensor[:, t, :] - baseline[:, t, :]) * avg_grads
    
    return attributions

def history_aware_stig(model, input_tensor, baseline=None, target_class=None, steps=50):
    """
    STIG with history-dependent baseline that captures temporal flow
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    batch_size, T, features = input_tensor.shape
    attributions = torch.zeros_like(input_tensor)
    
    # Compute model output at each partial sequence
    partial_outputs = []
    for t in range(T + 1):
        if t == 0:
            partial_input = baseline.clone()
        else:
            partial_input = baseline.clone()
            partial_input[:, :t, :] = input_tensor[:, :t, :]
        
        with torch.no_grad():
            output = model(partial_input)
            partial_outputs.append(output)
    
    # For each timestep, compute attribution using history-dependent baseline
    for t in range(T):
        # The baseline for timestep t is the model's state after seeing 0:t-1
        if t == 0:
            history_baseline = baseline[:, 0, :]
            history_output = partial_outputs[0]
        else:
            # Use the actual input up to t-1 as the historical context
            history_baseline = baseline[:, t, :]  # Still baseline value at t
            history_output = partial_outputs[t]  # But model has seen 0:t-1
        
        # Now interpolate only timestep t, given the history
        accumulated_grads = torch.zeros(batch_size, features).to(input_tensor.device)
        
        for i in range(steps + 1):
            alpha = i / steps
            
            interpolated = baseline.clone()
            if t > 0:
                interpolated[:, :t, :] = input_tensor[:, :t, :]
            
            # Interpolate current timestep
            interpolated[:, t, :] = history_baseline + alpha * (input_tensor[:, t, :] - history_baseline)
            
            interpolated.requires_grad_(True)
            output = model(interpolated)
            
            # Target is the change from history_output
            if target_class is None:
                target = (output - history_output.detach()).sum()
            else:
                target = (output[:, target_class] - history_output[:, target_class].detach()).sum()
            
            model.zero_grad()
            target.backward()
            
            grad_t = interpolated.grad[:, t, :]
            weight = 0.5 if i == 0 or i == steps else 1.0
            accumulated_grads += grad_t * weight
        
        avg_grads = accumulated_grads / steps
        attributions[:, t, :] = (input_tensor[:, t, :] - history_baseline) * avg_grads
    
    return attributions

def nonlinear_temporal_stig(model, input_tensor, baseline=None, target_class=None, 
                           steps=50, temporal_decay=0.5):
    """
    STIG with non-linear temporal interpolation that weights recent history more
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    batch_size, T, features = input_tensor.shape
    
    # Define non-linear temporal interpolation function
    def temporal_interpolation_path(alpha, t, T):
        """
        Non-linear function that interpolates differently based on temporal position
        Early timesteps interpolate faster, later ones slower
        """
        # Exponential decay based on position
        position_weight = np.exp(-temporal_decay * t / T)
        
        # Modified interpolation that depends on temporal position
        if alpha < position_weight:
            # Faster interpolation for early timesteps
            return alpha / position_weight
        else:
            # Slower for later timesteps
            return 1.0
    
    accumulated_grads = torch.zeros_like(input_tensor)
    
    for i in range(steps + 1):
        alpha = i / steps
        
        # Create temporally-aware interpolation
        interpolated = baseline.clone()
        
        for t in range(T):
            # Each timestep has its own interpolation schedule
            t_alpha = temporal_interpolation_path(alpha, t, T)
            interpolated[:, t, :] = baseline[:, t, :] + t_alpha * (input_tensor[:, t, :] - baseline[:, t, :])
        
        interpolated.requires_grad_(True)
        output = model(interpolated)
        
        if target_class is None:
            target = output.sum()
        else:
            target = output[:, target_class].sum()
        
        model.zero_grad()
        target.backward()
        
        # Weight by derivative of the path
        path_derivatives = torch.zeros(batch_size, T, features).to(input_tensor.device)
        for t in range(T):
            if alpha < np.exp(-temporal_decay * t / T):
                path_derivatives[:, t, :] = 1.0 / np.exp(-temporal_decay * t / T)
            else:
                path_derivatives[:, t, :] = 0.0
        
        weighted_grads = interpolated.grad * path_derivatives
        
        weight = 0.5 if i == 0 or i == steps else 1.0
        accumulated_grads += weighted_grads * weight
    
    avg_grads = accumulated_grads / steps
    attributions = (input_tensor - baseline) * avg_grads
    
    return attributions


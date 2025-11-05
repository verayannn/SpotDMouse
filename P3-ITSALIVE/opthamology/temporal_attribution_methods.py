import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

def temporal_masked_stig(model, input_tensor, baseline=None, target_class=None, steps=50):
    """
    Modified STIG that respects temporal causality by masking future information.
    Works with your existing Feedforward, GRU, and LSTM models.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    device = input_tensor.device
    
    # Handle both 2D and 3D inputs
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)
        baseline = baseline.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, seq_len, features = input_tensor.shape
    attributions = torch.zeros_like(input_tensor)
    
    # For each timestep, compute its marginal contribution
    for t in range(seq_len):
        accumulated_grads = torch.zeros(batch_size, features).to(device)
        
        for i in range(steps + 1):
            alpha = i / steps
            
            # Create interpolated input with temporal masking
            interpolated = baseline.clone()
            
            # Past timesteps are at full input value (history)
            if t > 0:
                interpolated[:, :t, :] = input_tensor[:, :t, :]
            
            # Current timestep is interpolated
            interpolated[:, t, :] = baseline[:, t, :] + alpha * (input_tensor[:, t, :] - baseline[:, t, :])
            
            # Future timesteps remain at baseline (no peeking ahead)
            # Already set by cloning baseline
            
            interpolated.requires_grad_(True)
            
            # Set model to train mode for RNNs
            model_mode = model.training
            model.train()
            
            output = model(interpolated)
            
            if target_class is None:
                target = output.sum()
            else:
                if output.dim() > 1 and output.size(1) > 1:
                    target = output[:, target_class].sum()
                else:
                    target = output.sum()
            
            model.zero_grad()
            target.backward()
            
            # Get gradient only for current timestep
            grad_t = interpolated.grad[:, t, :]
            
            # Restore model mode
            model.train(model_mode)
            
            # Trapezoidal rule
            weight = 0.5 if i == 0 or i == steps else 1.0
            accumulated_grads += grad_t * weight
        
        avg_grads = accumulated_grads / steps
        
        # Marginal attribution for timestep t given history
        attributions[:, t, :] = (input_tensor[:, t, :] - baseline[:, t, :]) * avg_grads
    
    if squeeze_output:
        attributions = attributions.squeeze(0)
    
    return attributions

def incremental_stig(model, input_tensor, baseline=None, target_class=None, steps=20):
    """
    Computes attribution by measuring incremental contribution of each timestep.
    Particularly effective for RNNs/LSTMs.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    device = input_tensor.device
    
    # Handle dimensions
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)
        baseline = baseline.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, seq_len, features = input_tensor.shape
    attributions = torch.zeros_like(input_tensor)
    
    # Store outputs at each timestep
    model_mode = model.training
    model.eval()
    
    with torch.no_grad():
        # Get baseline output
        baseline_output = model(baseline)
        
        # Get incremental outputs
        incremental_outputs = []
        for t in range(seq_len + 1):
            if t == 0:
                partial_input = baseline.clone()
            else:
                partial_input = baseline.clone()
                partial_input[:, :t, :] = input_tensor[:, :t, :]
            
            output = model(partial_input)
            incremental_outputs.append(output)
    
    model.train(model_mode)
    
    # Now compute attributions for each timestep
    for t in range(seq_len):
        # The contribution of timestep t is the difference it makes
        # when added to the sequence
        if t == 0:
            prev_output = baseline_output
        else:
            prev_output = incremental_outputs[t]
        
        curr_output = incremental_outputs[t + 1]
        
        # Compute integrated gradients for this specific transition
        accumulated_grads = torch.zeros(batch_size, features).to(device)
        
        for i in range(steps + 1):
            alpha = i / steps
            
            # Interpolate only at timestep t
            interpolated = baseline.clone()
            if t > 0:
                interpolated[:, :t, :] = input_tensor[:, :t, :]
            
            interpolated[:, t, :] = baseline[:, t, :] + alpha * (input_tensor[:, t, :] - baseline[:, t, :])
            
            interpolated.requires_grad_(True)
            
            model.train()
            output = model(interpolated)
            
            # Target is to match the incremental change
            if target_class is None:
                target = output.sum()
            else:
                if output.dim() > 1 and output.size(1) > 1:
                    target = output[:, target_class].sum()
                else:
                    target = output.sum()
            
            model.zero_grad()
            target.backward()
            
            grad_t = interpolated.grad[:, t, :]
            
            weight = 0.5 if i == 0 or i == steps else 1.0
            accumulated_grads += grad_t * weight
        
        avg_grads = accumulated_grads / steps
        
        # Scale by actual output change
        output_change = (curr_output - prev_output).abs().mean().item()
        
        # Attribution weighted by output change
        attributions[:, t, :] = (input_tensor[:, t, :] - baseline[:, t, :]) * avg_grads * (1 + output_change)
    
    model.train(model_mode)
    
    if squeeze_output:
        attributions = attributions.squeeze(0)
    
    return attributions

def attention_weighted_stig(model, input_tensor, baseline=None, target_class=None, steps=50):
    """
    STIG that weights attributions by temporal importance.
    Uses gradient magnitudes to estimate attention/importance.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    device = input_tensor.device
    
    # Handle dimensions
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)
        baseline = baseline.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # First, compute temporal importance weights
    temporal_weights = compute_temporal_importance_weights(
        model, input_tensor, baseline, target_class
    )
    
    # Then compute standard IG
    from standard_ig import integrated_gradients as ig
    standard_attr = ig(model, input_tensor.squeeze(0) if squeeze_output else input_tensor, 
                      baseline.squeeze(0) if squeeze_output else baseline, 
                      target_class, steps)
    
    if standard_attr.dim() == 2 and not squeeze_output:
        standard_attr = standard_attr.unsqueeze(0)
    
    # Weight by temporal importance
    weighted_attr = standard_attr * temporal_weights
    
    # Now do temporal-aware interpolation
    batch_size, seq_len, features = input_tensor.shape
    accumulated_grads = torch.zeros_like(input_tensor)
    
    for i in range(steps + 1):
        alpha = i / steps
        
        # Create temporally-weighted interpolation
        interpolated = baseline.clone()
        
        for t in range(seq_len):
            # Use cumulative importance to determine interpolation progress
            if t == 0:
                t_alpha = alpha * temporal_weights[:, 0, :].mean()
            else:
                cumulative_weight = temporal_weights[:, :t+1, :].mean(dim=2).sum(dim=1, keepdim=True)
                cumulative_weight = cumulative_weight / temporal_weights[:, :, :].mean(dim=2).sum(dim=1, keepdim=True)
                t_alpha = alpha * cumulative_weight
            
            t_alpha = t_alpha.clamp(0, 1)
            
            if t_alpha.dim() == 2:
                t_alpha = t_alpha.unsqueeze(-1)
            
            interpolated[:, t, :] = baseline[:, t, :] + t_alpha * (input_tensor[:, t, :] - baseline[:, t, :])
        
        interpolated.requires_grad_(True)
        
        model_mode = model.training
        model.train()
        
        output = model(interpolated)
        
        if target_class is None:
            target = output.sum()
        else:
            if output.dim() > 1 and output.size(1) > 1:
                target = output[:, target_class].sum()
            else:
                target = output.sum()
        
        model.zero_grad()
        target.backward()
        
        model.train(model_mode)
        
        weight = 0.5 if i == 0 or i == steps else 1.0
        accumulated_grads += interpolated.grad * weight
    
    avg_grads = accumulated_grads / steps
    final_attr = (input_tensor - baseline) * avg_grads * temporal_weights
    
    if squeeze_output:
        final_attr = final_attr.squeeze(0)
    
    return final_attr

def compute_temporal_importance_weights(model, input_tensor, baseline, target_class=None):
    """
    Compute importance weights for each timestep.
    """
    device = input_tensor.device
    batch_size, seq_len, features = input_tensor.shape
    
    importance_scores = torch.zeros(batch_size, seq_len).to(device)
    
    model_mode = model.training
    model.train()
    
    for t in range(seq_len):
        # Create input with only up to timestep t
        masked_input = baseline.clone()
        if t >= 0:
            masked_input[:, :t+1, :] = input_tensor[:, :t+1, :]
        
        masked_input.requires_grad_(True)
        
        output = model(masked_input)
        
        if target_class is None:
            target = output.sum()
        else:
            if output.dim() > 1 and output.size(1) > 1:
                target = output[:, target_class].sum()
            else:
                target = output.sum()
        
        model.zero_grad()
        target.backward()
        
        # Importance is gradient magnitude at timestep t
        if masked_input.grad is not None:
            importance_scores[:, t] = masked_input.grad[:, t, :].abs().mean(dim=-1)
    
    model.train(model_mode)
    
    # Normalize
    importance_scores = importance_scores / (importance_scores.sum(dim=1, keepdim=True) + 1e-8)
    
    # Expand to match input dimensions
    return importance_scores.unsqueeze(-1).expand(-1, -1, features)
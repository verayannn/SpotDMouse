import numpy as np
import torch

def compute_gradients(model, inputs, target_class=None):

    inputs = inputs.requires_grad_(True)
    output = model(inputs)

    if target_class is None:
        target_class = output.argmax(dim=1)

    if isinstance(target_class, int):
        target = output[:,target_class]
    else:
        target = output.gather(1,target_class.view(-1,1)).squeeze()

    model.zero_grad()
    target.sum().backward(retain_graph=True)

    gradients = inputs.grad.detach().clone()

    return output.detach(), gradients

def spatiotemporal_integrated_gradients(model, input_tensor, interval=5, baseline=None, target_class=None, steps=50):

    #Initialize X'
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    assert baseline.shape == input_tensor.shape, "baseline, input shapes mismatch"

    device = input_tensor.device
    batch_size = input_tensor.size(0)
    T = input_tensor.size(1)

    sig_attributions = torch.zeros_like(input_tensor)

    for t in range(1, T + 1):

        betas = torch.linspace(0, 1, steps_per_segment, device=device)

        segment_gradients = []

        for beta in betas:

            X_partial = baseline.clone()

            if t > 1:
                X_partial[:, :t-1] =  input_tensor[:, :t-1]

            X_partial[:, t-1] = baseline[:, t-1] + beta * (input_tensor[:, t-1] - baseline[:, t-1])
            X_partial.requires_grad_(True)

            _, grads = compute_gradients(model, X_partial, target_class)

            grad_t = grads[:, t-1].clone()

            segment_gradients.append(grad_t)

        segment_gradients = torch.stack(segment_gradients, dim=0)

        IG_t = torch.zeros_like(segment_gradients[0])

        for j in range(steps_per_segment - 1):
            IG_t += 0.5 * (segment_gradients[j] + segment_gradients[j + 1])

        IG_t = T * IG_t / (steps_per_segment - 1)

        frame_diff = input_tensor[:, t-1] - baseline[:, t-1]
        SIG_t = frames_diff * IG_t

        sig_attributions[:, t-1] = SIG_t

    return sig_attributions 


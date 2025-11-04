import numpy as np
import torch
import voltron
from PIL import Image
import numpy as np
import torch
from standard_ig import integrated_gradients as ig
from spatiotemporal_ig import spatiotemporal_integrated_gradients as stig
import torchvision.transforms as T
import matplotlib.pyplot as plt
import voltron.nn as vnn

def compute_continuous_attributions_comparison(model, stimulus, target_unit, 
                                             window_size=30, stride=1,
                                             baseline=None):
    """Compare IG and STIG with sliding windows"""
    
    n_frames = len(stimulus)
    ig_attributions = []
    stig_attributions = []
    responses = []
    
    if baseline is None:
        baseline = torch.zeros_like(stimulus)
    
    # Slide through the entire stimulus
    for t in range(0, n_frames - window_size + 1, stride):
        # Extract windows
        window = stimulus[t:t+window_size].unsqueeze(0)
        baseline_window = baseline[t:t+window_size].unsqueeze(0)
        
        # Standard IG
        ig_attr = ig(model, window, baseline=baseline_window, target_class=target_unit)
        
        # Spatiotemporal IG
        stig_attr = stig(model, window, baseline=baseline_window, target_class=target_unit)
        
        # Get model response
        with torch.no_grad():
            response = model(window)[0, target_unit]
        
        ig_attributions.append(ig_attr)
        stig_attributions.append(stig_attr)
        responses.append(response)
    
    return (torch.stack(ig_attributions), 
            torch.stack(stig_attributions), 
            torch.tensor(responses))

# Visualize the differences
def visualize_attribution_differences(ig_attrs, stig_attrs, responses, 
                                    flash_frame_idx=14):
    """Plot to highlight differences"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Model responses over time
    axes[0].plot(responses.cpu().numpy())
    axes[0].axvline(x=flash_frame_idx, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Model Response vs Window Position')
    axes[0].set_xlabel('Window Start Position')
    axes[0].set_ylabel('Response')
    
    # 2. Attribution strength at flash location
    ig_flash_attrs = []
    stig_flash_attrs = []
    
    for t in range(len(ig_attrs)):
        # Where is the flash in this window?
        flash_pos_in_window = flash_frame_idx - t
        if 0 <= flash_pos_in_window < 30:
            ig_flash_attrs.append(
                ig_attrs[t, 0, flash_pos_in_window, 50, 50].item()
            )
            stig_flash_attrs.append(
                stig_attrs[t, 0, flash_pos_in_window, 50, 50].item()
            )
        else:
            ig_flash_attrs.append(0)
            stig_flash_attrs.append(0)
    
    axes[1].plot(ig_flash_attrs, label='IG', alpha=0.7)
    axes[1].plot(stig_flash_attrs, label='STIG', alpha=0.7)
    axes[1].axvline(x=flash_frame_idx, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Attribution at Flash Location')
    axes[1].set_xlabel('Window Start Position')
    axes[1].set_ylabel('Attribution Magnitude')
    axes[1].legend()
    
    # 3. Difference map
    diff = np.array(stig_flash_attrs) - np.array(ig_flash_attrs)
    axes[2].plot(diff)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2].axvline(x=flash_frame_idx, color='r', linestyle='--', alpha=0.5)
    axes[2].set_title('STIG - IG Difference')
    axes[2].set_xlabel('Window Start Position')
    axes[2].set_ylabel('Attribution Difference')
    
    plt.tight_layout()
    return fig

def create_full_attribution_videos(ig_attrs, stig_attrs, output_path="attribution_videos"):
    """Create videos for each frame position showing how its attribution changes"""
    
    import os
    os.makedirs(f"{output_path}/frame_evolution", exist_ok=True)
    
    # For specific frame positions, show how attribution changes
    frame_positions = [0, 10, 14, 20, 29]  # Key positions including flash
    
    for frame_pos in frame_positions:
        ig_sequence = []
        stig_sequence = []
        
        # Collect attributions for this frame position across all windows
        for t in range(len(ig_attrs)):
            if frame_pos < ig_attrs[t].shape[2]:  # Check if frame exists in this window
                ig_sequence.append(ig_attrs[t, 0, frame_pos, :, :].cpu().numpy())
                stig_sequence.append(stig_attrs[t, 0, frame_pos, :, :].cpu().numpy())
        
        if len(ig_sequence) > 0:
            voltron.frames2gif(np.array(ig_sequence), 
                             f"{output_path}/frame_evolution/ig_frame_{frame_pos}.gif")
            voltron.frames2gif(np.array(stig_sequence), 
                             f"{output_path}/frame_evolution/stig_frame_{frame_pos}.gif")

def create_full_attribution_videos(ig_attrs, stig_attrs, output_path="attribution_videos"):
    """Create videos for each frame position showing how its attribution changes"""
    
    import os
    os.makedirs(f"{output_path}/frame_evolution", exist_ok=True)
    
    # For specific frame positions, show how attribution changes
    frame_positions = [0, 10, 14, 20, 29]  # Key positions including flash
    
    for frame_pos in frame_positions:
        ig_sequence = []
        stig_sequence = []
        
        # Collect attributions for this frame position across all windows
        for t in range(len(ig_attrs)):
            if frame_pos < ig_attrs[t].shape[2]:  # Check if frame exists in this window
                ig_sequence.append(ig_attrs[t, 0, frame_pos, :, :].cpu().numpy())
                stig_sequence.append(stig_attrs[t, 0, frame_pos, :, :].cpu().numpy())
        
        if len(ig_sequence) > 0:
            voltron.frames2gif(np.array(ig_sequence), 
                             f"{output_path}/frame_evolution/ig_frame_{frame_pos}.gif")
            voltron.frames2gif(np.array(stig_sequence), 
                             f"{output_path}/frame_evolution/stig_frame_{frame_pos}.gif")

device = torch.device("cuda:0")

mantis_image_path = "/home/grandline/Downloads/mantis.jpg" #"/Users/javierweddington/Downloads/mantis.jpg"
mantis_image = Image.open(mantis_image_path)
mantis_image = mantis_image.convert("L")

transform = T.Compose([
    T.Resize((100,100)),
    T.ToTensor()
    ])

mantis_tensor = transform(mantis_image).squeeze(0)
mantis_frames = mantis_tensor.repeat(30,1,1).unsqueeze(0).to(device)
print("mantis frames shape: ", mantis_frames.shape)

#######################
FRAME_COUNT = 30
H, W = 100, 100
FLASH_FRAME_INDEX = 14 # Frame 15 is at index 14

# 1. Initialize a black baseline tensor [1, 30, 100, 100] (Batch, Time, H, W)
dynamic_flash_frames = torch.zeros(1, FRAME_COUNT, H, W).to(device)

# 2. Create the single white disk stimulus [100, 100]
flash_frame = torch.zeros(H, W) 
center_y, center_x = H // 2, W // 2
radius = 20 # Define the size of the disk

# Use a meshgrid to identify pixels inside the disk radius
Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
distance_from_center = torch.sqrt((X - center_x)**2 + (Y - center_x)**2)

# Set pixels within the radius to white (1.0)
flash_frame[distance_from_center <= radius] = 1.0 

# 3. Insert the flash into the sequence at Frame 15 (index 14)
# Frames 0-13 and 15-29 remain black (baseline)
dynamic_flash_frames[:, FLASH_FRAME_INDEX, :, :] = flash_frame.to(device) 

# Replace the original static input tensor with the new dynamic tensor
mantis_frames = dynamic_flash_frames

print("dynamic flash frames shape (used for mantis_frames): ", mantis_frames.shape)
#######################

model_pth = "/home/grandline/retinal/best_allstim_model.pt"
model = torch.load(model_pth, weights_only=False, map_location=device)
model.eval()

# Create stimulus with flash at frame 14
stimulus = torch.zeros(50, 100, 100).to(device)  # Longer than window
stimulus[14] = flash_frame.to(device)

# Compare methods
ig_attrs, stig_attrs, responses = compute_continuous_attributions_comparison(
    model, stimulus, target_unit=0, window_size=30
)

# Visualize
fig = visualize_attribution_differences(ig_attrs, stig_attrs, responses)
plt.savefig('ig_vs_stig_comparison.png')

create_full_attribution_videos(ig_attrs, stig_attrs)

def compute_attributions_with_sliding_window(model, full_stimulus, target_unit, window_size=30):
    """
    Compute attributions using the same sliding window approach as present_stim
    full_stimulus: shape (950, H, W) - long stimulus
    Returns: attributions for each output timepoint (900 attributions)
    """
    
    # Create dataset with sliding windows
    dataset = vnn.ArtificialStimulus(full_stimulus, window_size)
    
    all_ig_attrs = []
    all_stig_attrs = []
    all_responses = []
    
    # Process each window
    for i in range(len(dataset)):
        # Get window - this matches what the model sees
        window = dataset[i].unsqueeze(0).to(device)  # Shape: [1, 50, H, W]
        
        # Compute attributions for this window
        ig_attr = ig(model, window, target_class=target_unit)
        stig_attr = stig(model, window, target_class=target_unit)
        
        # Get response
        with torch.no_grad():
            response = model(window)[0, target_unit]
        
        # Store only the attribution for the "current" frame
        # This matches how present_stim gives you one output per window
        center_idx = window_size // 2  # or however your model determines output time
        all_ig_attrs.append(ig_attr[0, center_idx])
        all_stig_attrs.append(stig_attr[0, center_idx])
        all_responses.append(response)
    
    return (torch.stack(all_ig_attrs), 
            torch.stack(all_stig_attrs), 
            torch.tensor(all_responses))

# Use it exactly like present_stim
full_stimulus = torch.zeros(950, 100, 100).to(device)
full_stimulus[450] = flash_frame.to(device)  # Flash in middle

# This gives you 900 attributions, one for each output
ig_attrs, stig_attrs, responses = compute_attributions_with_sliding_window(
    model, full_stimulus, target_unit=0, window_size=30
)

print(f"Output shapes: {ig_attrs.shape}")  # Should be [900, 100, 100]

# Now you can make videos just like before
voltron.frames2gif(ig_attrs.cpu().numpy(), "ig_sliding_window.gif")
voltron.frames2gif(stig_attrs.cpu().numpy(), "stig_sliding_window.gif")


create_full_attribution_videos(ig_attrs, stig_attrs)

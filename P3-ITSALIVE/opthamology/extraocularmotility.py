import voltron
from PIL import Image
import numpy as np
import torch
from standard_ig import integrated_gradients as ig
from spatiotemporal_ig import spatiotemporal_integrated_gradients as stig
import torchvision.transforms as T
import matplotlib.pyplot as plt

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
# mantis_frames = dynamic_flash_frames

print("dynamic flash frames shape (used for mantis_frames): ", mantis_frames.shape)
#######################

model_pth = "/home/grandline/retinal/best_allstim_model.pt"
model = torch.load(model_pth, weights_only=False, map_location=device)
model.eval()

with torch.no_grad():
    output = model(mantis_frames)

print(f"output network shape:{output.shape}")

for k in range(15):
    stigs = stig(model, mantis_frames, target_class=k)

    print("Stigs shape",stigs.shape)

    plt.imshow(stigs.detach().cpu().numpy().squeeze(0)[0,:,:])
    plt.savefig(f"/home/grandline/spatiotemporal_ig_plots/first_frame_{k}")
    plt.close()

    stigs.detach().cpu().numpy().squeeze(0)[k,:,:]
    voltron.frames2gif(stigs.detach().cpu().numpy().squeeze(0), f"/home/grandline/spatiotemporal_ig_plots/frames_{k}.gif")

def test_motion_selectivity(model):
    """Test for differences between SIG and IG on motion stimulus"""
    
    # Get model device
    device = next(model.parameters()).device
    
    # Create motion stimulus
    def create_motion_stimulus(frames=30, height=100, width=100):
        video = torch.zeros(2, frames, height, width)
        for t in range(frames):
            x_pos = int(t * width / frames)
            if x_pos < width - 5:
                video[0, t, :, x_pos:x_pos+5] = 1.0
        return video.to(device)  # Move to device here
    
    def create_on_off_stimulus(frames=30, height=100, width=100):
        video = torch.zeros(1, frames, height, width)
        # ON period (frames 5-10)
        video[0, 5:10, 20:50, 20:50] = 1.0
        # OFF period (goes dark)
        # Another ON (frames 20-25)  
        video[0, 20:25, 20:50, 20:50] = 1.0
        return video.to(device)
    
    def create_temporal_frequency_stimulus(frames=30, height=100, width=100, frequency=2):
        """Flickering stimulus at specific frequency"""
        video = torch.zeros(1, frames, height, width)
        for t in range(frames):
            intensity = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * t / frames)
            video[0, t, :, :] = intensity
        return video.to(device)

    motion_video = create_motion_stimulus()
    
    # Run both attribution methods
    print("Computing SIG...")
    sig_attr = stig(model, motion_video)
    
    print("Computing IG...")
    ig_attr = ig(model, motion_video)
    
    # Compare results
    differences = []
    for unit in range(24):  # for each RGC unit
        # Get temporal profile at center pixel
        sig_temporal = sig_attr[0, :, :, :].cpu() if len(sig_attr.shape) == 4 else sig_attr[0, :].mean(dim=(1,2)).cpu()
        ig_temporal = ig_attr[0, :, :, :].cpu() if len(ig_attr.shape) == 4 else ig_attr[0, :].mean(dim=(1,2)).cpu()
        
        # Calculate difference
        diff = (sig_temporal - ig_temporal).abs().mean()
        differences.append(diff.item())
        
        if diff > 0.01:  # Threshold for meaningful difference
            print(f"Unit {unit}: Significant difference = {diff:.4f}")
    
    avg_diff = np.mean(differences)
    print(f"\nAverage difference across units: {avg_diff:.6f}")
    
    if avg_diff < 1e-6:
        print("No meaningful differences - model likely processes frames independently")
    else:
        print("Differences found - model has implicit temporal dependencies!")
    
    return sig_attr, ig_attr

# Run the test
sig_attr, ig_attr = test_motion_selectivity(model)
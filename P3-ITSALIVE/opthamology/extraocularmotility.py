import voltron
from PIL import Image
import numpy as np
import torch
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
mantis_frames = dynamic_flash_frames

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

from standard_ig import integrated_gradients
import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cpu")

mantis_image_path = "/Users/javierweddington/Downloads/mantis.jpg"
mantis_image = Image.open(mantis_image_path)
mantis_image = mantis_image.convert("L")

transform = T.Compose([
    T.Resize((68,102)),
    T.ToTensor()
    ])

mantis_tensor = transform(mantis_image).squeeze(0)
mantis_frames = mantis_tensor.repeat(60,1,1).unsqueeze(0).to(device)

print(f"Input shape:{mantis_frames.shape}")

#Plot example input frame
#plt.imshow(mantis_frames.squeeze(0)[0,:,:].detach().cpu().numpy())
#plt.show()

pt_pth = "/Users/javierweddington/cortical/best_charmander.pt"
network = torch.load(pt_pth, weights_only=False, map_location=device)
network.eval()

with torch.no_grad():
    output = network(mantis_frames)

print(f"Output shape {output.shape}")

# Compute Integrated Gradients for all 15 units
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

for unit in range(15):
    attributions = integrated_gradients(network, mantis_frames, target_class=unit)

    # Average across all 60 frames or show first frame
    attr_frame = attributions.squeeze(0)[0, :, :].detach().cpu().numpy()

    axes[unit].imshow(attr_frame, cmap='RdBu_r', vmin=-np.abs(attr_frame).max(),
                      vmax=np.abs(attr_frame).max())
    axes[unit].set_title(f'Unit {unit}\nActivation: {output[0, unit].item():.3f}')
    axes[unit].axis('off')

plt.tight_layout()
plt.suptitle('Integrated Gradients for all 15 Units', y=1.02, fontsize=16)
plt.show()

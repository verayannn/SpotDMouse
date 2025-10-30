from standard_ig import integrated_gradients
import models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda:1")#torch.device("cpu")

mantis_image_path = "/home/grandline/Downloads/mantis.jpg" #"/Users/javierweddington/Downloads/mantis.jpg"
mantis_image = Image.open(mantis_image_path)
mantis_image = mantis_image.convert("L")

'''
charmander input size: 60, 68, 102
goldroger input size: 50, 60, 112
retinal input size: 30, 100, 100

'''
transform = T.Compose([
    T.Resize((68,112)),
    T.ToTensor()
    ])

mantis_tensor = transform(mantis_image).squeeze(0)
mantis_frames = mantis_tensor.repeat(40,1,1).unsqueeze(0).to(device)

print(f"Input shape:{mantis_frames.shape}")

#Plot example input frame
#plt.imshow(mantis_frames.squeeze(0)[0,:,:].detach().cpu().numpy())
#plt.show()

#pt_pth = "/Users/javierweddington/retinal/best_allstim_model.pt"

network_name = "javier_cells_model"
pt_pth = "/home/grandline/cortical/javier_cells_model.pt" #/Users/javierweddington/cortical/best_charmander.pt"

network = torch.load(pt_pth, weights_only=False, map_location=device)
network.eval()

with torch.no_grad():
    output = network(mantis_frames)

print(f"Output shape {output.shape}")

original_img = mantis_tensor.detach().cpu().numpy()

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
plt.savefig(f"/home/grandline/standard_ig_plots/{network_name}_attrs.png")
plt.show()
plt.close()

fig2, axes2 = plt.subplots(3, 5, figsize=(20, 12))
axes2 = axes2.flatten()

for unit in range(15):
    attributions = integrated_gradients(network, mantis_frames, target_class=unit)
    attr_frame = attributions.squeeze(0)[0, :, :].detach().cpu().numpy()

    axes2[unit].imshow(original_img, cmap='gray')

    positive_attr = np.maximum(attr_frame, 0)
    threshold = np.percentile(positive_attr, 75)
    masked_attr = np.ma.masked_where(positive_attr < threshold, positive_attr)

    axes2[unit].imshow(masked_attr, cmap='Reds', alpha=0.7)
    axes2[unit].set_title(f'Unit {unit} - Top 25% Positive Attribution')
    axes2[unit].axis('off')

plt.tight_layout()
plt.suptitle('Top Positive Attributions Overlaid', y=1.0, fontsize=16)
plt.savefig(f"/home/grandline/standard_ig_plots/{network_name}_overlay.png")
plt.show()
plt.close()

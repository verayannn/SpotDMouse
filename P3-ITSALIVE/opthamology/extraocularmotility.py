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

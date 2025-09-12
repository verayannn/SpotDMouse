import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:1")

PTH = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models/best_model.pt"
OG_PTH = "/workspace/SpotDMouse/P2-Terrain_Challenge/sim2real/model_9999_with_stats.pt"
model = torch.load(PTH).to(DEVICE)
og_model = torch.load(OG_PTH).to(DEVICE)
# Compare model structures
print("=== Model Comparison ===")
print(f"Model type: {type(model)}")
print(f"OG Model type: {type(og_model)}")

# If they're state dicts, compare keys
if isinstance(model, dict) and isinstance(og_model, dict):
    print(f"\nModel keys: {len(model.keys())}")
    print(f"OG Model keys: {len(og_model.keys())}")
    
    # Compare parameter shapes
    print("\n=== Parameter Shape Comparison ===")
    for key in model.keys():
        if key in og_model:
            model_shape = model[key].shape
            og_shape = og_model[key].shape
            match = "✓" if model_shape == og_shape else "✗"
            print(f"{key}: {model_shape} vs {og_shape} {match}")
        else:
            print(f"{key}: Only in model")
    
    for key in og_model.keys():
        if key not in model:
            print(f"{key}: Only in og_model")

# If they're model objects, compare architecture
elif hasattr(model, 'state_dict') and hasattr(og_model, 'state_dict'):
    print(f"\nModel architecture: {model}")
    print(f"OG Model architecture: {og_model}")
    
    # Compare state dict shapes
    model_state = model.state_dict()
    og_state = og_model.state_dict()
    
    print("\n=== State Dict Shape Comparison ===")
    for key in model_state.keys():
        if key in og_state:
            model_shape = model_state[key].shape
            og_shape = og_state[key].shape
            match = "✓" if model_shape == og_shape else "✗"
            print(f"{key}: {model_shape} vs {og_shape} {match}")

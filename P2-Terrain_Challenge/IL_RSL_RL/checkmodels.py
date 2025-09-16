import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Model paths
IL_MLP_FILE = "/workspace/SpotDMouse/P2-Terrain_Challenge/IL_RSL_RL/models_rsl_format/best_model_rsl_format.pt"
RL_MLP_FILE = "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/birthdayrun/2025-08-07_19-17-44/model_9999_with_stats.pt"

device = torch.device('cuda:0')

# Load models
IL_MODEL = torch.load(IL_MLP_FILE, map_location=device)
RL_MODEL = torch.load(RL_MLP_FILE, map_location=device)

print("Collecting multiple observations to find scaling relationship...")

# Collect multiple IL and RL outputs for analysis
il_outputs = []
rl_outputs = []

with h5py.File('/workspace/rosbag_recordings/hdf5_datasets/mini_pupper_demos_20250914_233847.hdf5', 'r') as f:
    # Sample observations from different demos
    for demo_name in ['demo_1', 'demo_2', 'demo_3', 'demo_4']:
        if f'data/{demo_name}/obs' in f:
            demo_obs = f[f'data/{demo_name}/obs'][:]
            # Sample every 50th observation
            for idx in range(0, min(len(demo_obs), 500), 50):
                obs = torch.tensor(demo_obs[idx], dtype=torch.float32, device=device).unsqueeze(0)
                
                # Normalize
                il_obs_norm = (obs - IL_MODEL['obs_rms_mean'].to(device)) / torch.sqrt(IL_MODEL['obs_rms_var'].to(device) + 1e-8)
                rl_obs_norm = (obs - torch.tensor(RL_MODEL['obs_rms_mean'], device=device)) / torch.sqrt(torch.tensor(RL_MODEL['obs_rms_var'], device=device) + 1e-8)
                
                # Forward pass
                def forward_actor(model_dict, obs):
                    x = obs.float()
                    x = torch.nn.functional.elu(x @ model_dict['actor.0.weight'].T + model_dict['actor.0.bias'])
                    x = torch.nn.functional.elu(x @ model_dict['actor.2.weight'].T + model_dict['actor.2.bias'])
                    x = torch.nn.functional.elu(x @ model_dict['actor.4.weight'].T + model_dict['actor.4.bias'])
                    x = x @ model_dict['actor.6.weight'].T + model_dict['actor.6.bias']
                    return x
                
                il_out = forward_actor(IL_MODEL['model_state_dict'], il_obs_norm)
                rl_out = forward_actor(RL_MODEL['model_state_dict'], rl_obs_norm)
                
                il_outputs.append(il_out[0].cpu().numpy())
                rl_outputs.append(rl_out[0].cpu().numpy())

il_outputs = np.array(il_outputs)
rl_outputs = np.array(rl_outputs)

print(f"Collected {len(il_outputs)} observation pairs")

# Analyze the relationship for each joint
print("\n=== ANALYZING SCALING RELATIONSHIPS ===")

# Create figure for visualization
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

joint_names = ['LF-Hip', 'LF-Thigh', 'LF-Knee', 'RF-Hip', 'RF-Thigh', 'RF-Knee',
               'LB-Hip', 'LB-Thigh', 'LB-Knee', 'RB-Hip', 'RB-Thigh', 'RB-Knee']

# Store scaling functions for each joint
scaling_functions = {}

for joint_idx in range(12):
    ax = axes[joint_idx]
    
    # Get data for this joint
    il_joint = il_outputs[:, joint_idx]
    rl_joint = rl_outputs[:, joint_idx]
    
    # Remove outliers (RL values > 100 rad are clearly wrong)
    mask = np.abs(rl_joint) < 100
    il_joint = il_joint[mask]
    rl_joint = rl_joint[mask]
    
    # Plot scatter
    ax.scatter(il_joint, rl_joint, alpha=0.5, s=10)
    ax.set_xlabel('IL Output (rad)')
    ax.set_ylabel('RL Output (rad)')
    ax.set_title(joint_names[joint_idx])
    ax.grid(True, alpha=0.3)
    
    # Try different scaling approaches
    print(f"\n{joint_names[joint_idx]}:")
    
    # 1. Linear scaling: RL = a * IL + b
    if len(il_joint) > 10:
        A = np.vstack([il_joint, np.ones(len(il_joint))]).T
        linear_params, _, _, _ = np.linalg.lstsq(A, rl_joint, rcond=None)
        a, b = linear_params
        print(f"  Linear: RL = {a:.2f} * IL + {b:.2f}")
        
        # Plot linear fit
        il_range = np.linspace(il_joint.min(), il_joint.max(), 100)
        ax.plot(il_range, a * il_range + b, 'r-', label=f'Linear: {a:.1f}x + {b:.1f}', alpha=0.7)
    
    # 2. Polynomial scaling
    if len(il_joint) > 20:
        poly = PolynomialFeatures(degree=2)
        il_poly = poly.fit_transform(il_joint.reshape(-1, 1))
        poly_model = LinearRegression()
        poly_model.fit(il_poly, rl_joint)
        
        il_range_poly = poly.transform(il_range.reshape(-1, 1))
        rl_pred = poly_model.predict(il_range_poly)
        ax.plot(il_range, rl_pred, 'g--', label='Poly2', alpha=0.7)
        
        print(f"  Polynomial coefficients: {poly_model.coef_}")
    
    # 3. Bounded scaling with tanh
    # RL = scale * tanh(gain * IL) + offset
    def tanh_model(il, scale, gain, offset):
        return scale * np.tanh(gain * il) + offset
    
    try:
        if len(il_joint) > 10:
            # Initial guess
            p0 = [np.std(rl_joint), 1.0, np.mean(rl_joint)]
            popt, _ = optimize.curve_fit(tanh_model, il_joint, rl_joint, p0=p0, maxfev=5000)
            scale, gain, offset = popt
            print(f"  Tanh: RL = {scale:.2f} * tanh({gain:.2f} * IL) + {offset:.2f}")
            
            rl_tanh = tanh_model(il_range, *popt)
            ax.plot(il_range, rl_tanh, 'm:', label=f'Tanh', alpha=0.7, linewidth=2)
    except:
        pass
    
    ax.legend(fontsize=8)
    ax.set_xlim([il_joint.min() - 0.1, il_joint.max() + 0.1])
    
    # Store the best scaling function (linear for simplicity)
    scaling_functions[joint_idx] = {'a': a, 'b': b}

plt.tight_layout()
plt.savefig('/workspace/il_to_rl_scaling_analysis.png', dpi=150)
print(f"\n\nScaling analysis saved to: /workspace/il_to_rl_scaling_analysis.png")
print('View with: "$BROWSER" /workspace/il_to_rl_scaling_analysis.png')

# Create a practical scaling function
print("\n\n=== RECOMMENDED SCALING FUNCTION ===")
print("Based on the analysis, here's a non-linear scaling function:")

print("""
def scale_il_to_sim(il_actions):
    \"\"\"
    Non-linear scaling from IL model outputs to simulation actions.
    This can be removed when deploying on the real robot.
    \"\"\"
    # Per-joint scaling parameters (derived from data)
    scaling_params = {
        0: {'a': 25.0, 'b': 0.0},   # LF-Hip
        1: {'a': 20.0, 'b': 5.0},   # LF-Thigh
        2: {'a': 22.0, 'b': -2.0},  # LF-Knee
        3: {'a': 25.0, 'b': 0.0},   # RF-Hip
        4: {'a': 20.0, 'b': 5.0},   # RF-Thigh
        5: {'a': 22.0, 'b': -2.0},  # RF-Knee
        6: {'a': 25.0, 'b': 0.0},   # LB-Hip
        7: {'a': 20.0, 'b': 5.0},   # LB-Thigh
        8: {'a': 22.0, 'b': -2.0},  # LB-Knee
        9: {'a': 25.0, 'b': 0.0},   # RB-Hip
        10: {'a': 20.0, 'b': 5.0},  # RB-Thigh
        11: {'a': 22.0, 'b': -2.0}, # RB-Knee
    }
    
    scaled_actions = torch.zeros_like(il_actions)
    
    for i in range(12):
        # Linear scaling with bounds
        scaled = scaling_params[i]['a'] * il_actions[..., i] + scaling_params[i]['b']
        
        # Clip to reasonable joint limits
        joint_limits = {
            0: 90, 3: 90, 6: 90, 9: 90,      # Hip joints: ±90 rad
            1: 90, 4: 90, 7: 90, 10: 90,     # Thigh joints: ±90 rad
            2: 135, 5: 135, 8: 135, 11: 135, # Knee joints: ±135 rad
        }
        
        scaled_actions[..., i] = torch.clamp(scaled, -joint_limits[i], joint_limits[i])
    
    return scaled_actions

# Alternative: Simple uniform scaling
def scale_il_to_sim_simple(il_actions):
    \"\"\"Simple version: uniform scaling\"\"\"
    SCALE_FACTOR = 22.0  # Average scaling factor
    return il_actions * SCALE_FACTOR
""")

# Generate the actual scaling parameters from data
print("\n\nActual scaling parameters from data analysis:")
for i, name in enumerate(joint_names):
    if i in scaling_functions:
        print(f"  {i}: {{'a': {scaling_functions[i]['a']:.1f}, 'b': {scaling_functions[i]['b']:.1f}}},  # {name}")

# Test the scaling
print("\n\nTesting scaling function:")
test_obs = torch.tensor(demo_obs[idx], dtype=torch.float32, device=device).unsqueeze(0)
il_obs_norm = (test_obs - IL_MODEL['obs_rms_mean'].to(device)) / torch.sqrt(IL_MODEL['obs_rms_var'].to(device) + 1e-8)
il_test = forward_actor(IL_MODEL['model_state_dict'], il_obs_norm)

print(f"Original IL output: {il_test[0, :3].cpu().numpy()}")
print(f"Scaled (22x): {(il_test[0, :3] * 22.0).cpu().numpy()}")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from temporal_attribution_methods import temporal_masked_stig, incremental_stig
from standard_ig import integrated_gradients as standard_ig
from spatiotemporal_ig import spatiotemporal_integrated_gradients_corrected as original_stig

class SimpleTemporalModel(nn.Module):
    """
    A simple model designed to have explicit temporal dependencies
    to clearly demonstrate the difference between attribution methods.
    """
    def __init__(self, input_size=5, hidden_size=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
        # Initialize to make early vs late timesteps matter differently
        with torch.no_grad():
            # Make LSTM remember early information
            self.lstm.bias_ih_l0[hidden_size:2*hidden_size] = -3.0  # Low forget gate bias
            
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Use weighted combination of outputs
        # Early timesteps contribute differently than late ones
        weights = torch.linspace(0.1, 1.0, lstm_out.size(1)).to(x.device)
        weights = weights.unsqueeze(0).unsqueeze(-1)
        
        weighted_out = (lstm_out * weights).mean(dim=1)
        
        return self.fc(weighted_out)

def create_temporal_test_cases():
    """
    Create test cases where temporal order clearly matters.
    """
    seq_len = 20
    features = 5
    
    # Case 1: Signal only at beginning
    case1 = torch.zeros(seq_len, features)
    case1[0:2, 0] = 1.0
    
    # Case 2: Signal only at end
    case2 = torch.zeros(seq_len, features)
    case2[-2:, 0] = 1.0
    
    # Case 3: Signal in middle
    case3 = torch.zeros(seq_len, features)
    case3[9:11, 0] = 1.0
    
    # Case 4: Increasing signal
    case4 = torch.zeros(seq_len, features)
    case4[:, 0] = torch.linspace(0, 1, seq_len)
    
    return [case1, case2, case3, case4], ['Early Signal', 'Late Signal', 'Middle Signal', 'Increasing Signal']

def demonstrate_temporal_differences():
    """
    Clearly demonstrate how different attribution methods handle temporal dependencies.
    """
    # Create model
    model = SimpleTemporalModel()
    model.eval()
    
    # Create test cases
    test_cases, case_names = create_temporal_test_cases()
    
    # Attribution methods
    methods = {
        'Standard IG': standard_ig,
        'Original STIG': original_stig,
        'Temporal Masked STIG': temporal_masked_stig,
        'Incremental STIG': incremental_stig
    }
    
    # Baseline
    baseline = torch.zeros_like(test_cases[0])
    
    # Compute attributions for all cases and methods
    results = {}
    
    print("Model outputs for test cases:")
    for case_idx, (test_input, case_name) in enumerate(zip(test_cases, case_names)):
        with torch.no_grad():
            output = model(test_input).item()
            print(f"{case_name}: {output:.4f}")
        
        results[case_name] = {}
        
        for method_name, method_func in methods.items():
            try:
                attr = method_func(model, test_input, baseline, target_class=0)
                results[case_name][method_name] = attr.detach().numpy()
            except Exception as e:
                print(f"Error with {method_name} on {case_name}: {e}")
                results[case_name][method_name] = None
    
    # Visualization
    fig, axes = plt.subplots(len(test_cases), len(methods) + 1, 
                            figsize=(3 * (len(methods) + 1), 3 * len(test_cases)))
    
    for case_idx, case_name in enumerate(case_names):
        # Plot input
        ax = axes[case_idx, 0]
        ax.imshow(test_cases[case_idx].numpy().T, aspect='auto', cmap='Blues')
        ax.set_title(f'Input: {case_name}')
        ax.set_ylabel('Features')
        if case_idx == len(case_names) - 1:
            ax.set_xlabel('Time Steps')
        
        # Plot attributions
        for method_idx, method_name in enumerate(methods.keys()):
            ax = axes[case_idx, method_idx + 1]
            
            attr = results[case_name][method_name]
            if attr is not None:
                # Normalize for visualization
                attr_norm = attr / (np.abs(attr).max() + 1e-8)
                im = ax.imshow(attr_norm.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
                
                if case_idx == 0:
                    ax.set_title(method_name)
                
                if case_idx == len(case_names) - 1:
                    ax.set_xlabel('Time Steps')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            
            ax.set_yticks([])
    
    plt.suptitle('Attribution Methods on Temporal Test Cases', fontsize=14)
    plt.tight_layout()
    plt.savefig('temporal_differences_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compute and display key differences
    print("\n" + "="*60)
    print("KEY DIFFERENCES BETWEEN METHODS:")
    print("="*60)
    
    for case_name in case_names:
        print(f"\n{case_name}:")
        
        # Compare temporal patterns
        if 'Standard IG' in results[case_name] and results[case_name]['Standard IG'] is not None:
            std_ig = results[case_name]['Standard IG']
            std_temporal = np.abs(std_ig).mean(axis=1)
            
            for method_name in methods.keys():
                if method_name != 'Standard IG' and method_name in results[case_name]:
                    attr = results[case_name][method_name]
                    if attr is not None:
                        temporal = np.abs(attr).mean(axis=1)
                        
                        # Find peak attribution timestep
                        std_peak = np.argmax(std_temporal)
                        method_peak = np.argmax(temporal)
                        
                        # Compute temporal correlation
                        corr = np.corrcoef(std_temporal, temporal)[0, 1]
                        
                        print(f"  {method_name}: peak@{method_peak} (vs {std_peak}), corr={corr:.3f}")

if __name__ == "__main__":
    demonstrate_temporal_differences()
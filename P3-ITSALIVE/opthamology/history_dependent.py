import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def create_temporal_dependency_test():
    """
    Create a synthetic model and data where temporal order matters
    """
    class TemporalOrderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(5, 10, batch_first=True)
            self.fc = nn.Linear(10, 1)
            
            # Initialize to make the model sensitive to temporal order
            with torch.no_grad():
                # Make LSTM forget gate low (remember everything)
                self.lstm.bias_ih_l0[10:20] = -2.0  # forget gate bias
                # Make output depend strongly on final hidden state
                self.fc.weight.fill_(0.0)
                self.fc.weight[0, 0] = 1.0
        
        def forward(self, x):
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use only last hidden state
            return self.fc(h_n.squeeze(0))
    
    model = TemporalOrderModel()
    model.eval()
    
    # Test case 1: Signal at beginning
    input1 = torch.zeros(1, 10, 5)
    input1[0, 0, 0] = 1.0
    
    # Test case 2: Signal at end
    input2 = torch.zeros(1, 10, 5)
    input2[0, 9, 0] = 1.0
    
    # Test case 3: Signal throughout
    input3 = torch.zeros(1, 10, 5)
    input3[0, :, 0] = 0.1
    
    return model, [input1, input2, input3]

def compare_stig_variants():
    """
    Compare different STIG implementations
    """
    model, test_inputs = create_temporal_dependency_test()
    
    # Import all methods
    from spatiotemporal_ig import spatiotemporal_integrated_gradients_corrected as original_stig
    from standard_ig import integrated_gradients as standard_ig
    from modified_stig import modified_stig_with_temporal_masking
    from history_aware_stig import history_aware_stig
    from nonlinear_temporal_stig import nonlinear_temporal_stig
    
    methods = {
        'Standard IG': standard_ig,
        'Original STIG': original_stig,
        'Modified STIG (Temporal Masking)': modified_stig_with_temporal_masking,
        'History-Aware STIG': history_aware_stig,
        'Non-linear Temporal STIG': nonlinear_temporal_stig
    }
    
    # Compute attributions for each method and input
    results = {}
    
    for method_name, method_func in methods.items():
        results[method_name] = []
        for inp in test_inputs:
            attr = method_func(model, inp)
            results[method_name].append(attr)
    
    # Plot results
    fig, axes = plt.subplots(len(methods), 3, figsize=(15, len(methods) * 3))
    
    for i, (method_name, attrs) in enumerate(results.items()):
        for j, attr in enumerate(attrs):
            im = axes[i, j].imshow(attr.squeeze().T, aspect='auto', cmap='RdBu', 
                                   vmin=-0.5, vmax=0.5)
            axes[i, j].set_title(f'{method_name}\nTest Case {j+1}')
            
            if i == 0:
                axes[i, j].set_title(f'Test {j+1}: ' + 
                                    ['Early Signal', 'Late Signal', 'Uniform Signal'][j] + 
                                    f'\n{method_name}')
            
            if j == 0:
                axes[i, j].set_ylabel('Features')
            
            if i == len(methods) - 1:
                axes[i, j].set_xlabel('Time Steps')
            
            plt.colorbar(im, ax=axes[i, j])
    
    plt.tight_layout()
    plt.savefig('stig_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compute numerical difference between methods
    print("\nNumerical differences from Standard IG:")
    for method_name in methods:
        if method_name != 'Standard IG':
            total_diff = 0
            for i in range(len(test_inputs)):
                diff = torch.norm(results[method_name][i] - results['Standard IG'][i])
                total_diff += diff.item()
            print(f"{method_name}: {total_diff:.6f}")

if __name__ == "__main__":
    compare_stig_variants()
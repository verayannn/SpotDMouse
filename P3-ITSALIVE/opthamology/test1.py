import torch
import matplotlib.pyplot as plt
import numpy as np
from simple_model import generate_stock_data, LN_Lorenz, RNN_Lorenz, LSTM_Lorenz, train_model
from standard_ig import integrated_gradients as standard_ig
from spatiotemporal_ig import spatiotemporal_integrated_gradients_corrected as original_stig
from temporal_attribution_methods import temporal_masked_stig, incremental_stig, attention_weighted_stig

def test_all_attribution_methods():
    """
    Test all attribution methods on your trained models with stock data.
    """
    # Generate stock data
    print("Generating stock data...")
    X_train, y_train, X_test, y_test = generate_stock_data(
        ticker="AAPL",
        seq_len=50,
        pred_horizon=20,
    )
    
    # Initialize and train models
    models = {
        'Feedforward': LN_Lorenz(seq_len=50, input_dim=5, hidden_dim=64),
        'GRU': RNN_Lorenz(input_size=5, hidden_size=64),
        'LSTM': LSTM_Lorenz(input_size=5, hidden_size=64)
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        train_losses, test_losses = train_model(model, X_train, y_train, X_test, y_test, epochs=50)
        trained_models[name] = model
    
    # Select test sample
    test_idx = 42
    test_input = X_test[test_idx:test_idx+1].to(device)
    true_value = y_test[test_idx].item()
    
    # Define attribution methods
    attribution_methods = {
        'Standard IG': lambda m, x, b: standard_ig(m, x, b, target_class=0),
        'Original STIG': lambda m, x, b: original_stig(m, x, b, target_class=0),
        'Temporal Masked STIG': lambda m, x, b: temporal_masked_stig(m, x, b, target_class=0),
        'Incremental STIG': lambda m, x, b: incremental_stig(m, x, b, target_class=0),
        'Attention Weighted STIG': lambda m, x, b: attention_weighted_stig(m, x, b, target_class=0)
    }
    
    # Test with different baselines
    baselines = {
        'zeros': torch.zeros_like(test_input),
        'mean': torch.ones_like(test_input) * test_input.mean()
    }
    
    # Store all results
    all_results = {}
    
    for baseline_name, baseline in baselines.items():
        print(f"\n{'='*60}")
        print(f"Testing with {baseline_name} baseline")
        print(f"{'='*60}")
        
        results = {}
        
        for model_name, model in trained_models.items():
            print(f"\n{model_name} Model:")
            model.eval()
            
            # Get predictions
            with torch.no_grad():
                pred_input = model(test_input).item()
                pred_baseline = model(baseline).item()
                print(f"Prediction (input): {pred_input:.4f}")
                print(f"Prediction (baseline): {pred_baseline:.4f}")
                print(f"True value: {true_value:.4f}")
            
            # Compute attributions with each method
            model_results = {}
            
            for method_name, method_func in attribution_methods.items():
                try:
                    print(f"\n  Computing {method_name}...", end='')
                    attr = method_func(model, test_input, baseline)
                    
                    # Ensure consistent shape
                    if attr.dim() == 3:
                        attr = attr.squeeze(0)
                    
                    model_results[method_name] = attr.cpu().detach().numpy()
                    print(" Done!")
                    
                except Exception as e:
                    print(f" Failed: {e}")
                    model_results[method_name] = None
            
            results[model_name] = model_results
        
        all_results[baseline_name] = results
    
    # Visualize results
    visualize_attribution_comparison(all_results, test_input.cpu().numpy())
    
    # Compute similarity metrics
    compute_method_similarities(all_results)
    
    return all_results

def visualize_attribution_comparison(all_results, test_input):
    """
    Create comprehensive visualization of all attribution methods.
    """
    # Get list of methods and models
    baseline_names = list(all_results.keys())
    model_names = list(all_results[baseline_names[0]].keys())
    method_names = list(all_results[baseline_names[0]][model_names[0]].keys())
    
    # Create large comparison figure
    fig = plt.figure(figsize=(20, 15))
    
    # Feature names for stock data
    feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for baseline_idx, baseline_name in enumerate(baseline_names):
        for model_idx, model_name in enumerate(model_names):
            for method_idx, method_name in enumerate(method_names):
                
                subplot_idx = (baseline_idx * len(model_names) * len(method_names) + 
                              model_idx * len(method_names) + method_idx + 1)
                
                ax = plt.subplot(len(baseline_names) * len(model_names), len(method_names), subplot_idx)
                
                attr = all_results[baseline_name][model_name][method_name]
                
                if attr is not None:
                    # Normalize attribution for visualization
                    attr_norm = attr / (np.abs(attr).max() + 1e-8)
                    
                    im = ax.imshow(attr_norm.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
                    
                    if baseline_idx == 0 and model_idx == 0:
                        ax.set_title(method_name, fontsize=10)
                    
                    if method_idx == 0:
                        ax.set_ylabel(f'{model_name}\n{baseline_name}', fontsize=8)
                    
                    if model_idx == len(model_names) - 1 and baseline_idx == len(baseline_names) - 1:
                        ax.set_xlabel('Time Steps', fontsize=8)
                    
                    # Set y-tick labels to feature names
                    if method_idx == 0:
                        ax.set_yticks(range(len(feature_names)))
                        ax.set_yticklabels(feature_names, fontsize=6)
                    else:
                        ax.set_yticks([])
                    
                    # Add colorbar for last column
                    if method_idx == len(method_names) - 1:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'Failed', ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
    
    plt.suptitle('Attribution Method Comparison Across Models and Baselines', fontsize=16)
    plt.tight_layout()
    plt.savefig('comprehensive_attribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create temporal pattern comparison
    fig, axes = plt.subplots(len(model_names), len(baseline_names), 
                            figsize=(12, 8), sharex=True, sharey=True)
    
    if len(baseline_names) == 1:
        axes = axes.reshape(-1, 1)
    
    for model_idx, model_name in enumerate(model_names):
        for baseline_idx, baseline_name in enumerate(baseline_names):
            ax = axes[model_idx, baseline_idx]
            
            # Plot temporal patterns for each method
            for method_name in method_names:
                attr = all_results[baseline_name][model_name][method_name]
                if attr is not None:
                    # Average absolute attribution across features
                    temporal_pattern = np.abs(attr).mean(axis=1)
                    ax.plot(temporal_pattern, label=method_name, alpha=0.7)
            
            ax.set_title(f'{model_name} - {baseline_name}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Avg Attribution Magnitude')
            
            if model_idx == 0 and baseline_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Temporal Attribution Patterns', fontsize=14)
    plt.tight_layout()
    plt.savefig('temporal_patterns_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_method_similarities(all_results):
    """
    Compute and display similarity metrics between different attribution methods.
    """
    print("\n" + "="*80)
    print("SIMILARITY ANALYSIS BETWEEN ATTRIBUTION METHODS")
    print("="*80)
    
    for baseline_name, baseline_results in all_results.items():
        print(f"\nBaseline: {baseline_name}")
        print("-" * 60)
        
        for model_name, model_results in baseline_results.items():
            print(f"\n  Model: {model_name}")
            
            # Get valid methods
            valid_methods = {k: v for k, v in model_results.items() if v is not None}
            method_names = list(valid_methods.keys())
            
            if len(valid_methods) < 2:
                print("    Not enough valid methods for comparison")
                continue
            
            # Compute pairwise correlations
            print("\n    Pairwise Correlations:")
            for i, method1 in enumerate(method_names):
                for j, method2 in enumerate(method_names):
                    if i < j:
                        attr1 = valid_methods[method1].flatten()
                        attr2 = valid_methods[method2].flatten()
                        
                        corr = np.corrcoef(attr1, attr2)[0, 1]
                        print(f"      {method1} vs {method2}: {corr:.4f}")
            
            # Check if temporal patterns differ
            print("\n    Temporal Pattern Differences:")
            temporal_patterns = {}
            
            for method_name, attr in valid_methods.items():
                temporal_patterns[method_name] = np.abs(attr).mean(axis=1)
            
            # Compare standard IG with each temporal method
            if 'Standard IG' in temporal_patterns:
                std_pattern = temporal_patterns['Standard IG']
                
                for method_name, pattern in temporal_patterns.items():
                    if method_name != 'Standard IG':
                        # Compute temporal difference metric
                        temporal_diff = np.mean(np.abs(pattern - std_pattern))
                        temporal_corr = np.corrcoef(pattern, std_pattern)[0, 1]
                        
                        print(f"      {method_name}: diff={temporal_diff:.4f}, corr={temporal_corr:.4f}")

if __name__ == "__main__":
    # Run the comprehensive test
    results = test_all_attribution_methods()
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("1. Temporal Masked STIG should show different patterns for RNNs/LSTMs")
    print("2. Incremental STIG captures marginal contributions at each timestep")
    print("3. Attention Weighted STIG emphasizes important temporal regions")
    print("4. These differences should be more pronounced in sequential models (RNN/LSTM)")
    print("5. Feedforward models may show similar results across methods")
    print("="*80)
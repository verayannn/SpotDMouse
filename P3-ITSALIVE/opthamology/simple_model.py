# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from spatiotemporal_ig import spatiotemporal_integrated_gradients as stig
# from standard_ig import integrated_gradients as ig

# class LN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.nonlinear = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.nonlinear(x)
#         x = self.linear2(x)
#         return x
# # Create input tensor WITHOUT requires_grad first
# input_tensor = torch.zeros(8, 10)  # Extended from 4 to 8 time steps
# input_tensor[1:4, :3] = 1.0  # Extended square wave: off, on, on, on, off, off, off, off
# input_tensor[2:6, 6:] = 0.8  # Extended square wave: off, off, on, on, on, on, off, off

# baseline = torch.zeros_like(input_tensor)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = LN(input_dim=10, hidden_dim=5, output_dim=1)
# model.eval()
# model.to(device)

# input_tensor = input_tensor.to(device)
# baseline = baseline.to(device)

# print("Input shape:", input_tensor.shape)
# print("Input tensor:")
# # print(input_tensor)

# # Forward pass
# with torch.no_grad():
#     output = model(input_tensor)
#     print("Model output shape:", output.shape)
#     # print("Model output:", output)

# # 1. Standard Gradients - NOW set requires_grad=True on a clean copy
# input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
# output_grad = model(input_tensor_grad)
# loss = output_grad.sum()  # Or use specific target
# loss.backward()
# gradients = input_tensor_grad.grad
# print("Gradients shape:", gradients.shape)

# # 2. Integrated Gradients
# try:
#     stanig = ig(model, input_tensor.detach(), baseline=baseline.detach(), target_class=0)
#     print("IG shape:", stanig.shape)
#     # print("IG values:\n", stanig)
# except Exception as e:
#     print(f"IG error: {e}")

# # 3. Spatiotemporal Integrated Gradients
# try:
#     # Assuming STIG expects temporal dimension
#     stig_result = stig(model, input_tensor.detach(), baseline=baseline.detach(), target_class=0)
#     print("STIG shape:", stig_result.shape)
#     # print("STIG values:\n", stig_result)
# except Exception as e:
#     print(f"STIG error: {e}")

# # Visualization
# def plot_results():
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     # Input - TRANSPOSED so time is on x-axis
#     im1 = axes[0, 0].imshow(input_tensor.cpu().detach().numpy().T, aspect='auto', cmap='viridis')
#     axes[0, 0].set_title('Input Tensor')
#     axes[0, 0].set_xlabel('Time Steps')
#     axes[0, 0].set_ylabel('Features')
#     plt.colorbar(im1, ax=axes[0, 0])
    
#     # Gradients - TRANSPOSED so time is on x-axis
#     im2 = axes[0, 1].imshow(gradients.cpu().detach().numpy().T, aspect='auto', cmap='RdBu')
#     axes[0, 1].set_title('Standard Gradients')
#     axes[0, 1].set_xlabel('Time Steps')
#     axes[0, 1].set_ylabel('Features')
#     plt.colorbar(im2, ax=axes[0, 1])
    
#     # IG (if available) - TRANSPOSED so time is on x-axis
#     try:
#         im3 = axes[0, 2].imshow(stanig.cpu().detach().numpy().T, aspect='auto', cmap='RdBu')
#         axes[0, 2].set_title('Integrated Gradients')
#         axes[0, 2].set_xlabel('Time Steps')
#         axes[0, 2].set_ylabel('Features')
#         plt.colorbar(im3, ax=axes[0, 2])
#     except:
#         axes[0, 2].text(0.5, 0.5, 'IG not available', ha='center', va='center')
#         axes[0, 2].set_title('Integrated Gradients')
#         axes[0, 2].set_xlabel('Time Steps')
#         axes[0, 2].set_ylabel('Features')
    
#     # Feature importance over time
#     axes[1, 0].plot(gradients.cpu().detach().numpy().mean(axis=1), 'o-', label='Avg Gradient')
#     axes[1, 0].set_title('Average Gradient per Time Step')
#     axes[1, 0].set_xlabel('Time Steps')
#     axes[1, 0].set_ylabel('Average Gradient')
#     axes[1, 0].legend()
    
#     # Feature importance across features
#     axes[1, 1].plot(gradients.cpu().detach().numpy().mean(axis=0), 'o-', label='Avg Gradient')
#     axes[1, 1].set_title('Average Gradient per Feature')
#     axes[1, 1].set_xlabel('Features')
#     axes[1, 1].set_ylabel('Average Gradient')
#     axes[1, 1].legend()
    
#     # STIG (if available) - TRANSPOSED so time is on x-axis
#     try:
#         im6 = axes[1, 2].imshow(stig_result.cpu().detach().numpy().T, aspect='auto', cmap='RdBu')
#         axes[1, 2].set_title('Spatiotemporal IG')
#         axes[1, 2].set_xlabel('Time Steps')
#         axes[1, 2].set_ylabel('Features')
#         plt.colorbar(im6, ax=axes[1, 2])
#     except:
#         axes[1, 2].text(0.5, 0.5, 'STIG not available', ha='center', va='center')
#         axes[1, 2].set_title('Spatiotemporal IG')
#         axes[1, 2].set_xlabel('Time Steps')
#         axes[1, 2].set_ylabel('Features')
    
#     plt.tight_layout()
#     plt.savefig('gradient_analysis.png', dpi=300, bbox_inches='tight')
#     plt.show()

# plot_results()

# # Quantitative Comparison between STIG and IG
# def compare_attribution_methods():
#     print("\n=== Attribution Method Comparison ===")
    
#     # Check if both methods worked
#     try:
#         stanig_available = 'stanig' in locals() or 'stanig' in globals()
#         stig_available = 'stig_result' in locals() or 'stig_result' in globals()
        
#         if not (stanig_available and stig_available):
#             print("Cannot compare: One or both attribution methods failed")
#             return
        
#         # Convert to numpy for easier computation
#         ig_attr = stanig.cpu().detach().numpy()
#         stig_attr = stig_result.cpu().detach().numpy()
#         grad_attr = gradients.cpu().detach().numpy()
        
#         print(f"IG attribution shape: {ig_attr.shape}")
#         print(f"STIG attribution shape: {stig_attr.shape}")
#         print(f"Gradient attribution shape: {grad_attr.shape}")
        
#         # Ensure shapes match
#         if ig_attr.shape != stig_attr.shape:
#             print(f"Shape mismatch: IG {ig_attr.shape} vs STIG {stig_attr.shape}")
#             return
        
#         # 1. Correlation Analysis
#         ig_flat = ig_attr.flatten()
#         stig_flat = stig_attr.flatten()
#         grad_flat = grad_attr.flatten()
        
#         correlation_ig_stig = np.corrcoef(ig_flat, stig_flat)[0, 1]
#         correlation_ig_grad = np.corrcoef(ig_flat, grad_flat)[0, 1]
#         correlation_stig_grad = np.corrcoef(stig_flat, grad_flat)[0, 1]
        
#         print(f"\nCorrelation Analysis:")
#         print(f"IG vs STIG: {correlation_ig_stig:.4f}")
#         print(f"IG vs Gradient: {correlation_ig_grad:.4f}")
#         print(f"STIG vs Gradient: {correlation_stig_grad:.4f}")
        
#         # 2. Mean Squared Error
#         mse_ig_stig = np.mean((ig_attr - stig_attr) ** 2)
#         mse_ig_grad = np.mean((ig_attr - grad_attr) ** 2)
#         mse_stig_grad = np.mean((stig_attr - grad_attr) ** 2)
        
#         print(f"\nMean Squared Error:")
#         print(f"IG vs STIG: {mse_ig_stig:.6f}")
#         print(f"IG vs Gradient: {mse_ig_grad:.6f}")
#         print(f"STIG vs Gradient: {mse_stig_grad:.6f}")
        
#         # 3. Cosine Similarity
#         def cosine_similarity(a, b):
#             return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
#         cos_sim_ig_stig = cosine_similarity(ig_flat, stig_flat)
#         cos_sim_ig_grad = cosine_similarity(ig_flat, grad_flat)
#         cos_sim_stig_grad = cosine_similarity(stig_flat, grad_flat)
        
#         print(f"\nCosine Similarity:")
#         print(f"IG vs STIG: {cos_sim_ig_stig:.4f}")
#         print(f"IG vs Gradient: {cos_sim_ig_grad:.4f}")
#         print(f"STIG vs Gradient: {cos_sim_stig_grad:.4f}")
        
#         # 4. Statistical Tests
#         from scipy.stats import spearmanr, kendalltau
        
#         spearman_ig_stig, p_spearman = spearmanr(ig_flat, stig_flat)
#         kendall_ig_stig, p_kendall = kendalltau(ig_flat, stig_flat)
        
#         print(f"\nRank Correlation (IG vs STIG):")
#         print(f"Spearman's ρ: {spearman_ig_stig:.4f} (p={p_spearman:.4f})")
#         print(f"Kendall's τ: {kendall_ig_stig:.4f} (p={p_kendall:.4f})")
        
#         # 5. Top-k Feature Agreement
#         def top_k_agreement(attr1, attr2, k=5):
#             # Get top k indices for each attribution
#             top_k_1 = np.unravel_index(np.argpartition(attr1.flatten(), -k)[-k:], attr1.shape)
#             top_k_2 = np.unravel_index(np.argpartition(attr2.flatten(), -k)[-k:], attr2.shape)
            
#             # Convert to set of coordinate tuples
#             set1 = set(zip(top_k_1[0], top_k_1[1]))
#             set2 = set(zip(top_k_2[0], top_k_2[1]))
            
#             # Calculate Jaccard similarity
#             intersection = len(set1.intersection(set2))
#             union = len(set1.union(set2))
            
#             return intersection / union if union > 0 else 0
        
#         for k in [3, 5, 10]:
#             agreement = top_k_agreement(np.abs(ig_attr), np.abs(stig_attr), k)
#             print(f"Top-{k} feature agreement (IG vs STIG): {agreement:.4f}")
        
#         # 6. Temporal Pattern Analysis
#         print(f"\nTemporal Pattern Analysis:")
#         ig_temporal = np.abs(ig_attr).mean(axis=1)  # Average over features
#         stig_temporal = np.abs(stig_attr).mean(axis=1)
#         grad_temporal = np.abs(grad_attr).mean(axis=1)
        
#         temporal_corr_ig_stig = np.corrcoef(ig_temporal, stig_temporal)[0, 1]
#         temporal_corr_ig_grad = np.corrcoef(ig_temporal, grad_temporal)[0, 1]
#         temporal_corr_stig_grad = np.corrcoef(stig_temporal, grad_temporal)[0, 1]
        
#         print(f"Temporal correlation IG vs STIG: {temporal_corr_ig_stig:.4f}")
#         print(f"Temporal correlation IG vs Gradient: {temporal_corr_ig_grad:.4f}")
#         print(f"Temporal correlation STIG vs Gradient: {temporal_corr_stig_grad:.4f}")
        
#         # 7. Feature Pattern Analysis
#         print(f"\nFeature Pattern Analysis:")
#         ig_feature = np.abs(ig_attr).mean(axis=0)  # Average over time
#         stig_feature = np.abs(stig_attr).mean(axis=0)
#         grad_feature = np.abs(grad_attr).mean(axis=0)
        
#         feature_corr_ig_stig = np.corrcoef(ig_feature, stig_feature)[0, 1]
#         feature_corr_ig_grad = np.corrcoef(ig_feature, grad_feature)[0, 1]
#         feature_corr_stig_grad = np.corrcoef(stig_feature, grad_feature)[0, 1]
        
#         print(f"Feature correlation IG vs STIG: {feature_corr_ig_stig:.4f}")
#         print(f"Feature correlation IG vs Gradient: {feature_corr_ig_grad:.4f}")
#         print(f"Feature correlation STIG vs Gradient: {feature_corr_stig_grad:.4f}")
        
#         # 8. Create comparison visualization
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
#         # Scatter plot IG vs STIG
#         axes[0, 0].scatter(ig_flat, stig_flat, alpha=0.6)
#         axes[0, 0].plot([ig_flat.min(), ig_flat.max()], [ig_flat.min(), ig_flat.max()], 'r--', alpha=0.8)
#         axes[0, 0].set_xlabel('IG Attribution')
#         axes[0, 0].set_ylabel('STIG Attribution')
#         axes[0, 0].set_title(f'IG vs STIG (r={correlation_ig_stig:.3f})')
        
#         # Difference heatmap
#         diff_map = ig_attr - stig_attr
#         im = axes[0, 1].imshow(diff_map.T, aspect='auto', cmap='RdBu')
#         axes[0, 1].set_title('Difference (IG - STIG)')
#         axes[0, 1].set_xlabel('Time Steps')
#         axes[0, 1].set_ylabel('Features')
#         plt.colorbar(im, ax=axes[0, 1])
        
#         # Temporal patterns
#         axes[1, 0].plot(ig_temporal, 'o-', label='IG', alpha=0.7)
#         axes[1, 0].plot(stig_temporal, 's-', label='STIG', alpha=0.7)
#         axes[1, 0].plot(grad_temporal, '^-', label='Gradient', alpha=0.7)
#         axes[1, 0].set_xlabel('Time Steps')
#         axes[1, 0].set_ylabel('Average Attribution Magnitude')
#         axes[1, 0].set_title('Temporal Attribution Patterns')
#         axes[1, 0].legend()
        
#         # Feature patterns
#         axes[1, 1].plot(ig_feature, 'o-', label='IG', alpha=0.7)
#         axes[1, 1].plot(stig_feature, 's-', label='STIG', alpha=0.7)
#         axes[1, 1].plot(grad_feature, '^-', label='Gradient', alpha=0.7)
#         axes[1, 1].set_xlabel('Features')
#         axes[1, 1].set_ylabel('Average Attribution Magnitude')
#         axes[1, 1].set_title('Feature Attribution Patterns')
#         axes[1, 1].legend()
        
#         plt.tight_layout()
#         plt.savefig('attribution_comparison.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#     except Exception as e:
#         print(f"Comparison failed: {e}")
#         import traceback
#         traceback.print_exc()

# # Run the comparison
# compare_attribution_methods()

# # Analysis
# print("\n=== Analysis ===")
# print(f"Max gradient magnitude: {gradients.abs().max().item():.4f}")
# print(f"Mean gradient: {gradients.mean().item():.4f}")
# print(f"Gradient std: {gradients.std().item():.4f}")

# # Check which time steps and features are most important
# time_importance = gradients.abs().mean(dim=1)
# feature_importance = gradients.abs().mean(dim=0)

# print(f"Most important time step: {time_importance.argmax().item()} (value: {time_importance.max().item():.4f})")
# print(f"Most important feature: {feature_importance.argmax().item()} (value: {feature_importance.max().item():.4f})")

# # Analysis
# print("\n=== Analysis ===")
# print(f"Max gradient magnitude: {gradients.abs().max().item():.4f}")
# print(f"Mean gradient: {gradients.mean().item():.4f}")
# print(f"Gradient std: {gradients.std().item():.4f}")

# # Check which time steps and features are most important
# time_importance = gradients.abs().mean(dim=1)
# feature_importance = gradients.abs().mean(dim=0)

# print(f"Most important time step: {time_importance.argmax().item()} (value: {time_importance.max().item():.4f})")
# print(f"Most important feature: {feature_importance.argmax().item()} (value: {feature_importance.max().item():.4f})")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from spatiotemporal_ig import spatiotemporal_integrated_gradients_corrected as stig
from standard_ig import integrated_gradients as ig
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def generate_stock_data(ticker="SPY", start_date="2010-01-01", end_date="2024-01-01", 
                        seq_len=50, pred_horizon=10, train_ratio=0.8):
    """
    Downloads stock data and transforms it into sequences for time series prediction.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'SPY' for S&P 500 ETF).
        start_date (str): Start date for data download.
        end_date (str): End date for data download.
        seq_len (int): Use past N steps as input sequence.
        pred_horizon (int): Predict the Close price M steps ahead.
        train_ratio (float): Fraction of data to use for training.
        
    Returns:
        X_train, y_train, X_test, y_test (torch.Tensors)
    """
    
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # 1. Download Data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Use features: Open, High, Low, Close, Volume
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = data[features].copy()
    
    # 2. Preprocessing (Scaling)
    # Important: Normalize features (e.g., prices) using the whole dataset
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)
    
    # 3. Create Sequences
    X_sequences = []
    y_sequences = [] # Target is the 'Close' price (index 3) pred_horizon steps ahead
    
    # We need enough data for the sequence (seq_len) and the prediction (pred_horizon)
    for i in range(seq_len, len(data_scaled) - pred_horizon):
        # Input X: sequence of scaled features
        X_seq = data_scaled[i-seq_len:i, :]
        
        # Output y: scaled Close price, pred_horizon steps ahead
        # Close price is the 4th column (index 3)
        y_seq = data_scaled[i + pred_horizon, 3] 
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)
    
    print(f"Total sequences created: {len(X_sequences)}")
    print(f"X sequence shape: {X_sequences.shape}")
    
    # 4. Split Train/Test
    n_train = int(len(X_sequences) * train_ratio)
    
    X_train = torch.from_numpy(X_sequences[:n_train])
    y_train = torch.from_numpy(y_sequences[:n_train])
    X_test = torch.from_numpy(X_sequences[n_train:])
    y_test = torch.from_numpy(y_sequences[n_train:])
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test

# Generate Lorenz attractor data
def generate_lorenz_data(n_samples=10000, dt=0.01, train_ratio=0.8):
    """Generate Lorenz attractor time series data"""
    # Lorenz parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0
    
    # Initialize
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    z = np.zeros(n_samples)
    
    # Initial conditions
    x[0] = np.random.randn()
    y[0] = np.random.randn()
    z[0] = np.random.randn()
    
    # Generate trajectory
    for i in range(1, n_samples):
        dx = sigma * (y[i-1] - x[i-1])
        dy = x[i-1] * (rho - z[i-1]) - y[i-1]
        dz = x[i-1] * y[i-1] - beta * z[i-1]
        
        x[i] = x[i-1] + dx * dt
        y[i] = y[i-1] + dy * dt
        z[i] = z[i-1] + dz * dt
    
    # Normalize
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    z = (z - z.mean()) / z.std()
    
    # Create sequences for prediction
    seq_len = 20  # Use past 20 steps
    pred_horizon = 5  # Predict 5 steps ahead
    
    X_sequences = []
    y_sequences = []
    
    for i in range(seq_len, n_samples - pred_horizon):
        # Input: past seq_len timesteps of (x, y, z)
        X_seq = np.stack([x[i-seq_len:i], y[i-seq_len:i], z[i-seq_len:i]], axis=1)
        # Output: future x value
        y_seq = x[i+pred_horizon]
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)
    
    # Split train/test
    n_train = int(len(X_sequences) * train_ratio)
    X_train = torch.from_numpy(X_sequences[:n_train])
    y_train = torch.from_numpy(y_sequences[:n_train])
    X_test = torch.from_numpy(X_sequences[n_train:])
    y_test = torch.from_numpy(y_sequences[n_train:])
    
    return X_train, y_train, X_test, y_test

# Define models adapted for Lorenz prediction
class LN_Lorenz(nn.Module):
    def __init__(self, seq_len=20, input_dim=3, hidden_dim=64, output_dim=1):
        super().__init__()
        self.flatten_size = seq_len * input_dim
        self.linear1 = nn.Linear(self.flatten_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, seq_len, features) or (seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten
        
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        
        return x

class RNN_Lorenz(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) or (seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        gru_out, _ = self.gru(x)
        # Use last timestep
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        
        return output

class LSTM_Lorenz(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features) or (seq_len, features)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        
        return output

# Training function
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Testing
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs.squeeze(), y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    return train_losses, test_losses

# Modify the comparison function to test different baselines
def compare_methods_on_trained_models():
    # Generate data
    print("Generating Lorenz attractor data...")
    # X_train, y_train, X_test, y_test = generate_lorenz_data(n_samples=10000)
    X_train, y_train, X_test, y_test = generate_stock_data(
    ticker="GOOGL", # or "SPY" or any major stock/index
    seq_len=**50**,       # Use 50 time steps (approx 10 trading weeks)
    pred_horizon=**20**,  # Predict 20 steps ahead (approx 4 trading weeks)
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize models
    # models = {
    #     'Feedforward': LN_Lorenz(seq_len=20, input_dim=3, hidden_dim=64),
    #     'GRU': RNN_Lorenz(input_size=3, hidden_size=64),
    #     'LSTM': LSTM_Lorenz(input_size=3, hidden_size=64)
    # }
    # Update models initialization to input_dim=5 and use the new seq_len=50
    models = {
        'Feedforward': LN_Lorenz(seq_len=50, input_dim=5, hidden_dim=64),
        'GRU': RNN_Lorenz(input_size=5, hidden_size=64),
        'LSTM': LSTM_Lorenz(input_size=5, hidden_size=64)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train each model
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        train_losses, test_losses = train_model(model, X_train, y_train, X_test, y_test, epochs=200)
        trained_models[name] = model
        
        # Plot training curve
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'{name} Training Curve')
        plt.legend()
        plt.savefig(f'{name.lower()}_training_curve.png')
        plt.show()
    
    # Compare IG vs STIG on a test sample with different baselines
    test_sample_idx = 100
    test_input = X_test[test_sample_idx:test_sample_idx+1].to(device)  # (1, 20, 3)
    
    # Different baseline strategies
    baseline_strategies = {
        'zeros': torch.zeros_like(test_input).to(device),
        'constant_0.5': torch.ones_like(test_input).to(device) * 0.5,
        'constant_0.8': torch.ones_like(test_input).to(device) * 0.8,
        'mean': torch.ones_like(test_input).to(device) * test_input.mean(),
        'gaussian_noise': torch.randn_like(test_input).to(device) * 0.5,
        'uniform_random': torch.rand_like(test_input).to(device),
    }
    
    all_results = {}
    
    for baseline_name, baseline in baseline_strategies.items():
        print(f"\n{'='*60}")
        print(f"Testing with baseline: {baseline_name}")
        print(f"{'='*60}")
        
        results = {}
        
        for model_name, model in trained_models.items():
            print(f"\nAnalyzing {model_name} Model with {baseline_name} baseline")
            
            # Get prediction first (in eval mode)
            model.eval()
            with torch.no_grad():
                prediction = model(test_input).item()
                baseline_prediction = model(baseline).item()
                true_value = y_test[test_sample_idx].item()
                print(f"Input prediction: {prediction:.4f}")
                print(f"Baseline prediction: {baseline_prediction:.4f}")
                print(f"True value: {true_value:.4f}")
                print(f"Prediction difference: {abs(prediction - baseline_prediction):.4f}")
            
            # Set model to training mode for gradient computation (required for RNNs)
            model.train()
            
            # Compute attributions
            try:
                # Standard IG
                ig_attr = ig(model, test_input, baseline=baseline, target_class=0)
                
                # STIG
                stig_attr = stig(model, test_input, baseline=baseline, target_class=0)
                
                # Remove batch dimension for analysis
                ig_attr = ig_attr.squeeze(0)  # (20, 3)
                stig_attr = stig_attr.squeeze(0)  # (20, 3)
                
                # Compute metrics
                correlation = torch.corrcoef(torch.stack([ig_attr.flatten(), stig_attr.flatten()]))[0,1].item()
                mse = (ig_attr - stig_attr).pow(2).mean().item()
                cosine_sim = torch.cosine_similarity(ig_attr.flatten(), stig_attr.flatten(), dim=0).item()
                
                # Temporal analysis
                ig_temporal = ig_attr.abs().mean(dim=1)
                stig_temporal = stig_attr.abs().mean(dim=1)
                temporal_corr = torch.corrcoef(torch.stack([ig_temporal, stig_temporal]))[0,1].item()
                
                # Temporal variation
                ig_temporal_diff = torch.diff(ig_temporal).abs().mean().item()
                stig_temporal_diff = torch.diff(stig_temporal).abs().mean().item()
                
                results[model_name] = {
                    'correlation': correlation,
                    'mse': mse,
                    'cosine_sim': cosine_sim,
                    'temporal_corr': temporal_corr,
                    'ig_temporal_variation': ig_temporal_diff,
                    'stig_temporal_variation': stig_temporal_diff,
                    'prediction_diff': abs(prediction - baseline_prediction),
                    'ig_attr': ig_attr.cpu().detach().numpy(),
                    'stig_attr': stig_attr.cpu().detach().numpy()
                }
                
                print(f"IG vs STIG correlation: {correlation:.4f}")
                print(f"IG vs STIG MSE: {mse:.6f}")
                
            except Exception as e:
                print(f"Error computing attributions: {e}")
                import traceback
                traceback.print_exc()
            
            # Set back to eval mode
            model.eval()
        
        all_results[baseline_name] = results
    
    # Create summary visualization
    fig, axes = plt.subplots(3, len(baseline_strategies), figsize=(5*len(baseline_strategies), 12))
    
    for col_idx, (baseline_name, results) in enumerate(all_results.items()):
        for row_idx, (model_name, metrics) in enumerate(results.items()):
            if 'ig_attr' not in metrics:
                continue
            
            # Calculate difference
            diff = metrics['ig_attr'] - metrics['stig_attr']
            
            # Plot difference heatmap
            im = axes[row_idx, col_idx].imshow(diff.T, aspect='auto', cmap='RdBu')
            axes[row_idx, col_idx].set_title(f'{model_name}\n{baseline_name}\nr={metrics["correlation"]:.3f}')
            axes[row_idx, col_idx].set_xlabel('Time Steps')
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel('Features')
            plt.colorbar(im, ax=axes[row_idx, col_idx], fraction=0.046)
    
    plt.suptitle('IG vs STIG Difference Maps for Different Baselines', fontsize=16)
    plt.tight_layout()
    plt.savefig('baseline_comparison_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create correlation comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = ['Feedforward', 'GRU', 'LSTM']
    baseline_names = list(baseline_strategies.keys())
    
    for model_idx, model_name in enumerate(model_names):
        correlations = []
        for baseline_name in baseline_names:
            if baseline_name in all_results and model_name in all_results[baseline_name]:
                correlations.append(all_results[baseline_name][model_name]['correlation'])
            else:
                correlations.append(0)
        
        x = np.arange(len(baseline_names))
        ax.plot(x, correlations, 'o-', label=model_name, markersize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(baseline_names, rotation=45)
    ax.set_xlabel('Baseline Type')
    ax.set_ylabel('IG vs STIG Correlation')
    ax.set_title('Effect of Baseline Choice on IG vs STIG Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig('baseline_correlation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary table for all baselines
    print("\n" + "="*100)
    print("SUMMARY: Effect of Baseline on IG vs STIG Correlation")
    print("="*100)
    print(f"{'Baseline':<20} {'Model':<15} {'Correlation':<12} {'MSE':<12} {'Pred Diff':<12}")
    print("-" * 100)
    
    for baseline_name, results in all_results.items():
        for model_name, metrics in results.items():
            print(f"{baseline_name:<20} {model_name:<15} {metrics['correlation']:<12.4f} "
                  f"{metrics['mse']:<12.6f} {metrics['prediction_diff']:<12.4f}")
    
    print("\n" + "="*100)
    print("KEY INSIGHTS:")
    print("- Different baselines should reveal different aspects of model behavior")
    print("- Larger prediction differences (input vs baseline) may show bigger IG vs STIG differences")
    print("- Non-zero baselines might better reveal temporal dependencies in RNNs")
    print("="*100)

# Run the experiment
if __name__ == "__main__":
    compare_methods_on_trained_models()
"""
Train MesNet on Mini Pupper IMU data.

Ground truth options:
  1. IsaacSim: Export IMU + ground truth velocity/pose from simulation
  2. OptiTrack/Vicon: External motion capture (if available)
  3. Self-supervised: Use foot contact detection as pseudo ground truth
     (robot is stationary when all feet on ground → velocity = 0)

Usage:
    python train_mesnet.py --data data/sim_imu_gt.csv --epochs 50
    python train_mesnet.py --data data/walk_01.csv --mode self-supervised
"""

import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ai_imu_filter import MesNet, IEKFState


class IMUDataset(Dataset):
    """Dataset of IMU windows with ground truth velocity."""

    def __init__(self, csv_path, window_size=20, mode="supervised"):
        """
        Args:
            csv_path: Path to CSV with columns:
                supervised: timestamp, gx, gy, gz, ax, ay, az, vx_gt, vy_gt, vz_gt
                self-supervised: timestamp, gx, gy, gz, ax, ay, az [, contact_0..3]
            window_size: IMU window length for MesNet
            mode: "supervised" or "self-supervised"
        """
        self.window_size = window_size
        self.mode = mode

        # Load data
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.timestamps = np.array([float(r['timestamp']) for r in rows])
        self.imu = np.array([
            [float(r['gx']), float(r['gy']), float(r['gz']),
             float(r['ax']), float(r['ay']), float(r['az'])]
            for r in rows
        ])

        if mode == "supervised":
            self.gt_vel = np.array([
                [float(r['vx_gt']), float(r['vy_gt']), float(r['vz_gt'])]
                for r in rows
            ])
        else:
            # Self-supervised: zero velocity when all feet in contact
            if 'contact_0' in rows[0]:
                contacts = np.array([
                    [float(r[f'contact_{i}']) for i in range(4)]
                    for r in rows
                ])
                # All feet on ground → stationary
                self.stationary_mask = np.all(contacts > 0.5, axis=1)
            else:
                # No contact info — use low IMU acceleration as proxy
                accel_mag = np.linalg.norm(self.imu[:, 3:6] - np.array([0, 0, -9.81]), axis=1)
                self.stationary_mask = accel_mag < 0.5  # Nearly gravity-only

            self.gt_vel = np.zeros((len(rows), 3))  # Zero when stationary

        # Compute normalization stats
        self.u_loc = self.imu.mean(axis=0)
        self.u_std = self.imu.std(axis=0)
        self.u_std[self.u_std < 1e-6] = 1.0

        self.n_samples = len(rows) - window_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        window = self.imu[idx:idx + self.window_size]
        normed = (window - self.u_loc) / self.u_std

        # (6, window_size) for Conv1d
        x = torch.tensor(normed.T, dtype=torch.float32)

        if self.mode == "supervised":
            target_vel = torch.tensor(self.gt_vel[idx + self.window_size - 1], dtype=torch.float32)
            weight = 1.0
        else:
            target_vel = torch.zeros(3)
            weight = 1.0 if self.stationary_mask[idx + self.window_size - 1] else 0.0

        return x, target_vel, torch.tensor(weight, dtype=torch.float32)


def train(args):
    print(f"[TRAIN] Loading data from {args.data}")
    dataset = IMUDataset(args.data, window_size=args.window, mode=args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"[TRAIN] {len(dataset)} samples, mode={args.mode}")
    print(f"[TRAIN] IMU stats: loc={dataset.u_loc}, std={dataset.u_std}")

    model = MesNet(in_dim=6, out_dim=2, hidden=32)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0
        n_batches = 0

        for x_batch, vel_gt, weights in dataloader:
            # Forward: get covariance params
            cov_params = model(x_batch)  # (batch, 2)

            # Loss: the covariance params should be small when
            # the IEKF prediction matches ground truth, large when it doesn't.
            # Simplified: minimize weighted MSE of velocity prediction error
            # scaled by learned covariance.
            #
            # Full version would run IEKF forward and backprop through it.
            # This simplified version trains the network to predict
            # uncertainty from IMU patterns.

            scale = 10.0 ** cov_params  # (batch, 2)

            # Negative log-likelihood loss (learned covariance)
            # For each sample: loss = 0.5 * (error^2 / sigma^2 + log(sigma^2))
            # This encourages the network to output high variance when
            # prediction is bad, low variance when prediction is good.
            vel_error = vel_gt.norm(dim=1)  # Scalar error per sample
            sigma2 = scale[:, 0]  # Use first output as velocity uncertainty

            nll = 0.5 * (vel_error ** 2 / sigma2 + torch.log(sigma2))
            loss = (nll * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")

    # Save model and normalization stats
    output_path = args.output or f"mesnet_{args.mode}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'u_loc': dataset.u_loc.tolist(),
        'u_std': dataset.u_std.tolist(),
    }, output_path)
    print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training CSV path")
    parser.add_argument("--mode", default="supervised", choices=["supervised", "self-supervised"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="Output model path")
    args = parser.parse_args()

    train(args)

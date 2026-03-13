# AI-IMU Dead Reckoning for Mini Pupper 2

Adapted from [mbrossar/ai-imu-dr](https://github.com/mbrossar/ai-imu-dr) (MIT License).

## Purpose
Replace the fake/noisy observations in the RL controller:
- `base_lin_vel`: currently `cmd * 0.7` (no real odometry)
- `base_ang_vel`: currently raw gyro with offset subtraction (noisy)
- `projected_gravity`: currently hardcoded `[0, 0, -1]`

## Architecture
```
ESP32 IMU (accel + gyro @ 50Hz)
        │
        ▼
  AI-IMU MesNet (Conv1d, tiny)
        │
        ▼
  Learned covariance matrices
        │
        ▼
  IEKF (Iterated Extended Kalman Filter)
        │
        ▼
  base_lin_vel [3], base_ang_vel [3], projected_gravity [3]
```

## Pipeline
1. `collect_imu_data.py` — Record raw IMU + servo data on Mini Pupper
2. `train_mesnet.py` — Train MesNet on collected data (needs ground truth from sim or OptiTrack)
3. `ai_imu_filter.py` — IEKF + MesNet inference module (drop-in for controller)
4. Integration: replace obs[0:9] in mlp_controller_v3.py

## Key Differences from Original
- Original: 100 Hz KITTI car data, 6-DOF ground truth from GPS/LiDAR
- Ours: 50 Hz ESP32 IMU, ground truth from IsaacSim or manual annotation
- Network is identical (Conv1d 6→32→32→2, ~2K params, runs on RPi4)

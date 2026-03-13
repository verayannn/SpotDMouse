"""
Collect raw IMU data from Mini Pupper 2 ESP32 for AI-IMU training.

Records timestamped accelerometer + gyroscope readings at ~50 Hz.
Run this on the Mini Pupper RPi4 while walking the robot around.

Usage:
    python collect_imu_data.py --duration 60 --output data/walk_01.csv
    python collect_imu_data.py --duration 60 --output data/walk_01.csv --with-servos

Output CSV columns:
    timestamp, gx, gy, gz, ax, ay, az [, servo_pos_0..11]
"""

import argparse
import csv
import time
import numpy as np


def collect(duration, output_path, with_servos=False, rate=50):
    # Import hardware interface (only available on Mini Pupper)
    from MangDang.mini_pupper.HardwareInterface import HardwareInterface
    from MangDang.mini_pupper.Config import Configuration

    config = Configuration()
    hardware = HardwareInterface()
    esp32 = hardware.pwm_params.esp32
    time.sleep(0.3)

    # Calibrate gyro/accel offsets at rest
    print("[CALIBRATING] Hold robot still for 2 seconds...")
    gyro_samples, accel_samples = [], []
    for _ in range(100):
        imu = esp32.imu_get_data()
        if imu:
            gyro_samples.append([imu['gx'], imu['gy'], imu['gz']])
            accel_samples.append([imu['ax'], imu['ay'], imu['az']])
        time.sleep(0.02)

    gyro_offset = np.mean(gyro_samples, axis=0) if gyro_samples else np.zeros(3)
    accel_mean = np.mean(accel_samples, axis=0) if accel_samples else np.zeros(3)
    print(f"[CAL] Gyro offset: {gyro_offset}")
    print(f"[CAL] Accel mean:  {accel_mean}")

    # Prepare CSV
    fieldnames = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']
    if with_servos:
        fieldnames += [f'servo_{i}' for i in range(12)]

    dt_target = 1.0 / rate
    rows = []
    n_samples = int(duration * rate)

    print(f"[RECORDING] {duration}s at {rate}Hz ({n_samples} samples)")
    print("[RECORDING] Move the robot around now!")

    gyro_scale = np.pi / 180.0  # deg/s to rad/s
    accel_scale = 9.81           # g to m/s^2

    start_time = time.time()
    for i in range(n_samples):
        loop_start = time.time()

        imu = esp32.imu_get_data()
        if imu is None:
            continue

        row = {
            'timestamp': time.time() - start_time,
            # Raw IMU in SI units (rad/s, m/s^2)
            'gx': (imu['gx'] - gyro_offset[0]) * gyro_scale,
            'gy': (imu['gy'] - gyro_offset[1]) * gyro_scale,
            'gz': (imu['gz'] - gyro_offset[2]) * gyro_scale,
            'ax': imu['ax'] * accel_scale,
            'ay': imu['ay'] * accel_scale,
            'az': imu['az'] * accel_scale,
        }

        if with_servos:
            positions = esp32.servos_get_position()
            if positions:
                for j in range(min(12, len(positions))):
                    row[f'servo_{j}'] = positions[j]

        rows.append(row)

        elapsed = time.time() - loop_start
        if elapsed < dt_target:
            time.sleep(dt_target - elapsed)

        if (i + 1) % (rate * 5) == 0:
            print(f"  [{i+1}/{n_samples}] {(i+1)/rate:.0f}s elapsed")

    # Save
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    actual_duration = rows[-1]['timestamp'] if rows else 0
    actual_rate = len(rows) / actual_duration if actual_duration > 0 else 0
    print(f"[DONE] Saved {len(rows)} samples to {output_path}")
    print(f"  Actual duration: {actual_duration:.1f}s, rate: {actual_rate:.1f}Hz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect IMU data from Mini Pupper 2")
    parser.add_argument("--duration", type=float, default=60, help="Recording duration (seconds)")
    parser.add_argument("--output", type=str, default="imu_data.csv", help="Output CSV path")
    parser.add_argument("--with-servos", action="store_true", help="Also record servo positions")
    parser.add_argument("--rate", type=int, default=50, help="Sampling rate (Hz)")
    args = parser.parse_args()

    collect(args.duration, args.output, args.with_servos, args.rate)

#!/usr/bin/env python3
import time
import numpy as np
from MangDang.mini_pupper.HardwareInterface import HardwareInterface

hw = HardwareInterface()
esp32 = hw.pwm_params.esp32

TARGET_FREQ = 50
TARGET_DT = 1.0 / TARGET_FREQ
NUM_ITERATIONS = 200

print("=" * 60)
print(f"CONTROL LOOP TIMING TEST ({TARGET_FREQ}Hz target)")
print("=" * 60)

def dummy_policy(obs):
    return np.zeros(12)

def simulate_control_loop():
    loop_times = []
    read_times = []
    compute_times = []
    write_times = []

    prev_pos = None

    for i in range(NUM_ITERATIONS):
        loop_start = time.perf_counter()

        t0 = time.perf_counter()
        pos = esp32.servos_get_position()
        load = esp32.servos_get_load()
        imu = esp32.imu_get_data()
        t1 = time.perf_counter()

        obs = np.zeros(60)
        if pos and load and imu:
            obs[12:24] = pos
            obs[36:48] = load
            obs[3:6] = [imu['gx'], imu['gy'], imu['gz']]
        action = dummy_policy(obs)
        t2 = time.perf_counter()

        cmd = [512] * 12
        esp32.servos_set_position(cmd)
        t3 = time.perf_counter()

        read_times.append(t1 - t0)
        compute_times.append(t2 - t1)
        write_times.append(t3 - t2)

        elapsed = time.perf_counter() - loop_start
        sleep_time = TARGET_DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        loop_times.append(time.perf_counter() - loop_start)

    return np.array(loop_times), np.array(read_times), np.array(compute_times), np.array(write_times)

print("Running simulated control loop...")
loop_times, read_times, compute_times, write_times = simulate_control_loop()

print(f"\nLoop timing (target: {TARGET_DT*1000:.1f}ms):")
print(f"  Actual:  {np.mean(loop_times)*1000:.2f}ms ± {np.std(loop_times)*1000:.2f}ms")
print(f"  Min/Max: {np.min(loop_times)*1000:.2f}ms / {np.max(loop_times)*1000:.2f}ms")
print(f"  Achieved freq: {1/np.mean(loop_times):.1f}Hz")

print(f"\nBreakdown:")
print(f"  Read (pos+load+imu): {np.mean(read_times)*1000:.2f}ms ± {np.std(read_times)*1000:.2f}ms")
print(f"  Compute (policy):    {np.mean(compute_times)*1000:.2f}ms ± {np.std(compute_times)*1000:.2f}ms")
print(f"  Write (commands):    {np.mean(write_times)*1000:.2f}ms ± {np.std(write_times)*1000:.2f}ms")

overhead = loop_times - (read_times + compute_times + write_times)
print(f"  Sleep/overhead:      {np.mean(overhead)*1000:.2f}ms")

print(f"\nJitter analysis:")
dts = np.diff(np.cumsum(loop_times))
print(f"  dt between loops: {np.mean(dts)*1000:.2f}ms ± {np.std(dts)*1000:.2f}ms")

missed = np.sum(loop_times > TARGET_DT * 1.1)
print(f"  Missed deadlines (>10% over): {missed}/{NUM_ITERATIONS} ({100*missed/NUM_ITERATIONS:.1f}%)")

print("\n" + "=" * 60)
print("TIMING HISTOGRAM (loop times in ms)")
print("=" * 60)
hist, edges = np.histogram(loop_times * 1000, bins=10)
for i in range(len(hist)):
    bar = "#" * int(hist[i] / max(hist) * 40)
    print(f"  {edges[i]:5.1f}-{edges[i+1]:5.1f}ms: {bar} ({hist[i]})")

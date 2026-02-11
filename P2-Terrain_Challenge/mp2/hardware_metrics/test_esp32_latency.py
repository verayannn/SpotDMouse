#!/usr/bin/env python3
import time
import numpy as np
from MangDang.mini_pupper.HardwareInterface import HardwareInterface

hw = HardwareInterface()
esp32 = hw.pwm_params.esp32

def measure_latency(func, name, iterations=100):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        if result is not None:
            times.append((t1 - t0) * 1000)
    if times:
        print(f"{name}: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms (min={np.min(times):.2f}, max={np.max(times):.2f})")
    else:
        print(f"{name}: ALL CALLS FAILED")

print("=" * 60)
print("ESP32 LATENCY TEST (100 iterations each)")
print("=" * 60)

measure_latency(esp32.servos_get_position, "servos_get_position")
measure_latency(esp32.servos_get_load, "servos_get_load")
measure_latency(esp32.imu_get_data, "imu_get_data")

def all_reads():
    p = esp32.servos_get_position()
    l = esp32.servos_get_load()
    i = esp32.imu_get_data()
    return (p, l, i) if all([p, l, i]) else None

measure_latency(all_reads, "ALL THREE COMBINED")

print("=" * 60)
print("SEQUENTIAL vs TIMING")
print("=" * 60)
times_between = []
for _ in range(50):
    t0 = time.perf_counter()
    esp32.servos_get_position()
    t1 = time.perf_counter()
    esp32.servos_get_load()
    t2 = time.perf_counter()
    esp32.imu_get_data()
    t3 = time.perf_counter()
    times_between.append([t1-t0, t2-t1, t3-t2])

tb = np.array(times_between) * 1000
print(f"Position read:  {np.mean(tb[:,0]):.2f}ms")
print(f"Load read:      {np.mean(tb[:,1]):.2f}ms")
print(f"IMU read:       {np.mean(tb[:,2]):.2f}ms")
print(f"Total:          {np.mean(np.sum(tb, axis=1)):.2f}ms")

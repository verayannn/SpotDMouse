#!/usr/bin/env python3
import time
import numpy as np
from MangDang.mini_pupper.HardwareInterface import HardwareInterface

hw = HardwareInterface()
esp32 = hw.pwm_params.esp32

print("=" * 60)
print("SERVO VELOCITY CAPABILITY TEST")
print("=" * 60)

print("\nAvailable ESP32 methods:")
methods = [m for m in dir(esp32) if not m.startswith('_') and callable(getattr(esp32, m))]
for m in methods:
    print(f"  - {m}")

velocity_methods = [m for m in methods if 'vel' in m.lower() or 'speed' in m.lower()]
print(f"\nVelocity-related methods found: {velocity_methods if velocity_methods else 'NONE'}")

print("\n" + "=" * 60)
print("COMPUTING VELOCITY FROM POSITION DIFFERENCES")
print("=" * 60)

positions = []
timestamps = []

print("Collecting 100 position samples...")
for i in range(100):
    t = time.perf_counter()
    p = esp32.servos_get_position()
    if p:
        positions.append(p)
        timestamps.append(t)
    time.sleep(0.01)

positions = np.array(positions)
timestamps = np.array(timestamps)
dts = np.diff(timestamps)
dpositions = np.diff(positions, axis=0)
velocities = dpositions / dts[:, np.newaxis]

print(f"\nSamples collected: {len(positions)}")
print(f"Sample rate: {1/np.mean(dts):.1f} Hz")
print(f"dt stats: {np.mean(dts)*1000:.2f}ms ± {np.std(dts)*1000:.2f}ms")
print(f"\nComputed velocity stats (raw servo units/sec):")
print(f"  Mean:  {np.mean(np.abs(velocities)):.1f}")
print(f"  Max:   {np.max(np.abs(velocities)):.1f}")
print(f"  Noise floor (stationary robot): {np.std(velocities[:10]):.1f}")

print("\n" + "=" * 60)
print("LOAD (EFFORT) DATA TEST")
print("=" * 60)
loads = []
for _ in range(50):
    l = esp32.servos_get_load()
    if l:
        loads.append(l)
    time.sleep(0.02)

loads = np.array(loads)
print(f"Load samples: {len(loads)}")
print(f"Load range: [{np.min(loads)}, {np.max(loads)}]")
print(f"Load mean per servo: {np.mean(loads, axis=0).astype(int)}")
print(f"Load std per servo:  {np.std(loads, axis=0).astype(int)}")

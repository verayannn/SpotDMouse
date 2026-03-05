import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

def sine_func(t, amplitude, phase, offset, freq):
    """Standard sine wave function for fitting."""
    omega = 2 * np.pi * freq
    return amplitude * np.sin(omega * t + phase) + offset

def analyze_file(filename):
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

    freqs = sorted(df['freq_hz'].unique())
    results = {}
    for f in freqs:
        subset = df[df['freq_hz'] == f]
        t = subset['time_s'].values
        t = t - t[0]
        cmd = subset['command_rad'].values
        act = subset['actual_rad'].values
        omega = 2 * np.pi * f
        try:
            # Fit sine waves to Command and Actual data
            popt_cmd, _ = curve_fit(lambda t, a, p, o: sine_func(t, a, p, o, f), t, cmd, p0=[0.3, 0.0, 0.0])
            popt_act, _ = curve_fit(lambda t, a, p, o: sine_func(t, a, p, o, f), t, act, p0=[0.3, 0.0, 0.0])
            
            amp_cmd, phi_cmd = abs(popt_cmd[0]), popt_cmd[1]
            if popt_cmd[0] < 0: phi_cmd += np.pi
            
            amp_act, phi_act = abs(popt_act[0]), popt_act[1]
            if popt_act[0] < 0: phi_act += np.pi
            
            # Calculate Phase Difference (phi_cmd - phi_act)
            phase_diff = np.arctan2(np.sin(phi_cmd - phi_act), np.cos(phi_cmd - phi_act))
            
            # Convert Phase to Lag in milliseconds
            lag_s = phase_diff / omega
            if lag_s < 0:
                lag_s += (1.0/f)
            
            results[f] = {'lag_ms': lag_s * 1000, 'ratio': amp_act / amp_cmd}
        except:
            continue
    return results

# --- CONFIGURATION ---
import glob

JOINTS = ["LF_hip", "LF_thigh", "LF_calf"]
AMPLITUDE = 0.3

# Load all joint data
sim_data = {}
real_data = {}
for joint_name in JOINTS:
    sim_matches = glob.glob(f"sim_motor_*_{joint_name}_amp{AMPLITUDE}.csv")
    real_matches = glob.glob(f"real_motor_*_{joint_name}_amp{AMPLITUDE}.csv")
    if sim_matches:
        sim_data[joint_name] = analyze_file(sim_matches[0])
    if real_matches:
        real_data[joint_name] = analyze_file(real_matches[0])

all_freqs = sorted(set().union(
    *(r.keys() for r in sim_data.values() if r),
    *(r.keys() for r in real_data.values() if r),
))

print(f"\n{'='*80}")
print(f"  LF LEG COMPARISON — SIM vs REAL  (Amplitude={AMPLITUDE} rad)")
print(f"{'='*80}")

# Header
jw = 22  # column width per joint
print(f"\n{'':>8}", end="")
for j in JOINTS:
    print(f" | {j:^{jw}}", end="")
print()
print(f"{'Freq':>8}", end="")
for _ in JOINTS:
    print(f" | {'Sim Lag':>7} {'Real Lag':>8} {'LagΔ':>6}", end="")
print()
print("-" * (9 + (jw + 3) * len(JOINTS)))

for f in all_freqs:
    print(f"{f:>6.1f}Hz", end="")
    for j in JOINTS:
        s = sim_data.get(j, {})
        r = real_data.get(j, {})
        sl = s[f]['lag_ms'] if s and f in s else float('nan')
        rl = r[f]['lag_ms'] if r and f in r else float('nan')
        dl = sl - rl if not (np.isnan(sl) or np.isnan(rl)) else float('nan')
        print(f" | {sl:>6.1f}ms {rl:>7.1f}ms {dl:>+5.1f}", end="")
    print()

# Amplitude ratio table
print(f"\n{'':>8}", end="")
for j in JOINTS:
    print(f" | {j:^{jw}}", end="")
print()
print(f"{'Freq':>8}", end="")
for _ in JOINTS:
    print(f" | {'Sim Amp':>7} {'Real Amp':>8} {'AmpΔ':>6}", end="")
print()
print("-" * (9 + (jw + 3) * len(JOINTS)))

for f in all_freqs:
    print(f"{f:>6.1f}Hz", end="")
    for j in JOINTS:
        s = sim_data.get(j, {})
        r = real_data.get(j, {})
        sr = s[f]['ratio'] if s and f in s else float('nan')
        rr = r[f]['ratio'] if r and f in r else float('nan')
        dr = sr - rr if not (np.isnan(sr) or np.isnan(rr)) else float('nan')
        print(f" | {sr:>7.2f} {rr:>8.2f} {dr:>+5.2f}", end="")
    print()

# Summary averages
print(f"\n{'─'*80}")
print(f"  AVERAGES ACROSS FREQUENCIES")
print(f"{'─'*80}")
print(f"{'Joint':<12} | {'Avg Sim Lag':>11} | {'Avg Real Lag':>12} | {'Avg Lag Δ':>9} | {'Avg Sim Amp':>11} | {'Avg Real Amp':>12} | {'Avg Amp Δ':>9}")
print("-" * 88)
for j in JOINTS:
    s = sim_data.get(j, {})
    r = real_data.get(j, {})
    common = sorted(set(s.keys() if s else []) & set(r.keys() if r else []))
    if common:
        avg_sl = np.mean([s[f]['lag_ms'] for f in common])
        avg_rl = np.mean([r[f]['lag_ms'] for f in common])
        avg_sr = np.mean([s[f]['ratio'] for f in common])
        avg_rr = np.mean([r[f]['ratio'] for f in common])
        print(f"{j:<12} | {avg_sl:>9.1f}ms | {avg_rl:>10.1f}ms | {avg_sl-avg_rl:>+8.1f} | {avg_sr:>11.2f} | {avg_rr:>12.2f} | {avg_sr-avg_rr:>+8.2f}")
    else:
        print(f"{j:<12} | {'N/A':>11} | {'N/A':>12} | {'N/A':>9} | {'N/A':>11} | {'N/A':>12} | {'N/A':>9}")

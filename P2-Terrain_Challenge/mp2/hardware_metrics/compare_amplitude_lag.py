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
sim_file = 'sim_motor_5_LF_thigh_amp0.3.csv'
real_file = 'real_motor_1_LF_thigh_amp0.3.csv'

print(f"\n{'='*60}\nACTUATOR COMPARISON TOOL\n{'='*60}")

sim_results = analyze_file(sim_file)
real_results = analyze_file(real_file)

if sim_results:
    print(f"\nSIM Data: {sim_file}")
    print(f"{'Freq (Hz)':<10} | {'Lag (ms)':<10} | {'Amp Ratio':<10}")
    print("-" * 35)
    for f, data in sim_results.items():
        print(f"{f:<10.1f} | {data['lag_ms']:<10.2f} | {data['ratio']:<10.2f}")

if real_results:
    print(f"\nREAL Data: {real_file}")
    print(f"{'Freq (Hz)':<10} | {'Lag (ms)':<10} | {'Amp Ratio':<10}")
    print("-" * 35)
    for f, data in real_results.items():
        print(f"{f:<10.1f} | {data['lag_ms']:<10.2f} | {data['ratio']:<10.2f}")
else:
    print(f"\n[Note] Real motor file '{real_file}' not found for side-by-side comparison.")

if sim_results and real_results:
    print(f"\n{'='*15} SIDE-BY-SIDE (SIM vs REAL) {'='*15}")
    print(f"{'Freq':<6} | {'Sim Lag':<8} | {'Real Lag':<8} | {'Lag Diff':<8} | {'Sim Ratio':<9} | {'Real Ratio':<10}")
    print("-" * 75)
    common_freqs = sorted(set(sim_results.keys()) & set(real_results.keys()))
    for f in common_freqs:
        s_l, r_l = sim_results[f]['lag_ms'], real_results[f]['lag_ms']
        s_r, r_r = sim_results[f]['ratio'], real_results[f]['ratio']
        print(f"{f:<6.1f} | {s_l:<8.2f} | {r_l:<8.2f} | {s_l-r_l:<8.2f} | {s_r:<9.2f} | {r_r:<10.2f}")

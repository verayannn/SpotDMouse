#!/usr/bin/env python3
import time
import csv
import numpy as np
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration

config = Configuration()
hw = HardwareInterface()
esp32 = hw.pwm_params.esp32
servo_params = hw.servo_params
pwm_params = hw.pwm_params
time.sleep(0.3)

HW_TO_ISAAC = {0: 1, 1: 0, 2: 3, 3: 2}
REAL_DEFAULTS = np.array([0.0, 0.785, -0.785] * 4)
SIM_DEFAULTS = np.array([0.0, 0.785, -1.57] * 4)
JOINT_DIR = np.array([+1.0]*12)

def read_positions_isaac():
    raw = esp32.servos_get_position()
    if raw is None:
        return None
    angles_hw = np.zeros((3, 4))
    for leg in range(4):
        for axis in range(3):
            sid = pwm_params.servo_ids[axis, leg]
            pos = raw[sid - 1]
            dev = (servo_params.neutral_position - pos) / servo_params.micros_per_rad
            angles_hw[axis, leg] = dev / servo_params.servo_multipliers[axis, leg] + servo_params.neutral_angles[axis, leg]
    angles = np.zeros(12)
    for hw_col, isaac_leg in HW_TO_ISAAC.items():
        for axis in range(3):
            angles[isaac_leg * 3 + axis] = angles_hw[axis, hw_col]
    return angles

def sim_to_real(angles_sim):
    dev = (angles_sim - SIM_DEFAULTS) * JOINT_DIR
    return REAL_DEFAULTS + dev

def send_positions_sim_frame(flat_sim):
    real = sim_to_real(flat_sim)
    matrix = np.zeros((3, 4))
    matrix[:, 1] = real[0:3]
    matrix[:, 0] = real[3:6]
    matrix[:, 3] = real[6:9]
    matrix[:, 2] = real[9:12]
    hw.set_actuator_postions(matrix)

def run_freq_test(motor_idx, freq, amplitude, duration, square_wave=False):
    rows = []
    t_start = time.perf_counter()
    phase = 0.0
    prev_t = t_start

    while True:
        t_now = time.perf_counter()
        elapsed = t_now - t_start
        if elapsed > duration:
            break
        dt = t_now - prev_t
        prev_t = t_now
        phase += freq * 2 * np.pi * dt

        if square_wave:
            signal = np.sign(np.sin(phase))
        else:
            signal = np.sin(phase)
        command = signal * amplitude

        target = SIM_DEFAULTS.copy()
        target[motor_idx] += command
        send_positions_sim_frame(target)

        actual = read_positions_isaac()
        if actual is not None:
            actual_rel = actual[motor_idx] - SIM_DEFAULTS[motor_idx]
        else:
            actual_rel = float('nan')

        rows.append([elapsed, phase, amplitude, freq, motor_idx, signal, command, actual_rel])
        time.sleep(0.018)

    return rows

NAMES = ['LF_hip','LF_thigh','LF_calf','RF_hip','RF_thigh','RF_calf',
         'LB_hip','LB_thigh','LB_calf','RB_hip','RB_thigh','RB_calf']

TEST_MOTORS = [1, 4, 7, 10]
TEST_FREQS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
AMPLITUDE = 0.3
DURATION_PER_FREQ = 3.0
SQUARE = False

print("=" * 60)
print("REAL MOTOR RESPONSE CHARACTERIZATION")
print(f"Motors: {[NAMES[m] for m in TEST_MOTORS]}")
print(f"Freqs: {TEST_FREQS} Hz, Amp: {AMPLITUDE} rad")
print("=" * 60)

print("Moving to default pose...")
send_positions_sim_frame(SIM_DEFAULTS)
time.sleep(2.0)

for motor_idx in TEST_MOTORS:
    all_rows = []
    for freq in TEST_FREQS:
        print(f"  {NAMES[motor_idx]} @ {freq}Hz...", end=" ", flush=True)
        rows = run_freq_test(motor_idx, freq, AMPLITUDE, DURATION_PER_FREQ, SQUARE)
        all_rows.extend(rows)
        send_positions_sim_frame(SIM_DEFAULTS)
        time.sleep(0.5)
        print(f"{len(rows)} samples")

    fname = f"real_motor_{motor_idx}_{NAMES[motor_idx]}_amp{AMPLITUDE}.csv"
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s','phase','amp_rad','freq_hz','motor_idx','signal','command_rad','actual_rad'])
        w.writerows(all_rows)
    print(f"  Saved: {fname}")

send_positions_sim_frame(SIM_DEFAULTS)
print("\nDone. Compare these CSVs against simulation output.")

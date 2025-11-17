import numpy as np
from MangDang.mini_pupper.ServoCalibration import MICROS_PER_RAD, NEUTRAL_ANGLE_DEGREES
from MangDang.mini_pupper.HardwareConfig import PS4_COLOR, PS4_DEACTIVATED_COLOR
from enum import Enum

# TODO: put these somewhere else
class PWMParams:
    def __init__(self):
        self.pins = np.array([[15, 12, 9, 6], [14, 11, 8, 5], [13, 10, 7, 4]])
        self.range = 4096  ## ADC 12 bits
        self.freq = 250  ## PWM freq


class ServoParams:
    def __init__(self):
        self.neutral_position_pwm = 1500  # Middle position
        self.micros_per_rad = MICROS_PER_RAD  # Must be calibrated

        # The neutral angle of the joint relative to the modeled zero-angle in degrees, for each joint
        self.neutral_angle_degrees = NEUTRAL_ANGLE_DEGREES

        self.servo_multipliers = np.array(
            [[1, 1, -1, -1], [-1, 1, -1, 1], [-1, 1, -1, 1]]
        )

    @property
    def neutral_angles(self):
        return np.array(self.neutral_angle_degrees) * np.pi / 180.0  # Convert to radians


class Configuration:
    def __init__(self):
        ################# CONTROLLER BASE COLOR ##############
        self.ps4_color = PS4_COLOR
        self.ps4_deactivated_color = PS4_DEACTIVATED_COLOR

        #################### COMMANDS ####################
        self.max_x_velocity = 0.20
        self.max_y_velocity = 0.20
        self.max_yaw_rate = 2
        self.max_pitch = 20.0 * np.pi / 180.0

        #################### MOVEMENT PARAMS ####################
        self.z_time_constant = 0.02
        self.z_speed = 0.01  # maximum speed [m/s]
        self.pitch_deadband = 0.02
        self.pitch_time_constant = 0.25
        self.max_pitch_rate = 0.15
        self.roll_speed = 0.16  # maximum roll rate [rad/s] 0.16
        self.yaw_time_constant = 0.3

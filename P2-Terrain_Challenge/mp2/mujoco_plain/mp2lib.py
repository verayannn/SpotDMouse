# This serves as a living document for the mp2 libaries

#Audio libs
import sounddevice as sd
import soundfile as sf
import time
import os

#Servo interface
from MangDang.mini_pupper.ESP32Interface import ESP32Interface

esp32 = ESP32Interface()

#Get Positions
positions = esp32.servos_get_position()
print(f"Num Positions:{len(positions)}", f"Values: {positions}")
#Set Positions
#esp32.servos_set_postion_torque(positon, torque)

#Get IMU Data from sensor
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time

esp32 = ESP32Interface()

#while True:
#    print(esp32.imu_get_data())
#    time.sleep(1 / 20)  # 20 Hz

data = esp32.imu_get_data()
print("Shape:", len(data) if isinstance(data, (list, tuple)) else "N/A")
print("Values:", data)
print("Labels: [ax, ay, az, gx, gy, gz]")



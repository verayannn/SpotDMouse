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




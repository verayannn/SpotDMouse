from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time
import numpy as np

esp32 = ESP32Interface()

# Test what methods are available
print("Available methods:")
print(dir(esp32))

# Try to get position feedback
print("\nTesting servo feedback:")
for i in range(5):
    positions = esp32.servos_get_position()
    print(f"Positions: {positions}")
    
    # Check if there are other feedback methods
    if hasattr(esp32, 'servos_get_velocity'):
        velocities = esp32.servos_get_velocity()
        print(f"Velocities: {velocities}")
    
    if hasattr(esp32, 'servos_get_current') or hasattr(esp32, 'servos_get_load'):
        currents = esp32.servos_get_current() if hasattr(esp32, 'servos_get_current') else esp32.servos_get_load()
        print(f"Currents/Load: {currents}")
    
    if hasattr(esp32, 'servos_get_feedback'):
        feedback = esp32.servos_get_feedback()
        print(f"Full feedback: {feedback}")
    
    time.sleep(0.1)

# Also check IMU through ESP32
if hasattr(esp32, 'imu_get_data') or hasattr(esp32, 'get_imu'):
    print("\nIMU data from ESP32:")
    imu_data = esp32.imu_get_data() if hasattr(esp32, 'imu_get_data') else esp32.get_imu()
    print(imu_data)
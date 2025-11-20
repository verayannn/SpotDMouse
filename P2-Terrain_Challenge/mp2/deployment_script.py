import numpy as np
import torch
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
import threading
from nav_msgs.msg import Odometry

class MP2RealObservation:
    def __init__(self, esp32_interface, dt=0.02):
        self.esp32 = esp32_interface
        self.dt = dt
        self.prev_positions = None
        self.prev_actions = np.zeros(12)
        self.prev_time = time.time()
        
        # Velocity tracking
        self.velocity_command = np.zeros(3)
        self.base_linear_velocity = np.zeros(3)
        
        # Servo calibration
        self.servo_offset = 512
        self.servo_scale = 1024 / (2 * np.pi)
        
        # Get CURRENT positions as default
        try:
            current_servo_positions = np.array(self.esp32.servos_get_position())
            self.default_positions = (current_servo_positions - self.servo_offset) / self.servo_scale
            print(f"Initialized with current positions: {current_servo_positions}")
        except Exception as e:
            print(f"Error getting initial positions: {e}")
            # Fallback to standard positions if read fails
            self.default_positions = np.array([
                -0.0803, 1.0781, -1.9834,  # LF
                 0.0803, 1.0781, -1.9834,  # RF
                -0.0803, 1.0781, -1.9834,  # LB
                 0.0803, 1.0781, -1.9834   # RB
            ])
        
        # Initialize prev_positions with current positions
        self.prev_positions = self.default_positions.copy()
        
    def set_velocity_command(self, vx, vy, yaw_rate):
        """Set velocity commands"""
        self.velocity_command = np.array([vx, vy, yaw_rate])
        
    def update_previous_actions(self, actions):
        """Store last commanded actions"""
        self.prev_actions = actions.copy()
        
    def get_observation(self):
        current_time = time.time()
        actual_dt = current_time - self.prev_time
        
        try:
            # Get ESP32 data
            raw_positions = np.array(self.esp32.servos_get_position())
            raw_loads = np.array(self.esp32.servos_get_load())
            imu_data = self.esp32.imu_get_data()
        except Exception as e:
            print(f"Error reading ESP32 data: {e}")
            # Return last known good observation
            return self.get_default_observation()
        
        # Convert positions to radians (relative to default)
        absolute_positions = (raw_positions - self.servo_offset) / self.servo_scale
        relative_positions = absolute_positions - self.default_positions
        
        # Estimate velocities
        if self.prev_positions is not None and actual_dt > 0:
            joint_velocities = (absolute_positions - self.prev_positions) / actual_dt
            # Clip velocities to reasonable range
            joint_velocities = np.clip(joint_velocities, -50, 50)
        else:
            joint_velocities = np.zeros(12)
        
        # Convert loads to normalized effort
        joint_efforts = raw_loads / 500.0
        
        # Process IMU data
        angular_velocity = np.array([
            imu_data['gx'] * 0.01745,  # deg/s to rad/s
            imu_data['gy'] * 0.01745,
            imu_data['gz'] * 0.01745
        ])
        
        # Get projected gravity from accelerometer
        acc_norm = np.sqrt(imu_data['ax']**2 + imu_data['ay']**2 + imu_data['az']**2)
        if acc_norm > 0.1:
            projected_gravity = np.array([
                imu_data['ax'] / acc_norm,
                imu_data['ay'] / acc_norm,
                imu_data['az'] / acc_norm
            ])
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        # Build observation vector (60 dimensions)
        obs = np.concatenate([
            self.base_linear_velocity,  # 3
            angular_velocity,           # 3
            projected_gravity,          # 3
            self.velocity_command,      # 3
            relative_positions,         # 12
            joint_velocities,          # 12
            joint_efforts,             # 12
            self.prev_actions          # 12
        ])
        
        # Update state
        self.prev_positions = absolute_positions.copy()
        self.prev_time = current_time
        
        return obs
    
    def get_default_observation(self):
        """Return a safe default observation"""
        return np.concatenate([
            np.zeros(3),  # base_linear_velocity
            np.zeros(3),  # angular_velocity
            np.array([0.0, 0.0, -1.0]),  # projected_gravity
            np.zeros(3),  # velocity_command
            np.zeros(12),  # relative_positions
            np.zeros(12),  # joint_velocities
            np.zeros(12),  # joint_efforts
            np.zeros(12)   # prev_actions
        ])

class MP2RLController(Node):
    def __init__(self):
        super().__init__('mp2_rl_controller')
        
        # SAFETY PARAMETERS
        self.ACTION_SCALE = 0.1  # Start even lower - 10% of original
        self.MAX_ACTION_CHANGE = 0.05  # Smaller max change
        self.JOINT_LIMITS = {
            'hip': 0.3,      # Reduce limits further
            'thigh': 0.5,    
            'calf': 0.5      
        }
        
        # Test ESP32 connection
        try:
            self.esp32 = ESP32Interface()
            # Wait a bit for connection to stabilize
            time.sleep(0.5)
            test_pos = self.esp32.servos_get_position()
            if test_pos is None or len(test_pos) != 12:
                self.get_logger().error("ESP32 connection failed!")
                raise RuntimeError("Cannot connect to ESP32")
            self.get_logger().info(f"Current servo positions: {test_pos}")
        except Exception as e:
            self.get_logger().error(f"ESP32 initialization failed: {e}")
            raise

        # Initialize observation handler with ESP32 interface
        self.obs_handler = MP2RealObservation(self.esp32)
        
        # Load trained policy
        try:
            self.policy = torch.jit.load("/home/ubuntu/mp2_mlp/policy_only.pt")
            self.policy.eval()
            self.get_logger().info("Policy loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            raise
        
        # Subscribe to cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Don't subscribe to odom if it doesn't exist
        # self.odom_sub = self.create_subscription(
        #     Odometry,
        #     '/odom',
        #     self.odom_callback,
        #     10
        # )
        
        # Control flags
        self.control_active = False
        self.first_command_received = False
        self.last_actions = np.zeros(12)  # Track last actions for smoothing
        
        # Start control loop in thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        self.get_logger().info('MP2 RL Controller initialized with safety limits')
        self.get_logger().info(f'Action scale: {self.ACTION_SCALE}, Max change: {self.MAX_ACTION_CHANGE}')

    def odom_callback(self, msg):
        """Update base velocity from odometry"""
        self.obs_handler.base_linear_velocity[0] = msg.twist.twist.linear.x
        self.obs_handler.base_linear_velocity[1] = msg.twist.twist.linear.y
        self.obs_handler.base_linear_velocity[2] = msg.twist.twist.linear.z
            
    def cmd_vel_callback(self, msg):
        """Handle incoming velocity commands from ROS2"""
        vx = msg.linear.x
        vy = msg.linear.y
        yaw_rate = msg.angular.z
        
        # Activate control when we get non-zero commands
        if not self.first_command_received and (abs(vx) > 0.01 or abs(vy) > 0.01 or abs(yaw_rate) > 0.01):
            self.first_command_received = True
            self.control_active = True
            self.get_logger().info('First command received, activating control')
        
        self.obs_handler.set_velocity_command(vx, vy, yaw_rate)
        self.get_logger().debug(f'Received cmd_vel: vx={vx:.2f}, vy={vy:.2f}, yaw_rate={yaw_rate:.2f}')
    
    def apply_safety_limits(self, actions):
        """Apply safety limits to actions"""
        # Scale down actions
        scaled_actions = actions * self.ACTION_SCALE
        
        # Apply joint-specific limits
        limited_actions = scaled_actions.copy()
        for i in range(4):  # 4 legs
            base_idx = i * 3
            # Hip joint
            limited_actions[base_idx] = np.clip(limited_actions[base_idx], 
                                               -self.JOINT_LIMITS['hip'], 
                                               self.JOINT_LIMITS['hip'])
            # Thigh joint
            limited_actions[base_idx + 1] = np.clip(limited_actions[base_idx + 1], 
                                                   -self.JOINT_LIMITS['thigh'], 
                                                   self.JOINT_LIMITS['thigh'])
            # Calf joint
            limited_actions[base_idx + 2] = np.clip(limited_actions[base_idx + 2], 
                                                   -self.JOINT_LIMITS['calf'], 
                                                   self.JOINT_LIMITS['calf'])
        
        # Smooth actions - limit rate of change
        action_change = limited_actions - self.last_actions
        action_change = np.clip(action_change, -self.MAX_ACTION_CHANGE, self.MAX_ACTION_CHANGE)
        smoothed_actions = self.last_actions + action_change
        
        # Update last actions
        self.last_actions = smoothed_actions.copy()
        
        return smoothed_actions
    
    def control_loop(self):
        """Main control loop running at 50Hz"""
        # Wait a bit for ROS connections to establish
        time.sleep(1.0)
        
        while rclpy.ok():
            try:
                # Only send commands if control is active
                if self.control_active:
                    # Get observation (60 dims)
                    obs = self.obs_handler.get_observation()
                    
                    # Run policy inference
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        raw_actions = self.policy(obs_tensor).squeeze().numpy()
                    
                    # Apply safety limits and smoothing
                    safe_actions = self.apply_safety_limits(raw_actions)
                    
                    # Update previous actions for next observation
                    self.obs_handler.update_previous_actions(safe_actions)
                    
                    # Convert actions to servo positions and send
                    servo_positions = self.actions_to_servo_positions(safe_actions)
                    
                    # Torque must be integers (0 or 1)
                    torque = [1] * 12  # Full torque
                    
                    try:
                        self.esp32.servos_set_position_torque(servo_positions, torque)
                    except Exception as e:
                        self.get_logger().error(f'Error sending servo commands: {type(e).__name__}: {str(e)}')
                        # Print servo positions for debugging
                        self.get_logger().error(f'Servo positions: {servo_positions}')
                        self.get_logger().error(f'Torque: {torque}')
                    
                    # Log occasionally
                    if int(time.time() * 10) % 50 == 0:  # Every 5 seconds
                        self.get_logger().info(f'Actions range: [{np.min(safe_actions):.2f}, {np.max(safe_actions):.2f}]')
                        self.get_logger().info(f'Servo range: [{np.min(servo_positions)}, {np.max(servo_positions)}]')
                else:
                    # Just read current state but don't send commands
                    obs = self.obs_handler.get_observation()
                    # Keep previous actions as zeros when not active
                    self.obs_handler.update_previous_actions(np.zeros(12))
                
                time.sleep(0.02)  # 50Hz control loop
                    
            except Exception as e:
                self.get_logger().error(f'Error in control loop: {e}')
                import traceback
                traceback.print_exc()
                break
                
        # Set servos to safe position
        self.get_logger().info('Shutting down, maintaining current position')
    
    def actions_to_servo_positions(self, actions):
        """Convert relative joint positions to servo positions"""
        # Actions are relative to the starting position
        absolute_radians = actions + self.obs_handler.default_positions
        servo_positions = (absolute_radians * self.obs_handler.servo_scale + 
                          self.obs_handler.servo_offset)
        # Extra safety: ensure servo positions are in valid range
        servo_positions = np.clip(servo_positions, 100, 924)  # Leave margin
        # Convert to Python list of integers (critical!)
        return [int(pos) for pos in servo_positions]

def main(args=None):
    rclpy.init(args=args)
    
    # Create controller node
    controller = MP2RLController()
    
    print("\nMini Pupper RL Controller Started with Safety Limits!")
    print("Action scaling: 10% of original (very conservative)")
    print("Robot is maintaining current position.")
    print("Send velocity commands to activate:")
    print("ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.05, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'")
    print("\nNote: Starting with lower velocity (0.05) for safety")
    print("\nPress Ctrl+C to stop")
    
    # Spin ROS2 node to receive messages
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
import numpy as np
import torch
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
import threading
from nav_msgs.msg import Odometry  # Add this line
from models.actor_critic_mlp import ActorCriticMLP

# Load your trained ActorCritic model
checkpoint = torch.load("TRAINED_AC.pt")
model = ActorCriticMLP(obs_dim=59, action_dim=12)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract just the actor (policy) network
policy_only = model.actor

# Export as TorchScript for deployment
traced_policy = torch.jit.trace(policy_only, torch.randn(1, 59))
torch.jit.save(traced_policy, "RSLRL_TRAINED_MLP.pt")

print("Policy exported successfully!")

class MP2RealObservation:
    def __init__(self, dt=0.02):
        self.esp32 = ESP32Interface()
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
        
        # Default positions from your actual robot
        self.default_positions = np.array([
            -0.0803, 1.0781, -1.9834,  # LF
             0.0803, 1.0781, -1.9834,  # RF
            -0.0803, 1.0781, -1.9834,  # LB
             0.0803, 1.0781, -1.9834   # RB
        ])
        
    def set_velocity_command(self, vx, vy, yaw_rate):
        """Set velocity commands"""
        self.velocity_command = np.array([vx, vy, yaw_rate])
        
    def update_previous_actions(self, actions):
        """Store last commanded actions"""
        self.prev_actions = actions.copy()
        
    def get_observation(self):
        current_time = time.time()
        actual_dt = current_time - self.prev_time
        
        # Get ESP32 data
        raw_positions = np.array(self.esp32.servos_get_position())
        raw_loads = np.array(self.esp32.servos_get_load())
        imu_data = self.esp32.imu_get_data()
        
        # Convert positions to radians (relative to default)
        absolute_positions = (raw_positions - self.servo_offset) / self.servo_scale
        relative_positions = absolute_positions - self.default_positions
        
        # Estimate velocities
        if self.prev_positions is not None:
            joint_velocities = (absolute_positions - self.prev_positions) / actual_dt
        else:
            joint_velocities = np.zeros(12)
        
        # Convert loads to normalized effort
        joint_efforts = raw_loads / 500.0
        
        # Process IMU data - matching your actual system units
        angular_velocity = np.array([
            imu_data['gx'] * 0.01745,  # deg/s to rad/s
            imu_data['gy'] * 0.01745,
            imu_data['gz'] * 0.01745
        ])
        
        # Get projected gravity from accelerometer
        # Your IMU shows az ≈ -1g when upright
        acc_norm = np.sqrt(imu_data['ax']**2 + imu_data['ay']**2 + imu_data['az']**2)
        if acc_norm > 0.1:
            projected_gravity = np.array([
                imu_data['ax'] / acc_norm,
                imu_data['ay'] / acc_norm
            ])
        else:
            projected_gravity = np.array([0.0, 0.0])
        
        # Build observation vector (59 dimensions)
        obs = np.concatenate([
            self.base_linear_velocity,  # 3 - base linear velocity
            angular_velocity,           # 3 - base angular velocity
            projected_gravity,          # 2 - projected gravity
            self.velocity_command,      # 3 - velocity commands
            relative_positions,         # 12 - joint positions (relative)
            joint_velocities,          # 12 - joint velocities
            joint_efforts,             # 12 - joint efforts
            self.prev_actions          # 12 - previous actions
        ])
        
        # Update state
        self.prev_positions = absolute_positions.copy()
        self.prev_time = current_time
        
        return obs

class MP2RLController(Node):
    def __init__(self, policy_path, obs_normalizer_path=None):
        super().__init__('mp2_rl_controller')
        
        # Test ESP32 connection
        try:
            self.esp32 = ESP32Interface()
            test_pos = self.esp32.servos_get_position()
            if test_pos is None:
                self.get_logger().error("ESP32 connection failed!")
                raise RuntimeError("Cannot connect to ESP32")
        except Exception as e:
            self.get_logger().error(f"ESP32 initialization failed: {e}")
            raise

        self.obs_handler = MP2RealObservation()
        self.esp32 = ESP32Interface()
        
        # Load trained policy
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # Load observation normalizer if provided
        if obs_normalizer_path:
            self.obs_normalizer = torch.load(obs_normalizer_path)
        
        # Subscribe to cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.get_logger().info('MP2 RL Controller initialized, listening to /cmd_vel')

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
        
        self.obs_handler.set_velocity_command(vx, vy, yaw_rate)
        self.get_logger().debug(f'Received cmd_vel: vx={vx:.2f}, vy={vy:.2f}, yaw_rate={yaw_rate:.2f}')
    
    def control_loop(self):
        """Main control loop running at 50Hz"""
        try:
            while rclpy.ok():
                # Get observation (59 dims)
                obs = self.obs_handler.get_observation()
                
                # Normalize if needed
                if hasattr(self, 'obs_normalizer'):
                    obs = self.obs_normalizer(obs)
                
                # Run policy inference
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    
                    # Check if it's ActorCritic or just policy
                    if hasattr(self.policy, 'get_action'):
                        # It's an ActorCritic model
                        actions = self.policy.get_action(obs_tensor, deterministic=True).squeeze().numpy()
                    else:
                        # It's a policy-only model
                        actions = self.policy(obs_tensor).squeeze().numpy()
                
                # Update previous actions for next observation
                self.obs_handler.update_previous_actions(actions)
                
                # Convert actions to servo positions and send
                servo_positions = self.actions_to_servo_positions(actions)
                torque = [1] * 12  # Full torque
                self.esp32.servos_set_position_torque(servo_positions, torque)
                
                time.sleep(0.02)  # 50Hz control loop
                    
            except KeyboardInterrupt:
                self.get_logger().info('Stopping controller...')
                # Set servos to safe position
                safe_positions = [512] * 12  # Center position
                self.esp32.servos_set_position_torque(safe_positions, [0] * 12)
    
    def actions_to_servo_positions(self, actions):
        """Convert relative joint positions to servo positions"""
        absolute_radians = actions + self.obs_handler.default_positions
        servo_positions = (absolute_radians * self.obs_handler.servo_scale + 
                          self.obs_handler.servo_offset)
        return np.clip(servo_positions, 0, 1024).astype(int).tolist()

def main(args=None):
    rclpy.init(args=args)
    
    # Create controller node
    controller = MP2RLController(
        policy_path="RSLRL_TRAINED_MLP.pt",
        obs_normalizer_path=None
    )
    
    print("Starting Mini Pupper RL Controller...")
    print("Listening for velocity commands on /cmd_vel")
    print("Example: ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\" -r 10")
    print("\nPress Ctrl+C to stop")
    
    # Run control loop in separate thread
    control_thread = threading.Thread(target=controller.control_loop)
    control_thread.daemon = True
    control_thread.start()
    
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
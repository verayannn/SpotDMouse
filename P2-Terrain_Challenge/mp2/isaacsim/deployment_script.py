import numpy as np
import torch
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu

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

class MP2RLController:
    def __init__(self, policy_path, obs_normalizer_path=None):
        self.obs_handler = MP2RealObservation()
        self.esp32 = ESP32Interface()
        
        # Load trained policy
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # Load observation normalizer if provided
        if obs_normalizer_path:
            self.obs_normalizer = torch.load(obs_normalizer_path)
            
    def set_velocity_command(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        """Set velocity command for the robot"""
        self.obs_handler.set_velocity_command(vx, vy, yaw_rate)
    
    def run(self):
        """Main control loop"""
        while True:
            # Get observation
            obs = self.obs_handler.get_observation()
            
            # Normalize if needed
            if hasattr(self, 'obs_normalizer'):
                obs = self.obs_normalizer(obs)
            
            # Run policy inference
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                actions = self.policy(obs_tensor).squeeze().numpy()
            
            # Update previous actions in observation handler
            self.obs_handler.update_previous_actions(actions)
            
            # Convert actions to servo positions and send
            servo_positions = self.actions_to_servo_positions(actions)
            torque = [1] * 12  # Full torque
            self.esp32.servos_set_position_torque(servo_positions, torque)
            
            time.sleep(0.02)  # 50Hz control loop
    
    def actions_to_servo_positions(self, actions):
        """Convert relative joint positions to servo positions"""
        # Actions are joint position targets relative to default
        absolute_radians = actions + self.obs_handler.default_positions
        servo_positions = (absolute_radians * self.obs_handler.servo_scale + 
                          self.obs_handler.servo_offset)
        return np.clip(servo_positions, 0, 1024).astype(int).tolist()

#For ROS2 ~/ros2_ws/src/mini_pupper_ros/mini_pupper_driver/mini_pupper_driver/esp32_feedback_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import numpy as np

class ESP32FeedbackNode(Node):
    def __init__(self):
        super().__init__('esp32_feedback_node')
        self.esp32 = ESP32Interface()
        
        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states_esp32', 10)
        self.observation_pub = self.create_publisher(Float32MultiArray, '/robot_observation', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        # Timer for 50Hz publishing
        self.timer = self.create_timer(0.02, self.publish_observation)
        
        # State tracking
        self.prev_positions = None
        self.prev_actions = np.zeros(12)
        self.velocity_command = np.zeros(3)
        self.base_linear_velocity = np.zeros(3)
        self.base_angular_velocity = np.zeros(3)
        self.imu_data = {'ax': 0, 'ay': 0, 'az': -1, 'gx': 0, 'gy': 0, 'gz': 0}
        
        # Joint configuration
        self.joint_names = [
            'base_lf1', 'lf1_lf2', 'lf2_lf3',
            'base_rf1', 'rf1_rf2', 'rf2_rf3',
            'base_lb1', 'lb1_lb2', 'lb2_lb3',
            'base_rb1', 'rb1_rb2', 'rb2_rb3'
        ]
        
        # Default positions from your /joint_states topic
        self.default_positions = np.array([
            -0.0803, 1.0781, -1.9834,  # LF
             0.0803, 1.0781, -1.9834,  # RF
            -0.0803, 1.0781, -1.9834,  # LB
             0.0803, 1.0781, -1.9834   # RB
        ])
        
    def cmd_vel_callback(self, msg):
        """Update velocity commands"""
        self.velocity_command[0] = msg.linear.x
        self.velocity_command[1] = msg.linear.y
        self.velocity_command[2] = msg.angular.z
        
    def odom_callback(self, msg):
        """Get base velocity from odometry"""
        self.base_linear_velocity[0] = msg.twist.twist.linear.x
        self.base_linear_velocity[1] = msg.twist.twist.linear.y
        self.base_linear_velocity[2] = msg.twist.twist.linear.z
        self.base_angular_velocity[0] = msg.twist.twist.angular.x
        self.base_angular_velocity[1] = msg.twist.twist.angular.y
        self.base_angular_velocity[2] = msg.twist.twist.angular.z
        
    def imu_callback(self, msg):
        """Store IMU data for projected gravity calculation"""
        # Store normalized acceleration for gravity direction
        acc_norm = np.sqrt(msg.linear_acceleration.x**2 + 
                          msg.linear_acceleration.y**2 + 
                          msg.linear_acceleration.z**2)
        if acc_norm > 0.1:
            self.imu_data['ax'] = msg.linear_acceleration.x / acc_norm
            self.imu_data['ay'] = msg.linear_acceleration.y / acc_norm
            self.imu_data['az'] = msg.linear_acceleration.z / acc_norm
        
    def get_projected_gravity(self):
        """Get gravity direction projected to x-y plane"""
        return np.array([self.imu_data['ax'], self.imu_data['ay']])
        
    def publish_observation(self):
        # Get ESP32 data for joint feedback
        raw_positions = np.array(self.esp32.servos_get_position())
        raw_loads = np.array(self.esp32.servos_get_load())
        
        # Convert positions to radians
        absolute_positions = (raw_positions - 512) / (1024 / (2 * np.pi))
        relative_positions = absolute_positions - self.default_positions
        
        # Estimate velocities
        if self.prev_positions is not None:
            joint_velocities = (absolute_positions - self.prev_positions) / 0.02
        else:
            joint_velocities = np.zeros(12)
            
        # Normalize efforts
        joint_efforts = raw_loads / 500.0
        
        # Publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = ''
        joint_msg.name = self.joint_names
        joint_msg.position = absolute_positions.tolist()
        joint_msg.velocity = joint_velocities.tolist()
        joint_msg.effort = joint_efforts.tolist()
        self.joint_pub.publish(joint_msg)
        
        # Get projected gravity
        projected_gravity = self.get_projected_gravity()
        
        # Build and publish observation vector (59 dims)
        obs_msg = Float32MultiArray()
        obs_msg.data = np.concatenate([
            self.base_linear_velocity,      # 3 dims
            self.base_angular_velocity,     # 3 dims
            projected_gravity,              # 2 dims
            self.velocity_command,          # 3 dims
            relative_positions,             # 12 dims
            joint_velocities,               # 12 dims
            joint_efforts,                  # 12 dims
            self.prev_actions              # 12 dims
        ]).tolist()
        self.observation_pub.publish(obs_msg)
        
        # Update state
        self.prev_positions = absolute_positions.copy()
        
    def update_previous_actions(self, actions):
        """Called when new actions are sent to robot"""
        self.prev_actions = actions.copy()



#For RSL RL IsaacSim
@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Base velocities (6 dims)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        
        # Orientation (2 dims for projected gravity)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # Commands (3 dims)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        
        # Joint states (36 dims)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "base_lf1", "lf1_lf2", "lf2_lf3",
                    "base_rf1", "rf1_rf2", "rf2_rf3",
                    "base_lb1", "lb1_lb2", "lb2_lb3",
                    "base_rb1", "rb1_rb2", "rb2_rb3"
                ])
            }, 
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "base_lf1", "lf1_lf2", "lf2_lf3",
                    "base_rf1", "rf1_rf2", "rf2_rf3",
                    "base_lb1", "lb1_lb2", "lb2_lb3",
                    "base_rb1", "rb1_rb2", "rb2_rb3"
                ])
            }, 
            noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "base_lf1", "lf1_lf2", "lf2_lf3",
                    "base_rf1", "rf1_rf2", "rf2_rf3",
                    "base_lb1", "lb1_lb2", "lb2_lb3",
                    "base_rb1", "rb1_rb2", "rb2_rb3"
                ])
            },
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        
        # Previous actions (12 dims)
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()

class MP2RLController:
    def __init__(self, policy_path, obs_normalizer_path=None):
        self.obs_handler = MP2RealObservation()
        self.esp32 = ESP32Interface()
        
        # Load your trained policy
        self.policy = torch.jit.load(policy_path)
        self.policy.eval()
        
        # Load observation normalizer if you used one
        if obs_normalizer_path:
            self.obs_normalizer = torch.load(obs_normalizer_path)
    
    def run(self):
        while True:
            # Get observation
            obs = self.obs_handler.get_observation()
            
            # Normalize if needed
            if hasattr(self, 'obs_normalizer'):
                obs = self.obs_normalizer(obs)
            
            # Run policy inference
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                actions = self.policy(obs_tensor).squeeze().numpy()
            
            # Convert actions to servo positions and send
            servo_positions = self.actions_to_servo_positions(actions)
            torque = [1] * 12  # Full torque for now
            self.esp32.servos_set_position_torque(servo_positions, torque)
            
            time.sleep(0.02)  # 50Hz control loop
        
    def actions_to_servo_positions(self, actions):
        # Actions are joint position targets relative to default
        absolute_radians = actions + self.obs_handler.default_positions
        servo_positions = (absolute_radians * self.obs_handler.servo_scale + 
                        self.obs_handler.servo_offset)
        return np.clip(servo_positions, 0, 1024).astype(int).tolist()

if __name__ == "__main__":
    controller = MP2RLController(
        policy_path="RSLRL_TRAINED_MLP.pt",
        obs_normalizer_path="RSLRL_TRAINED_MLP.pt"
    )
    controller.run()
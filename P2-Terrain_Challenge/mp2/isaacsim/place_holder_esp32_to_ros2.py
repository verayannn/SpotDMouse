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
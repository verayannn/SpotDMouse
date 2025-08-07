import torch

# Load your checkpoint and examine it
checkpoint_path = "/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/sim2real/walkingmlp.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=== MODEL ARCHITECTURE ===")
print(f"First layer input size: {checkpoint['model_state_dict']['actor.0.weight'].shape[1]}")

print("\n=== CHECKPOINT KEYS ===")
for key in checkpoint.keys():
    print(f"- {key}")

print("\n=== MODEL STATE DICT KEYS (first 10) ===")
model_keys = list(checkpoint['model_state_dict'].keys())[:10]
for key in model_keys:
    tensor_shape = checkpoint['model_state_dict'][key].shape
    print(f"- {key}: {tensor_shape}")

print("\n=== LOOKING FOR OBSERVATION INFO ===")
# Check if there's any observation-related info stored
for key in checkpoint.keys():
    if 'obs' in key.lower() or 'observation' in key.lower():
        print(f"Found observation-related key: {key}")
        print(f"Value: {checkpoint[key]}")

print(f"\n=== STD VALUES (Action scaling) ===")
if 'std' in checkpoint['model_state_dict']:
    std_values = checkpoint['model_state_dict']['std']
    print(f"Action std shape: {std_values.shape}")
    print(f"Action std values: {std_values}")

# Check if training config is stored
if 'config' in checkpoint or 'cfg' in checkpoint:
    print(f"\n=== TRAINING CONFIG ===")
    config_key = 'config' if 'config' in checkpoint else 'cfg'
    print(checkpoint[config_key])

#!/usr/bin/env python3
"""
Audit what sensors/observations the Mini Pupper actually provides
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class MiniPupperSensorAudit(Node):
    def __init__(self):
        super().__init__('minipupper_sensor_audit')
        
        # Track what observations we can actually get
        self.available_observations = {}
        
        # Subscribe to all potential sensor topics
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
            
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
            
        # Create timer to print summary
        self.timer = self.create_timer(2.0, self.print_summary)
        
        self.get_logger().info('Mini Pupper sensor audit started...')
        
    def joint_state_callback(self, msg):
        """Audit joint state information"""
        joint_data = {
            'joint_names': msg.name,
            'num_joints': len(msg.name),
            'has_positions': len(msg.position) > 0,
            'has_velocities': len(msg.velocity) > 0,
            'has_efforts': len(msg.effort) > 0,
            'position_values': list(msg.position) if msg.position else [],
            'velocity_values': list(msg.velocity) if msg.velocity else [],
            'effort_values': list(msg.effort) if msg.effort else []
        }
        self.available_observations['joint_states'] = joint_data
        
    def imu_callback(self, msg):
        """Audit IMU information"""
        imu_data = {
            'has_orientation': True,
            'has_angular_velocity': True,
            'has_linear_acceleration': True,
            'orientation_quat': [msg.orientation.w, msg.orientation.x, 
                               msg.orientation.y, msg.orientation.z],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, 
                               msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y,
                                  msg.linear_acceleration.z]
        }
        self.available_observations['imu'] = imu_data
        
    def odom_callback(self, msg):
        """Audit odometry information"""
        odom_data = {
            'has_pose': True,
            'has_twist': True,
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y, 
                        msg.pose.pose.position.z],
            'orientation_quat': [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                               msg.pose.pose.orientation.y, msg.pose.pose.orientation.z],
            'linear_velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                              msg.twist.twist.linear.z],
            'angular_velocity': [msg.twist.twist.angular.x, msg.twist.twist.angular.y,
                               msg.twist.twist.angular.z]
        }
        self.available_observations['odometry'] = odom_data
        
    def print_summary(self):
        """Print summary of available observations"""
        self.get_logger().info('=== MINI PUPPER AVAILABLE OBSERVATIONS ===')
        
        total_dims = 0
        
        for sensor_name, data in self.available_observations.items():
            self.get_logger().info(f'\n--- {sensor_name.upper()} ---')
            
            if sensor_name == 'joint_states':
                self.get_logger().info(f'Joint count: {data["num_joints"]}')
                self.get_logger().info(f'Has positions: {data["has_positions"]} ({len(data["position_values"])} dims)')
                self.get_logger().info(f'Has velocities: {data["has_velocities"]} ({len(data["velocity_values"])} dims)')
                self.get_logger().info(f'Has efforts: {data["has_efforts"]} ({len(data["effort_values"])} dims)')
                
                if data["has_positions"]:
                    total_dims += len(data["position_values"])
                if data["has_velocities"]:
                    total_dims += len(data["velocity_values"])
                if data["has_efforts"]:
                    total_dims += len(data["effort_values"])
                    
            elif sensor_name == 'imu':
                self.get_logger().info(f'Orientation quaternion: 4 dims')
                self.get_logger().info(f'Angular velocity: 3 dims') 
                self.get_logger().info(f'Linear acceleration: 3 dims')
                total_dims += 10  # 4 + 3 + 3
                
            elif sensor_name == 'odometry':
                self.get_logger().info(f'Position: 3 dims')
                self.get_logger().info(f'Orientation: 4 dims')
                self.get_logger().info(f'Linear velocity: 3 dims')
                self.get_logger().info(f'Angular velocity: 3 dims')
                total_dims += 13  # 3 + 4 + 3 + 3
        
        # Calculate what we can construct for the MLP
        self.get_logger().info(f'\n=== CONSTRUCTIBLE OBSERVATIONS FOR MLP ===')
        mlp_dims = 0
        
        # Base linear velocity (from odometry): 3 dims
        mlp_dims += 3
        self.get_logger().info(f'Base linear velocity: 3 dims')
        
        # Base angular velocity (from odometry or IMU): 3 dims  
        mlp_dims += 3
        self.get_logger().info(f'Base angular velocity: 3 dims')
        
        # Projected gravity (from IMU acceleration): 3 dims
        mlp_dims += 3
        self.get_logger().info(f'Projected gravity: 3 dims')
        
        # Velocity commands (we set these): 3 dims
        mlp_dims += 3
        self.get_logger().info(f'Velocity commands: 3 dims')
        
        # Joint positions: 12 dims
        mlp_dims += 12
        self.get_logger().info(f'Joint positions: 12 dims')
        
        # Joint velocities: 12 dims  
        mlp_dims += 12
        self.get_logger().info(f'Joint velocities: 12 dims')
        
        # Last action: 12 dims
        mlp_dims += 12
        self.get_logger().info(f'Last action: 12 dims')
        
        self.get_logger().info(f'\nTOTAL MLP OBSERVATION DIMS: {mlp_dims}')
        self.get_logger().info(f'MODEL EXPECTS: 76')
        self.get_logger().info(f'MISSING: {76 - mlp_dims} dimensions')
        
        # Suggest additional observations we could derive
        self.get_logger().info(f'\n=== POSSIBLE ADDITIONAL OBSERVATIONS ===')
        extra_dims = 0
        
        if 'joint_states' in self.available_observations and self.available_observations['joint_states']['has_efforts']:
            self.get_logger().info(f'Joint efforts: 12 dims')
            extra_dims += 12
            
        if 'imu' in self.available_observations:
            self.get_logger().info(f'Full IMU orientation quaternion: 4 dims') 
            extra_dims += 4
            
        self.get_logger().info(f'Full odometry pose: 7 dims (pos + quat)')
        extra_dims += 7
        
        self.get_logger().info(f'Estimated foot contact states: 4 dims')
        extra_dims += 4
        
        self.get_logger().info(f'POSSIBLE TOTAL: {mlp_dims + extra_dims} dims')


def main():
    rclpy.init()
    
    audit_node = MiniPupperSensorAudit()
    
    try:
        rclpy.spin(audit_node)
    except KeyboardInterrupt:
        pass
    finally:
        audit_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

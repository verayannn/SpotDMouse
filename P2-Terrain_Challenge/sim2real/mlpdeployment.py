#!/usr/bin/env python3
"""
Test scripts for systematic MLP deployment testing
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import numpy as np
from sensor_msgs.msg import JointState

class StationaryTest(Node):
    """Test 1: Keep robot stationary to verify basic control"""
    
    def __init__(self):
        super().__init__('stationary_test')
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_commands)
        
        self.get_logger().info('Stationary test started - robot should hold position')
    
    def publish_commands(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

class WalkingTest(Node):
    """Test 2: Progressive walking commands"""
    
    def __init__(self):
        super().__init__('walking_test')
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_commands)
        
        self.start_time = time.time()
        self.test_phases = [
            (0, 5, 0.0),    # Hold still for 5 seconds
            (5, 10, 0.5),   # Slow forward for 5 seconds  
            (10, 15, 1.0),  # Medium forward for 5 seconds
            (15, 20, 1.5),  # Fast forward for 5 seconds
            (20, 25, 0.0),  # Stop for 5 seconds
        ]
        
        self.get_logger().info('Walking test started - progressive velocity commands')
    
    def publish_commands(self):
        current_time = time.time() - self.start_time
        
        # Find current test phase
        target_vel = 0.0
        for start, end, vel in self.test_phases:
            if start <= current_time < end:
                target_vel = vel
                break
        
        # Gradual velocity transitions
        msg = Twist()
        msg.linear.x = target_vel
        msg.linear.y = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)
        
        # Log current phase
        if int(current_time) % 2 == 0:  # Every 2 seconds
            self.get_logger().info(f'Time: {current_time:.1f}s, Target vel: {target_vel:.1f} m/s')

class SafetyMonitor(Node):
    """Monitor robot state and provide emergency stop"""
    
    def __init__(self):
        super().__init__('safety_monitor')
        
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.monitor_callback, 10 # might crash because joint state is not a named veraible 
        )
        
        self.emergency_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Safety thresholds
        self.max_joint_velocity = 5.0  # rad/s
        self.max_joint_position = 3.14  # rad
        
        self.get_logger().info('Safety monitor active')
    
    def monitor_callback(self, msg):
        """Monitor for dangerous conditions"""
        try:
            # Check joint velocities
            if hasattr(msg, 'velocity') and len(msg.velocity) > 0:
                max_vel = max(abs(v) for v in msg.velocity)
                if max_vel > self.max_joint_velocity:
                    self.emergency_stop(f'High joint velocity: {max_vel:.2f} rad/s')
                    return
            
            # Check joint positions
            if hasattr(msg, 'position') and len(msg.position) > 0:
                max_pos = max(abs(p) for p in msg.position)
                if max_pos > self.max_joint_position:
                    self.emergency_stop(f'Extreme joint position: {max_pos:.2f} rad')
                    return
                    
        except Exception as e:
            self.get_logger().error(f'Safety monitor error: {e}')
    
    def emergency_stop(self, reason):
        """Send emergency stop command"""
        self.get_logger().error(f'EMERGENCY STOP: {reason}')
        
        # Send zero velocity command
        stop_msg = Twist()
        for _ in range(10):  # Send multiple times to ensure receipt
            self.emergency_pub.publish(stop_msg)
            time.sleep(0.01)

# Test execution functions
def run_stationary_test():
    rclpy.init()
    node = StationaryTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def run_walking_test():
    rclpy.init()
    node = WalkingTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def run_safety_monitor():
    rclpy.init()
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'stationary':
            run_stationary_test()
        elif sys.argv[1] == 'walking':
            run_walking_test()
        elif sys.argv[1] == 'safety':
            run_safety_monitor()
    else:
        print("Usage: python test_scripts.py [stationary|walking|safety]")
#!/usr/bin/env python3
"""
Test script to verify joint order and direction via ROS2.
Moves each joint individually by +0.52 radians (~30 degrees) to verify:
1. Which physical joint moves
2. Direction of movement (positive = which direction?)

Expected Isaac Sim order:
- LF: base_lf1, lf1_lf2, lf2_lf3  (indices 0-2)
- RF: base_rf1, rf1_rf2, rf2_rf3  (indices 3-5)
- LB: base_lb1, lb1_lb2, lb2_lb3  (indices 6-8)
- RB: base_rb1, rb1_rb2, rb2_rb3  (indices 9-11)
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import time
import numpy as np

class JointTester(Node):
    def __init__(self):
        super().__init__('joint_tester')

        # Publisher to joint trajectory controller
        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_group_effort_controller/joint_trajectory',
            10
        )

        # Subscriber to get current joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.current_joint_positions = None
        self.joint_names = [
            'base_lf1', 'lf1_lf2', 'lf2_lf3',  # LF (0-2)
            'base_rf1', 'rf1_rf2', 'rf2_rf3',  # RF (3-5)
            'base_lb1', 'lb1_lb2', 'lb2_lb3',  # LB (6-8)
            'base_rb1', 'rb1_rb2', 'rb2_rb3'   # RB (9-11)
        ]

        self.get_logger().info("Joint Tester Node Started")
        self.get_logger().info("Waiting for current joint positions...")

    def joint_state_callback(self, msg):
        """Store current joint positions"""
        # Map joint names to positions
        joint_dict = dict(zip(msg.name, msg.position))
        self.current_joint_positions = [joint_dict.get(name, 0.0) for name in self.joint_names]

    def publish_joint_command(self, positions):
        """Publish joint trajectory command"""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 500_000_000  # 0.5 seconds

        msg.points = [point]
        self.pub.publish(msg)

    def neutral_pose(self):
        """Move to neutral standing pose"""
        neutral = [
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57   # RB
        ]
        self.get_logger().info("Moving to neutral pose...")
        self.publish_joint_command(neutral)
        return neutral

    def test_joint(self, joint_idx, offset=0.52):
        """Test a single joint by adding offset"""
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint states received yet!")
            return

        test_position = self.current_joint_positions.copy()
        test_position[joint_idx] += offset

        leg_names = ['LF', 'LF', 'LF', 'RF', 'RF', 'RF', 'LB', 'LB', 'LB', 'RB', 'RB', 'RB']
        joint_types = ['Hip', 'Thigh', 'Calf'] * 4

        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"Testing Joint {joint_idx}: {self.joint_names[joint_idx]}")
        self.get_logger().info(f"  Leg: {leg_names[joint_idx]}, Type: {joint_types[joint_idx]}")
        self.get_logger().info(f"  Command: {test_position[joint_idx]:.3f} rad (+{offset:.3f})")
        self.get_logger().info(f"{'='*60}")

        self.publish_joint_command(test_position)

    def run_test_sequence(self):
        """Run through all joints systematically"""
        # Wait for joint states
        rate = self.create_rate(10)
        while self.current_joint_positions is None and rclpy.ok():
            self.get_logger().info("Waiting for joint states...")
            rclpy.spin_once(self, timeout_sec=0.1)

        if not rclpy.ok():
            return

        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("JOINT ORDER AND DIRECTION TEST")
        self.get_logger().info("="*60)

        # Move to neutral
        neutral = self.neutral_pose()
        time.sleep(2.0)

        # Test each joint
        for i in range(12):
            input(f"\nPress Enter to test joint {i} ({self.joint_names[i]})...")
            self.test_joint(i, offset=0.52)  # +30 degrees
            time.sleep(1.5)

            # Return to neutral
            self.publish_joint_command(neutral)
            time.sleep(1.0)

        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("TEST COMPLETE")
        self.get_logger().info("="*60)

def main():
    rclpy.init()
    node = JointTester()

    try:
        node.run_test_sequence()
    except KeyboardInterrupt:
        node.get_logger().info("Test interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

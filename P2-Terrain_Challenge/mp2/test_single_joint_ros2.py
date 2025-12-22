#!/usr/bin/env python3
"""
Quick test script to move a SINGLE joint via ROS2.
Usage: python3 test_single_joint_ros2.py <joint_index> <angle_radians>

Example:
  python3 test_single_joint_ros2.py 0 0.52    # Test LF hip with +30 degrees
  python3 test_single_joint_ros2.py 1 -0.52   # Test LF thigh with -30 degrees
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sys
import time

class SingleJointTester(Node):
    def __init__(self):
        super().__init__('single_joint_tester')

        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_group_effort_controller/joint_trajectory',
            10
        )

        self.joint_names = [
            'base_lf1', 'lf1_lf2', 'lf2_lf3',  # LF (0-2)
            'base_rf1', 'rf1_rf2', 'rf2_rf3',  # RF (3-5)
            'base_lb1', 'lb1_lb2', 'lb2_lb3',  # LB (6-8)
            'base_rb1', 'rb1_rb2', 'rb2_rb3'   # RB (9-11)
        ]

        # Neutral standing pose
        self.neutral = [
            0.0,  0.785, -1.57,  # LF
            0.0,  0.785, -1.57,  # RF
            0.0,  0.785, -1.57,  # LB
            0.0,  0.785, -1.57   # RB
        ]

    def publish_joint_command(self, positions, duration_sec=0.5):
        """Publish joint trajectory command"""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(duration_sec * 1e9)

        msg.points = [point]
        self.pub.publish(msg)

    def test_joint(self, joint_idx, angle):
        """Move to neutral, then test specified joint"""
        leg_names = ['LF', 'LF', 'LF', 'RF', 'RF', 'RF', 'LB', 'LB', 'LB', 'RB', 'RB', 'RB']
        joint_types = ['Hip', 'Thigh', 'Calf'] * 4

        print(f"\n{'='*60}")
        print(f"Joint {joint_idx}: {self.joint_names[joint_idx]}")
        print(f"  Leg: {leg_names[joint_idx]}, Type: {joint_types[joint_idx]}")
        print(f"  Neutral: {self.neutral[joint_idx]:.3f} rad")
        print(f"  Commanded: {angle:.3f} rad")
        print(f"  Offset: {angle - self.neutral[joint_idx]:+.3f} rad")
        print(f"{'='*60}\n")

        print("All 12 joint positions (for debugging):")
        for i, name in enumerate(self.joint_names):
            print(f"  [{i}] {name}: {self.neutral[i]:.3f} rad")
        print()

        # Move to neutral first - publish multiple times to ensure it's received
        print("[1/3] Moving to neutral pose...")
        for _ in range(3):
            self.publish_joint_command(self.neutral, duration_sec=1.0)
            time.sleep(0.1)
        time.sleep(2.0)

        # Test the joint - create explicit position array
        test_position = [
            self.neutral[0], self.neutral[1], self.neutral[2],   # LF
            self.neutral[3], self.neutral[4], self.neutral[5],   # RF
            self.neutral[6], self.neutral[7], self.neutral[8],   # LB
            self.neutral[9], self.neutral[10], self.neutral[11]  # RB
        ]
        test_position[joint_idx] = angle

        print(f"[2/3] Moving joint {joint_idx} to {angle:.3f} rad...")
        print(f"  Other joints stay at neutral")
        for _ in range(3):
            self.publish_joint_command(test_position, duration_sec=1.0)
            time.sleep(0.1)
        time.sleep(2.0)

        print("[3/3] Returning to neutral...")
        for _ in range(3):
            self.publish_joint_command(self.neutral, duration_sec=1.0)
            time.sleep(0.1)
        time.sleep(1.0)

        print("\nDone! Check which leg moved.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 test_single_joint_ros2.py <joint_index> <angle_radians>")
        print("\nJoint indices:")
        print("  0-2:  LF (Hip, Thigh, Calf)")
        print("  3-5:  RF (Hip, Thigh, Calf)")
        print("  6-8:  LB (Hip, Thigh, Calf)")
        print("  9-11: RB (Hip, Thigh, Calf)")
        print("\nExample: python3 test_single_joint_ros2.py 0 0.52")
        sys.exit(1)

    joint_idx = int(sys.argv[1])
    angle = float(sys.argv[2])

    if joint_idx < 0 or joint_idx > 11:
        print("Error: joint_index must be 0-11")
        sys.exit(1)

    rclpy.init()
    node = SingleJointTester()

    # Give ROS2 a moment to connect
    time.sleep(0.5)

    try:
        node.test_joint(joint_idx, angle)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

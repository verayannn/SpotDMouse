#!/usr/bin/env python3
"""
Carefully test ONE joint at a time by:
1. Reading current joint states
2. Using exact joint names from the robot
3. Only modifying one joint value
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import sys
import time

class CarefulJointTester(Node):
    def __init__(self):
        super().__init__('careful_joint_tester')

        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_group_effort_controller/joint_trajectory',
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.current_joint_names = None
        self.current_joint_positions = None
        self.joint_state_received = False

    def joint_state_callback(self, msg):
        """Store EXACT joint names and positions from robot"""
        if not self.joint_state_received:
            print("\nReceived joint states from robot!")
            print(f"Number of joints: {len(msg.name)}")
            print("Joint names:")
            for i, name in enumerate(msg.name):
                print(f"  [{i:2d}] {name}")
            self.joint_state_received = True

        self.current_joint_names = list(msg.name)
        self.current_joint_positions = list(msg.position)

    def wait_for_joint_states(self, timeout=5.0):
        """Wait for joint states to be received"""
        start_time = time.time()
        rate = self.create_rate(10)

        while not self.joint_state_received and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.joint_state_received

    def publish_trajectory(self, joint_names, positions, duration_sec=1.0):
        """Publish joint trajectory"""
        msg = JointTrajectory()
        msg.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = []  # Empty velocities
        point.accelerations = []  # Empty accelerations
        point.effort = []  # Empty effort
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)

        msg.points = [point]

        self.get_logger().info(f"Publishing trajectory with {len(joint_names)} joints")
        self.pub.publish(msg)

    def test_single_joint(self, joint_index, delta_angle):
        """Test a single joint by modifying only its position"""

        if self.current_joint_names is None or self.current_joint_positions is None:
            print("ERROR: No joint states received yet!")
            return False

        if joint_index < 0 or joint_index >= len(self.current_joint_names):
            print(f"ERROR: Joint index {joint_index} out of range (0-{len(self.current_joint_names)-1})")
            return False

        print("\n" + "="*60)
        print(f"Testing joint {joint_index}: {self.current_joint_names[joint_index]}")
        print("="*60)

        # Create test positions - copy current positions
        test_positions = self.current_joint_positions.copy()

        print(f"\nCurrent position: {test_positions[joint_index]:.3f} rad")
        print(f"Delta: {delta_angle:+.3f} rad")

        # Modify only the target joint
        test_positions[joint_index] += delta_angle

        print(f"Target position: {test_positions[joint_index]:.3f} rad")
        print(f"\nAll joint positions being sent:")
        for i, (name, pos) in enumerate(zip(self.current_joint_names, test_positions)):
            marker = " <-- MODIFIED" if i == joint_index else ""
            print(f"  [{i:2d}] {name:20s}: {pos:+7.3f} rad{marker}")

        # Publish the command
        print("\n[1/2] Moving joint...")
        self.publish_trajectory(self.current_joint_names, test_positions, duration_sec=1.5)
        time.sleep(2.5)

        # Return to original
        print("[2/2] Returning to original position...")
        self.publish_trajectory(self.current_joint_names, self.current_joint_positions, duration_sec=1.0)
        time.sleep(1.5)

        print("\nTest complete!")
        return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 test_one_joint_careful.py <joint_index> <delta_angle_rad>")
        print("\nExample: python3 test_one_joint_careful.py 0 0.3")
        print("  This will move joint 0 by +0.3 radians from its current position")
        sys.exit(1)

    joint_index = int(sys.argv[1])
    delta_angle = float(sys.argv[2])

    rclpy.init()
    node = CarefulJointTester()

    print("="*60)
    print("CAREFUL JOINT TESTER")
    print("="*60)
    print("\nWaiting for joint states from robot...")

    if not node.wait_for_joint_states(timeout=5.0):
        print("\nERROR: No joint states received within timeout!")
        print("Is the robot running? Check: ros2 topic echo /joint_states")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Give system a moment to stabilize
    time.sleep(0.5)

    try:
        success = node.test_single_joint(joint_index, delta_angle)
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

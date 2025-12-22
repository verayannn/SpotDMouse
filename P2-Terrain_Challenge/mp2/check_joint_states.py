#!/usr/bin/env python3
"""
Check what's actually being published on /joint_states
This will tell us the actual joint names and order from the robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateChecker(Node):
    def __init__(self):
        super().__init__('joint_state_checker')

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.callback,
            10
        )
        self.count = 0

    def callback(self, msg):
        self.count += 1
        if self.count == 1:  # Only print first message
            print("\n" + "="*60)
            print("JOINT STATES MESSAGE")
            print("="*60)
            print(f"\nNumber of joints: {len(msg.name)}")
            print(f"\nJoint names (in order):")
            for i, name in enumerate(msg.name):
                pos = msg.position[i] if i < len(msg.position) else None
                vel = msg.velocity[i] if i < len(msg.velocity) else None
                eff = msg.effort[i] if i < len(msg.effort) else None

                pos_str = f"{pos:+7.3f}" if pos is not None else "N/A    "
                vel_str = f"{vel:+7.3f}" if vel is not None else "N/A    "
                eff_str = f"{eff:+7.3f}" if eff is not None else "N/A    "

                print(f"  [{i:2d}] {name:20s}  pos: {pos_str}  vel: {vel_str}  eff: {eff_str}")

            print("\n" + "="*60)
            print("Copy this joint name order for the MLP controller!")
            print("="*60)
            print(f"joint_names = {list(msg.name)}")
            print("\n")

            self.get_logger().info("First message received. Continuing to monitor...")

def main():
    rclpy.init()
    node = JointStateChecker()

    print("Listening to /joint_states...")
    print("Waiting for first message...\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

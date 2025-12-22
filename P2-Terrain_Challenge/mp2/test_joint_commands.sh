#!/bin/bash
# Test individual joints via ros2 topic pub
# Usage: ./test_joint_commands.sh

echo "Testing joint commands via ros2 topic pub"
echo "=========================================="
echo ""
echo "Joint order:"
echo "  0-2:  LF (base_lf1, lf1_lf2, lf2_lf3)"
echo "  3-5:  RF (base_rf1, rf1_rf2, rf2_rf3)"
echo "  6-8:  LB (base_lb1, lb1_lb2, lb2_lb3)"
echo "  9-11: RB (base_rb1, rb1_rb2, rb2_rb3)"
echo ""

# Neutral standing pose
NEUTRAL_LF1=0.0
NEUTRAL_LF2=0.785
NEUTRAL_LF3=-1.57

NEUTRAL_RF1=0.0
NEUTRAL_RF2=0.785
NEUTRAL_RF3=-1.57

NEUTRAL_LB1=0.0
NEUTRAL_LB2=0.785
NEUTRAL_LB3=-1.57

NEUTRAL_RB1=0.0
NEUTRAL_RB2=0.785
NEUTRAL_RB3=-1.57

echo "Press Enter to move to neutral pose..."
read

ros2 topic pub --once /joint_group_effort_controller/joint_trajectory trajectory_msgs/JointTrajectory "
header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names: ['base_lf1', 'lf1_lf2', 'lf2_lf3', 'base_rf1', 'rf1_rf2', 'rf2_rf3', 'base_lb1', 'lb1_lb2', 'lb2_lb3', 'base_rb1', 'rb1_rb2', 'rb2_rb3']
points:
- positions: [0.0, 0.785, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57]
  time_from_start: {sec: 1, nanosec: 0}
"

echo "Neutral pose commanded!"
echo ""
echo "Press Enter to test LF hip (joint 0) with +0.5 rad..."
read

# Test LF hip - move ONLY joint 0
ros2 topic pub --once /joint_group_effort_controller/joint_trajectory trajectory_msgs/JointTrajectory "
header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names: ['base_lf1', 'lf1_lf2', 'lf2_lf3', 'base_rf1', 'rf1_rf2', 'rf2_rf3', 'base_lb1', 'lb1_lb2', 'lb2_lb3', 'base_rb1', 'rb1_rb2', 'rb2_rb3']
points:
- positions: [0.5, 0.785, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57]
  time_from_start: {sec: 1, nanosec: 0}
"

echo "LF hip moved! Did the LEFT FRONT hip move outward?"
echo ""
echo "Press Enter to return to neutral..."
read

ros2 topic pub --once /joint_group_effort_controller/joint_trajectory trajectory_msgs/JointTrajectory "
header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names: ['base_lf1', 'lf1_lf2', 'lf2_lf3', 'base_rf1', 'rf1_rf2', 'rf2_rf3', 'base_lb1', 'lb1_lb2', 'lb2_lb3', 'base_rb1', 'rb1_rb2', 'rb2_rb3']
points:
- positions: [0.0, 0.785, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57]
  time_from_start: {sec: 1, nanosec: 0}
"

echo ""
echo "Press Enter to test LF thigh (joint 1) with +0.3 rad..."
read

# Test LF thigh - move ONLY joint 1
ros2 topic pub --once /joint_group_effort_controller/joint_trajectory trajectory_msgs/JointTrajectory "
header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names: ['base_lf1', 'lf1_lf2', 'lf2_lf3', 'base_rf1', 'rf1_rf2', 'rf2_rf3', 'base_lb1', 'lb1_lb2', 'lb2_lb3', 'base_rb1', 'rb1_rb2', 'rb2_rb3']
points:
- positions: [0.0, 1.085, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57, 0.0, 0.785, -1.57]
  time_from_start: {sec: 1, nanosec: 0}
"

echo "LF thigh moved! Did the LEFT FRONT thigh move up/forward?"
echo ""
echo "Test complete. You can add more tests following this pattern."

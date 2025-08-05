#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    
    # Declare arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/ubuntu/SpotDMouse/P2-Terrain_Challenge/sim2real/walkingmlp.pt',
        description='Path to the trained MLP model'
    )
    
    test_mode_arg = DeclareLaunchArgument(
        'test_mode',
        default_value='stationary',
        description='Test mode: stationary, walk, or teleop'
    )
    
    # Mini Pupper bringup
    bringup_cmd = Node(
        package='mini_pupper_bringup',
        executable='bringup.launch.py',
        name='mini_pupper_bringup'
    )
    
    # MLP Controller
    mlp_controller = Node(
        package='your_package_name',  # Update this
        executable='mlp_controller.py',
        name='mlp_controller',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'torque_scale': 0.3,
            'max_joint_change': 0.1,
            'control_frequency': 50.0
        }],
        output='screen'
    )
    
    # Conditional nodes based on test mode
    
    # For stationary test - just hold position
    stationary_cmd = Node(
        package='your_package_name', #What is this discretion?
        executable='stationary_test.py',
        name='stationary_test',
        condition=lambda context: LaunchConfiguration('test_mode').perform(context) == 'stationary'
    )
    
    # For walking test - send forward velocity commands
    walking_cmd = Node(
        package='your_package_name', #What is this discretion? What should the package actually be?
        executable='walking_test.py',
        name='walking_test',
        condition=lambda context: LaunchConfiguration('test_mode').perform(context) == 'walk'
    )
    
    # For teleop test - use keyboard control
    teleop_cmd = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        condition=lambda context: LaunchConfiguration('test_mode').perform(context) == 'teleop',
        prefix='gnome-terminal --'
    )
    
    return LaunchDescription([
        model_path_arg,
        test_mode_arg,
        bringup_cmd,
        TimerAction(
            period=3.0,  # Wait for bringup to complete
            actions=[mlp_controller]
        ),
        TimerAction(
            period=5.0,  # Wait for controller to initialize
            actions=[stationary_cmd, walking_cmd, teleop_cmd]
        )
    ])
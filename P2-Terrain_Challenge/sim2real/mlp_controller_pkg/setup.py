from setuptools import setup, find_packages

package_name = 'mlp_controller_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='MLP controller for Mini Pupper',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mlpcontroller = mlp_controller_pkg.mlpcontroller:main',
        ],
    },
)

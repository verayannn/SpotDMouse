from setuptools import find_packages, setup

package_name = 'mlp_controller_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='javiercw@stanford.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)

entry_points={
    'console_scripts': [
        'mlp_controller = mlp_controller_pkg.mlp_controller:main',
        'stationary_test = mlp_controller_pkg.test_scripts:run_stationary_test',
        'walking_test = mlp_controller_pkg.test_scripts:run_walking_test',
        'safety_monitor = mlp_controller_pkg.test_scripts:run_safety_monitor',
    ],
},

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'turtlesim_tf2_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yunho Lee',
    maintainer_email='dbs2911@unist.ac.kr',
    description='Launch files and resources to run the turtlesim TF2 tutorial demo.',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)

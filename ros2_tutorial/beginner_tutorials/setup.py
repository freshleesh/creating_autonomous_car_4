from setuptools import setup
import os
from glob import glob

package_name = 'beginner_tutorials'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yunho Lee',
    maintainer_email='dbs2911@unist.ac.kr',
    description='Simple ROS 2 pub/sub demo: Int32 counter',
    license='MIT',
    entry_points={
        'console_scripts': [
            'talker = beginner_tutorials.talker:main',
            'listener = beginner_tutorials.listener:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'controller'

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
    maintainer='nuc5',
    maintainer_email='jeongsangryu@gmail.com',
    description='Lateral and longitudinal controllers for F1TENTH autonomous racing',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'controller_node    = controller.controller_ros:main',
            'wall_follow_node   = controller.wallfollow:main',
            'gap_follow_node    = controller.gapfollow:main',
            'pp_node            = controller.PP:main',
            'pp_v2_node         = controller.PP_v2:main',
            'pp_v3_node         = controller.PP_v3:main',
            'pp_v4_node         = controller.PP_v4:main',
            'pp_v5_node         = controller.PP_v5:main',
            'pp_v6_node         = controller.PP_v6:main',
            'stanley_node       = controller.Stanley:main',
            'mppi_node          = controller.MPPI:main',
        ],
    },
)

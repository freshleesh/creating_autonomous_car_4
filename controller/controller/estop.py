import math
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time

class EStop:
    def __init__(self, node):
        p = lambda name: node.get_parameter(name).value
        self._logger     = node.get_logger()
        self.stop_start = None

    def should_stop(self, scan: LaserScan, odom: Odometry, cmd=None):
        if cmd is None:
            cmd = AckermannDriveStamped()

        # if self.stop_start is not None:
        #     while (time.time() - self.stop_start < 2):
        #         self._logger.warn(f'too close !!')
        #     self.stop_start = None
                

        stop_sign = False


        if odom is not None:
            if scan is not None:


                current_vel_x = odom.twist.twist.linear.x
                current_vel = [current_vel_x, 0]


                start = scan.angle_min
                # self._logger.warn(f'tts: {start}')

                for i, range in enumerate(scan.ranges):
                    if range <= 0.0 or not math.isfinite(range):
                        continue

                    current_angle = start + scan.angle_increment * i

                    # if abs(math.degrees(current_angle) - 135) > 50: 
                    #     pass 

                    point_x = range * math.cos(current_angle)
                    point_y = range * math.sin(current_angle)
                    point = np.array([point_x, point_y])


                    ttc = np.inf
                    if abs(point_y) <0.1 and point_x >0 and current_vel_x > 0:
                        ttc = point_x / current_vel_x

                    # point_wise_vel = current_vel @ point / range


                    # ttc = range / point_wise_vel
                    # # self._logger.warn(f'tts: {tts}')

                    if abs(point_y) < 0.14 and point_x <0.2 and point_x > 0:
                        stop_sign = True
                        self._logger.warn(f'too close !!')

                        self.stop_start = time.time()

                    # if point_wise_vel <= 0.0:
                    #     continue

                    if ttc < 0.1:
                        self._logger.warn(f'ttc: {ttc}')
                        stop_sign = True

            else:
                stop_sign = True

        else:
            stop_sign = True



        # TODO: Implement an emergency stop (E-Stop) using:
        #   - 2D LiDAR scan data
        #   - Wheel odometry data from the VESC
        #   - TTC (Time-to-Collision) based logic
        #
        # You may modify `cmd` (the original control command) in this function.
        #
        # Useful information:
        #   - scan.ranges                  : distance array [m] for each LiDAR beam
        #   - scan.angle_min               : angle of the first beam [rad]
        #   - scan.angle_max               : angle of the last beam [rad]
        #   - scan.angle_increment         : angular step between beams [rad]
        #   - odom.twist.twist.linear.x    : vehicle forward speed [m/s]
        #   - odom.twist.twist.angular.z   : vehicle yaw rate [rad/s]

        if stop_sign:
            cmd.drive.speed = 0.0
            cmd.drive.steering_angle = 0.0



            return cmd

        return cmd

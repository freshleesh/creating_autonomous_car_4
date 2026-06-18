#!/usr/bin/env python3
"""VESC 조향 캘리브레이션 - 일정 ackermann 명령 출력기.

고정 조향각 + 고정 속도를 /vesc/high_level/ackermann_cmd 로 계속 발행해서
차를 원 돌게 만든다. 반경은 Cartographer 로 직접 측정.

명령 경로(실차):
  이 스크립트 → /vesc/high_level/ackermann_cmd → simple_mux → /vesc/ackermann_cmd
              → ackermann_to_vesc → VESC
  ※ simple_mux 는 50Hz 로 항상 발행하며, 실차에서 current_host=None 이면 zero(정지)를
    내보낸다. 따라서 조이스틱 'auto' 버튼(버튼5)을 한 번 눌러 autodrive 로 바꿔야
    우리 명령이 통과한다. (humandrive=버튼4)
  ※ 명령은 1초 안에 갱신돼야 mux 가 fresh 로 인정 → 계속 재발행한다.

캘리브레이션 원리(offset 0.634 는 직진으로 검증됨):
  ackermann_to_vesc 는 servo = gain_old * steer + offset 로 변환한다(gain_old=-1.2).
  조향각 steer 를 명령하고 실제 반경 R 을 재면:
      delta_actual = atan(wheelbase / R)
  같은 servo 가 delta_actual 을 만들어야 하므로:
      gain_true = gain_old * steer / delta_actual
  즉 차가 명령보다 덜 꺾이면(delta_actual < steer) |gain| 을 키워야 한다.
  좌/우 여러 steer 에서 측정해 평균/대칭성 확인 권장.

사용 전:
        ros2 launch stack_master low_level.launch.xml sim:=false
  (컨트롤러 mppi/pp 는 끌 것. 넓고 평평한 바닥, 저속 권장)

실행:
  python3 steer_calib.py --steer 0.25 --speed 1.0
  그 다음 조이스틱 auto 버튼(버튼5)을 눌러 차를 출발시킨다.
  Ctrl-C → 0 명령 발행 후 종료(정지).
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


HIGH_LEVEL_TOPIC = '/vesc/high_level/ackermann_cmd'
GAIN_OLD = -1.2      # vehicle_config: steering_angle_to_servo_gain (현재값)
OFFSET = 0.634       # vehicle_config: steering_angle_to_servo_offset (검증됨)


class AckermannHold(Node):
    def __init__(self, steer, speed, rate_hz=50.0):
        super().__init__('steer_calib')
        self.steer = float(steer)
        self.speed = float(speed)
        self.pub = self.create_publisher(AckermannDriveStamped, HIGH_LEVEL_TOPIC, 10)
        self.create_timer(1.0 / rate_hz, self._tick)

    def _tick(self):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.drive.steering_angle = self.steer
        m.drive.speed = self.speed
        self.pub.publish(m)

    def stop(self):
        self.steer = 0.0
        self.speed = 0.0
        self._tick()


def main():
    ap = argparse.ArgumentParser(description="VESC 조향 캘리 ackermann 명령 출력기")
    ap.add_argument('--steer', type=float, required=True, help='고정 조향각 [rad] (예: 0.25)')
    ap.add_argument('--speed', type=float, default=1.0, help='속도 [m/s] (기본 1.0)')
    args, _ = ap.parse_known_args()

    servo_implied = GAIN_OLD * args.steer + OFFSET
    rclpy.init()
    node = AckermannHold(args.steer, args.speed)
    print(f"[steer_calib] steer={args.steer:+.3f} rad  speed={args.speed:.2f} m/s")
    print(f"  현재 gain({GAIN_OLD})으로 환산되는 servo ≈ {servo_implied:.3f}")
    if not (0.0 <= servo_implied <= 1.0):
        print(f"  ⚠ servo 가 [0,1] 밖 → 포화됨! steer 크기를 줄이세요.")
    print("  → 조이스틱 auto 버튼(버튼5)을 눌러 출발. Cartographer 로 반경 R 측정.")
    print("  측정 후:  delta_actual = atan(0.33/R),  gain_true = gain_old*steer/delta_actual")
    print("  Ctrl-C 로 정지.\n")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[steer_calib] 정지합니다.")
    finally:
        node.stop()
        time.sleep(0.2)
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

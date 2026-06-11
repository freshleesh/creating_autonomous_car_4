-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- /* Author: Darby Lim */

-- ===========================================================================
--  카토그래퍼 LOCALIZATION 튜닝 가이드 (실차용 / sim은 localization_2d_sim.lua)
-- ===========================================================================
--  동작 구조 (2단계):
--    [1] Local SLAM  : 매 스캔을 현재 서브맵에 정합 → 부드러운 고주파 포즈
--        (online correlative로 초기값 탐색 → ceres로 정밀 정합)
--    [2] Pose Graph  : 스캔을 pbstream의 옛 서브맵과 재정합(제약)하고
--        그래프 최적화로 누적 드리프트를 보정 → 이때 포즈가 "점프"함
--    즉 [1]은 부드럽지만 드리프트하고, [2]는 정확하지만 점프한다.
--    튜닝은 결국 이 둘 사이 균형 잡기다.
--
--  증상별 처방 (테스트 → 반응 → 조치):
--
--  ▶ 테스트 A: 최고 속도로 코너 진입 — RViz에서 scan이 벽에서 미끄러지듯 밀리나?
--    → 스캔 모션 왜곡 or 매칭 초기값 이탈.
--      num_subdivisions_per_laser_scan ↑ (10→20),
--      ceres rotation_weight ↓ (스캔이 회전을 더 자유롭게 보정).
--      그래도 밀리면 num_accumulated_range_data ↓ (갱신 주기 단축).
--
--  ▶ 테스트 B: 직선 정속 주행 — /tracked_pose가 자잘하게 떨리나(지터)?
--    → 매 스캔 정합이 튀는 것.
--      ceres rotation_weight ↑ (이전 자세 유지),
--      optimize_every_n_nodes ↑ (그래프 보정 점프 빈도 감소).
--      ※ EKF가 한 번 걸러주므로 /car_state/odom 기준으로 판단할 것.
--
--  ▶ 테스트 C: 5바퀴 연속 주행 — 랩이 쌓일수록 포즈가 서서히 벽 쪽으로 틀어지나?
--    → 드리프트 보정(제약)이 부족.
--      constraint_builder.sampling_ratio ↑ (0.1→0.3),
--      constraint_builder.min_score ↓ (0.7→0.6, 제약을 더 잘 받아들임),
--      optimize_every_n_nodes ↓ (보정 자주).
--
--  ▶ 테스트 D: 벽 모양이 비슷한 구간(평행 복도/반복 패턴) 통과 — 포즈가 순간
--    엉뚱한 곳으로 텔레포트하나?
--    → 가짜 제약이 채택된 것. 테스트 C와 반대 방향:
--      constraint_builder.min_score ↑ (0.7→0.75),
--      linear/angular_search_window ↓ (탐색 범위를 좁혀 오매칭 차단),
--      global_sampling_ratio는 0 유지.
--
--  ▶ 테스트 E: 시작 위치를 일부러 틀리게 주거나, 주행 중 차를 들어 옮김(키드냅)
--    — 스스로 복구하나?
--    → 현재 global_sampling_ratio = 0.0 이라 복구 불가(의도된 설정: 레이스 중
--      텔레포트 방지). 복구가 필요하면 0.003 정도로 켜고
--      global_localization_min_score로 채택 문턱 조절. 단, 테스트 D 위험 증가.
--
--  ▶ 테스트 F: CPU 점유율 확인 — cartographer_node가 코어를 다 먹나?
--    → 비용 순서대로: constraint_builder.sampling_ratio ↓,
--      search_window ↓, num_subdivisions ↓.
--      use_online_correlative_scan_matching이 가장 비싸지만 고속 주행 안정성의
--      핵심이므로 끄는 건 최후 수단.
--
--  ※ min_score(0.7)와 sampling_ratio(0.1)는 C↔D 트레이드오프의 현재 균형점.
--    한쪽을 만지면 반드시 반대쪽 테스트도 다시 돌릴 것.
-- ===========================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",                       -- 전역 고정 프레임
  tracking_frame = "base_link",            -- IMU/포즈 추정 기준 프레임 (IMU가 base_link로 remap되어 들어옴)
  published_frame = "base_link",           -- map → published_frame TF를 발행. odom TF가 없으므로 base_link 직결
  odom_frame = "odom",                     -- provide_odom_frame=false라 실제로는 미사용
  provide_odom_frame = false,              -- true면 map→odom→base_link 분리 발행. 우리는 EKF가 따로 있어 불필요
  publish_frame_projected_to_2d = true,    -- 포즈를 2D 평면에 투영 (roll/pitch/z 제거)
  use_odometry = false,                    -- /odom(=/vesc/odom) 토픽을 매칭 초기값으로 사용 안 함.
                                           -- VESC 오도메트리가 미끄러짐에 취약해서 끔. 스캔매칭 단독.
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 10,    -- 스캔 1바퀴를 10조각으로 쪼개 조각마다 다른 시각으로 정합
                                           -- → 주행 중 스캔 모션 왜곡 보정. 차가 빠르거나 라이다가
                                           -- 느릴수록 ↑ (테스트 A). CPU 비용도 같이 ↑
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,      -- TF 조회 대기 한도 [s]. TF 타임아웃 에러 뜨면 ↑
  submap_publish_period_sec = 0.3,         -- 서브맵 시각화 발행 주기 (성능 영향 미미)
  trajectory_publish_period_sec = 30e-3,   -- 궤적 시각화 발행 주기
  rangefinder_sampling_ratio = 1.,         -- 스캔 사용 비율. 1.0 = 전부 사용 (localization은 줄이지 말 것)
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
  publish_tracked_pose = true,             -- /tracked_pose 발행 → pose_to_odom → EKF → /car_state/odom
  pose_publish_period_sec = 1e-2,          -- /tracked_pose 100 Hz. 제어 50 Hz보다 빠르게 유지할 것
  publish_to_tf = true,                    -- 카토그래퍼가 직접 map→base_link TF 발행 (sim/real 동일)
}

MAP_BUILDER.use_trajectory_builder_2d = true -- for 2d slam or localization
MAP_BUILDER.num_background_threads = 6       -- 백그라운드(제약 탐색) 스레드 수. CPU 코어 수에 맞춰 조절

-- ---------------------------------------------------------------------------
-- [1] Local SLAM — 매 스캔을 서브맵에 정합 (고주파 포즈의 품질 결정)
-- ---------------------------------------------------------------------------
TRAJECTORY_BUILDER_2D.min_range = 0.12   -- 이보다 가까운 리턴 버림 (차체/마운트 반사 제거)
TRAJECTORY_BUILDER_2D.max_range = 10.    -- 이보다 먼 리턴 버림. 트랙이 넓어 먼 벽이 안 보이면 ↑,
                                         -- 먼 거리 노이즈로 매칭이 흔들리면 ↓
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 3.  -- max_range 밖 리턴을 이 길이만큼 "빈 공간"으로 취급
TRAJECTORY_BUILDER_2D.use_imu_data = true           -- IMU(자이로)로 회전 초기 추정 (실차 전용. sim 파일은 false)
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
                                         -- ceres 정밀 정합 전에 brute-force 탐색으로 초기값을 찾음.
                                         -- 급회전/미끄러짐에서 매칭 이탈을 막는 핵심 옵션. CPU 비쌈 (테스트 F)
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.1)
                                         -- 이 각도 미만 움직임이면 스캔을 버림. 0.1°로 거의 모든 스캔 사용.
                                         -- 정차 중 포즈가 떠다니면 ↑ (스캔 덜 씀), 반응이 둔하면 ↓
-- TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 5 --0.01
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 0.1 --25
                                         -- ceres가 "초기 회전 앵커"에서 벗어나는 걸 벌점주는 가중치.
                                         -- ※ 앵커의 출처 주의: correlative ON(현재)이면 IMU 예측이 아니라
                                         --   correlative 탐색 결과가 앵커다. IMU 직접 신뢰 ≠ 이 값.
                                         -- ↓ = 스캔이 회전을 자유롭게 보정 (코너에서 밀릴 때, 테스트 A)
                                         -- ↑ = 앵커 고수 (직선 회전 지터엔 ↑가 답이지만, correlative가
                                         --   틀린 회전을 잡은 경우 그걸 굳히는 부작용도 있음, 테스트 B)
-- TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.)
                                         -- (기본값 20°) correlative가 IMU 예측 주변을 탐색하는 회전 범위.
                                         -- 코너에서 yaw가 통째로 틀어지는 문제는 이걸 8~10°로 줄여
                                         -- IMU 예측 근처로 탐색을 묶는 게 직접적인 처방 (테스트 A 변형:
                                         -- 코너 후 ~10° 기울어짐 고착). 너무 줄이면 자이로 바이어스/슬립
                                         -- 순간의 실제 회전을 못 따라감
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 10
                                         -- 조각(subdivision) 10개 = 스캔 1바퀴를 모아 한 번 정합.
                                         -- num_subdivisions_per_laser_scan과 짝으로 움직일 것
                                         -- (subdivisions를 20으로 올리면 여기도 20으로).
                                         -- ↓ = 갱신 빠름/입력 빈약, ↑ = 입력 풍부/갱신 지연

-- ---------------------------------------------------------------------------
-- [2] Pose Graph — pbstream 옛 맵과 재정합해 드리프트 보정 (점프의 근원)
-- ---------------------------------------------------------------------------
POSE_GRAPH.constraint_builder.min_score = 0.65
                                         -- 제약(스캔↔옛 서브맵 매칭) 채택 최소 점수 [0~1].
                                         -- ↑ = 가짜 제약/텔레포트 방지 (테스트 D)
                                         -- ↓ = 보정 잘 받아들임, 드리프트 누적 방지 (테스트 C)
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60
                                         -- 전역 위치인식(초기 포즈 탐색) 채택 점수.
                                         -- global_sampling_ratio = 0 인 현재는 사실상 미사용

TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,               -- localization 모드의 핵심: 새 서브맵을 3개만 유지하고 버림
                                         -- → 맵을 갱신하지 않고 pbstream 맵에만 정합. 건드릴 일 없음
}
POSE_GRAPH.optimize_every_n_nodes = 3    -- 노드 3개마다 그래프 최적화. 매핑에 비해 낮춰줘야함.
                                         -- ↓ = 보정 자주(작은 점프 여러 번, 테스트 C)
                                         -- ↑ = 점프 드묾(대신 한 번에 크게 점프, 테스트 B)
POSE_GRAPH.constraint_builder.sampling_ratio = 0.1
                                         -- 제약 후보를 탐색할 노드 비율. ↑ = 보정 기회 많음/CPU 비쌈 (테스트 C↔F)

POSE_GRAPH.global_sampling_ratio = 0.00 -- 0.003 -- 전역 매칭 샘플링 비율.
                                         -- 0 = 전역 재인식 OFF: 초기 포즈는 set_initial_pose_node에 전적으로
                                         -- 의존하고, 추적을 잃으면 자력 복구 불가. 대신 레이스 중 비슷한 벽
                                         -- 구간으로 텔레포트할 위험이 없음 (테스트 D/E 트레이드오프)
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 3. -- 제약 탐색 병진 범위 [m]
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(30.) -- 제약 탐색 회전 범위
                                         -- 두 window: 드리프트가 이 범위를 넘으면 제약을 못 찾음 (↑ 필요, 테스트 C).
                                         -- 넓을수록 오매칭 위험 + CPU ↑ (테스트 D/F)

-- Localization 전용으로 설정
-- POSE_GRAPH.global_constraint_search_after_n_seconds = 10. -- 전역 제약 조건 검색 주기

return options

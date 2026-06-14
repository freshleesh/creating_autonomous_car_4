# KISS-ICP Localization 튜닝 가이드

scan-to-map ICP localizer (`scan_matching_localization/icp_localizer_node`)의 설정
튜닝 가이드입니다. 실제 KISS-ICP 패키지가 아니라 KISS-ICP의 두 가지 아이디어
(scan voxel downsampling + adaptive correspondence threshold)를 가져온 2D LiDAR용
경량 ICP 입니다.

- 노드 소스: `slam/scan_matching_localization/src/icp_localizer_node.cpp`
- 베이스 클래스: `slam/scan_matching_localization/include/scan_matching_localization/scan_matching_localizer.hpp`
- 설정 파일: `stack_master/config/icp.yaml`
- EKF 융합 설정: `stack_master/config/ekf_icp.yaml`
- 실행: `ros2 launch stack_master middle_level.launch.xml localization:=kiss-icp`

---

## 1. 파이프라인 한눈에 보기

스캔 1장 당 처리 흐름 (`scanCallback`):

```
/scan ─► base_link 좌표 변환 + range 필터(min/max_range)
            │
            ▼
   모션 예측 (predict)
     · 병진(translation): /vesc/odom 델타
     · 회전(heading)    : use_imu=true 면 IMU gyro 적분, 아니면 odom yaw
            │  predicted pose = ICP 초기값(seed)
            ▼
   ICP 정합 (align)  ── scan voxel 다운샘플 → map grid 최근접점 매칭 → Umeyama
            │           correspondence gate를 loose→tight 로 점점 조임
            ▼
   /icp/pose/odom (map 프레임 pose) ─► EKF ─► /car_state/odom
                                    └─► map→base_link TF (실차만)
```

핵심: **ICP는 모션 예측(seed)을 보정만** 합니다. seed가 크게 틀리면(odom/IMU 부정확,
빠른 회전) gate를 벗어나 정합이 깨집니다. 그래서 모션 모델과 ICP gate를 같이 봐야 합니다.

---

## 2. 파라미터 레퍼런스 & 튜닝 방향

### 2.1 ICP 정합 (`icp.yaml`)

| 파라미터 | 기본값 | 의미 | 키우면 | 줄이면 |
|---|---|---|---|---|
| `voxel_size` | 0.05 | 스캔 다운샘플 leaf [m] | 빠름, 거칠어짐 | 정밀, 느림 |
| `nn_cell_size` | 0.50 | map 최근접 grid 셀 [m] | 메모리↓, 쿼리 느려질 수 있음 | 쿼리 빠름, 메모리↑ |
| `max_correspondence_distance` | 2.0 | 첫 iteration의 loose gate [m] | seed 오차 더 허용(발산 위험↑) | 엄격(seed 나쁘면 매칭 실패) |
| `min_correspondence_distance` | 0.15 | 수렴 후 tight gate [m] | 정합 느슨 | 정밀(노이즈에 민감) |
| `correspondence_decay` | 0.85 | iteration 마다 gate에 곱하는 비율 | 천천히 조임(느슨/안정) | 빨리 조임(빠르지만 조기수렴) |
| `min_correspondence_pairs` | 10 | 정합에 필요한 최소 대응쌍 수 | 빈약한 스캔 거부(안전) | 적은 점에도 강행(위험) |
| `max_iterations` | 20 | ICP 최대 반복 | 정밀, 느림 | 빠름, 덜 수렴 |
| `transform_epsilon` | 0.001 | 병진 수렴 임계 [m] | 일찍 종료 | 더 반복 |
| `rotation_epsilon` | 0.0001 | 회전 수렴 임계 [rad] | 일찍 종료 | 더 반복 |

`max_correspondence_distance`가 `decay`로 줄어들어 `min_correspondence_distance`에
수렴합니다. 대략 `min ≈ max * decay^N` 가 되는 N 회 안에 gate가 닫히므로, decay와 두
gate를 함께 조정하세요. 기본값 기준 2.0 → 0.15 는 약 16 iteration 정도에 도달합니다.

### 2.2 모션 모델 / IMU (`icp.yaml` + launch)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `use_imu` | launch에서 설정 (실차 true / sim false) | heading을 IMU gyro 적분으로 예측. false면 odom yaw 사용 |
| `imu_yaw_scale` | 1.0 | gyro z축 부호 반대면 `-1.0` |

`use_imu`, `publish_tf`, `seed_pose_from_tf`는 `icp.yaml`이 아니라
`middle_level.launch.xml`에서 sim/실차에 따라 자동 설정됩니다(180~189행). 실차는
IMU heading + map→base_link TF publish, sim은 gym_bridge가 TF를 소유.

### 2.3 스캔 필터 (`icp.yaml`)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `min_range` / `max_range` | 0.1 / 15.0 | 사용할 LiDAR 거리 범위 [m] |
| `occupancy_threshold` | 65 | OccupancyGrid에서 "점유"로 볼 값 (0~100) |
| `min_scan_points` | 30 | 정합을 시도할 최소 스캔 점 수 (미만이면 예측값만 사용) |

`max_range`는 맵 크기에 맞추세요. 트랙보다 크게 잡으면 멀리 있는(맵에 없는) 벽/관중을
잡아 outlier가 됩니다. `occupancy_threshold`를 낮추면 맵 점이 많아져(불확실 셀 포함)
정합점이 늘지만 노이즈도 늘어납니다.

---

## 3. 증상별 처방

### 정합이 자주 튀거나 발산 (pose가 갑자기 점프)
1. `max_correspondence_distance` ↓ (예: 2.0 → 1.0) — outlier 매칭 차단.
2. `min_correspondence_pairs` ↑ (예: 10 → 20) — 빈약한 정합 거부.
3. 모션 예측 점검: `imu_yaw_scale` 부호, `/vesc/odom` 품질. seed가 나쁘면 ICP가
   못 따라잡습니다.
4. `max_range` ↓ 로 맵 밖 물체 컷.

### 빠른 회전(코너)에서 heading이 밀림
1. `use_imu=true` 인지 확인(실차 기본). IMU가 없으면 odom yaw라 코너에서 지연.
2. `max_correspondence_distance` ↑ 살짝 — 회전 중 seed 오차를 더 허용.
3. `imu_yaw_scale` 부호 검증: 가만히 두고 차를 좌회전시켰을 때 yaw가 +로 증가하는지.

### 정합이 둔하다 / 위치가 어림잡힌 느낌
1. `voxel_size` ↓ (0.05 → 0.03) — 스캔 해상도↑.
2. `min_correspondence_distance` ↓ (0.15 → 0.10) — 최종 gate 정밀화.
3. `max_iterations` ↑, `transform_epsilon`/`rotation_epsilon` ↓ — 더 수렴.

### CPU 부담 / `align` 시간이 김
`debug_timing: true` 로그(`align: X ms`, 1Hz throttle)를 보며:
1. `voxel_size` ↑ — 스캔 점 수가 가장 큰 비용 요인.
2. `max_iterations` ↓.
3. `max_range` ↓ — 처리할 점 감소.
4. `nn_cell_size`를 `min_correspondence_distance` 근처로 — grid 쿼리 효율.

### 시작 시 위치를 못 잡음
- 실차: RViz "2D Pose Estimate"(`/initialpose`)로 초기 pose 지정.
- sim: `seed_pose_from_tf=true`(launch 자동)로 gym_bridge TF에서 시작 pose 가져옴.
- `initial_x/y/yaw` 파라미터로 기본 시작 pose 지정 가능(현재 icp.yaml엔 없음, 기본 0).

---

## 4. 튜닝 절차 (권장 순서)

1. **모션 예측부터.** ICP를 신뢰하기 전에 `/vesc/odom`과 IMU가 정상인지 확인.
   `imu_yaw_scale` 부호를 먼저 검증. seed가 좋아야 ICP gate를 좁게 쓸 수 있음.
2. **정지 상태 정합.** 차를 세우고 pose가 안정적인지(떨림 없는지) 확인.
   떨리면 `voxel_size`/`min_correspondence_distance`/`max_iterations` 조정.
3. **저속 주행.** gate(`max_correspondence_distance`)와 `min_correspondence_pairs`로
   강건성 확보. 튀지 않는 최소 gate를 찾음.
4. **고속/코너.** heading 지연·발산을 보며 IMU 활용과 gate를 미세조정.
5. **CPU 예산.** `debug_timing`으로 `align` 시간을 보고 voxel/iteration으로 맞춤.

한 번에 하나씩 바꾸고 `debug_timing` 로그와 RViz의 map→base_link, `/icp/pose/odom`을
같이 보세요.

---

## 5. EKF와의 상호작용 (`ekf_icp.yaml`)

ICP pose(`/icp/pose/odom`)는 단독으로 쓰이지 않고 robot_localization EKF로 들어가
`/vesc/odom`의 twist(vx, vyaw)와 융합되어 `/car_state/odom`이 됩니다.

- ICP가 가끔 튀어도 EKF가 어느 정도 평활화하지만, **공분산**이 신뢰도를 좌우합니다.
- ICP pose 공분산은 노드에서 하드코딩 (`publishResult`, scan_matching_localizer.hpp:357):
  x,y = 0.025, yaw = 0.05. ICP를 더/덜 신뢰시키려면 이 값을 조정(코드 수정 필요).
- `odom1_config`가 ICP에서 x, y, yaw(절대 pose)를, `odom0_config`가 odom에서
  vx, vyaw(twist)를 사용. 둘의 역할이 겹치지 않게 설정되어 있음.

ICP가 안정적인데도 최종 `/car_state/odom`이 느리게/과하게 반응하면 EKF 쪽
(공분산, `frequency`, `sensor_timeout`)을 보세요.

---

## 6. 검증 체크리스트

- [ ] RViz에서 스캔이 맵 벽과 잘 겹치는가 (map→base_link TF 기준).
- [ ] `align: X ms` 로그가 scan 주기(예: 40Hz=25ms) 안에 드는가.
- [ ] 코너/급가속에서 pose 점프가 없는가.
- [ ] `/icp/pose/odom` 과 `/car_state/odom` 이 발산 없이 일치하는가.
- [ ] 정지 시 pose 떨림이 수 mm 이내인가.

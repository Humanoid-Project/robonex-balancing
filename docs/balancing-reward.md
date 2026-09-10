# Balancing reward — 2026-09-06 (v3.1)

상태: 구현 및 CPU 합성 테스트 완료. Isaac 학습으로 검증할 설계. 아래 가중치·허용 오차·규제 상한은 튜닝 제안값이며 측정된 최적값이 아니다.

## Codex 재검증 — 2026-09-06

Claude 대화 `92d29816-47e7-47ed-8f00-00b76dc02657`의 수정 이후 현재 파일로 재실행했다. 기존 `2026-09-06_04-26-18` 체크포인트를 사용했고, Isaac은 `--eval-baseline`, MuJoCo는 무외란 ONNX 폐루프로 각 14초 평가했다.

| 검증 | 재측정 결과 |
|---|---|
| 실제 Isaac 관측 700개를 ONNX에 입력 | iter1500·4999 모두 PyTorch action과 최대 절대 오차 `4.7684e-7` |
| MuJoCo 각속도 링크 좌표계 | 임의 root 회전 100개에서 최대 오차 `1.08e-7 rad/s` |
| 링크별 질량 / 주관성값 | 최대 차이 `1.98e-7 kg` / `8.14e-9 kg m²`, 총질량 `20.51391 kg` |
| iter1500 최종 몸통 tilt | Isaac `0.602°`, MuJoCo 현재 접촉 `0.389°`, 이전 접촉 설정 `1.497°` |
| iter1500 MuJoCo 최종 발 간격 | 이전 접촉 `0.25511 m` → 현재 `0.29353 m`; Isaac 10초 측정 `0.31958 m` |
| iter4999 최종 몸통 tilt | Isaac `0.470°`, MuJoCo `2.825°` |
| iter4999 MuJoCo 최종 발 간격 | `0.26989 m` |
| 실제 배포 CLI iter1500 | 14초 완료, 낙상 없음, 관절 한계 초과 0, runner clip 0; target clip 약 `0.012%` |

접촉 수정은 유효하지만 동일 궤적은 아니다. 잔여 발 간격 오차를 실측 근거 없이 정상 허용오차로 승인하지 않는다. ONNX/관측 경로의 일치는 모델 동역학이나 실물 전이의 일치와 다르다. 모델 질량·관성 일치 역시 접촉·수동관절 표현·적분기까지 동일함을 뜻하지 않는다. iter1500은 비교 기준이며 실물 배포 승인이 아니다.

v3.1 CPU 테스트 36개 통과. `2026-09-06_14-01-49_codex_v31_integration`에서 1024환경·3 iteration 완료, 12개 running 항 양수, terminating 음수, invalid_state 0, 신규 4개 로그 출력 확인. 3 iteration은 실행 검증일 뿐 수렴·외란 회복 검증은 아니다. 200 iteration 새 학습 후 별도 deterministic/외란 평가가 필요하다.

## v1에서 무엇이 바뀌었고 왜인가

v1 보상은 `2026-09-06_04-26-18` 5000-iteration run에서 **포화**했다. `Episode_Reward/*`는 `episode_sum / max_episode_length_s`이므로 각 항의 상한은 곧 가중치다. 그 기준으로 측정한 달성률:

| 항 | it 500 | it 1500 | it 2500 | it 4000 | it 4999 |
|---|---:|---:|---:|---:|---:|
| upright | 96.9% | 97.7% | 98.6% | 98.9% | 93.7% |
| height | 98.5% | 98.5% | 99.0% | 99.5% | 94.5% |
| linear_stillness | 70.0% | 81.5% | 86.0% | 88.0% | 83.0% |
| angular_stillness | 44.0% | 56.0% | 63.0% | 66.0% | 64.0% |
| stance | 83.3% | 90.7% | 91.3% | 94.7% | 90.7% |
| **합계** | **84.5%** | 89.4% | 91.5% | **92.9%** | **88.3%** |

`upright`는 iteration 500에 이미 96.9%였다. 반감 오차 10°가 정적 균형에 너무 관대해, 그 이후 기울기가 `0.59° → 0.85°`로 나빠져도 보상이 반응하지 않았다.

보상이 보지 못한 것을 `--eval-baseline` deterministic 평가로 측정했다.

| 체크포인트 | 발 간격 | `r_hip_yaw` | `l_hip_yaw` | `r_hip_roll` 정적 토크 |
|---|---:|---:|---:|---:|
| 1000 | 31.50 cm | +5.56° | +2.80° | 2.32 N·m (12%) |
| 1500 | 31.96 cm | +11.00° | −2.43° | 5.17 N·m (26%) |
| 4999 | 31.61 cm | **+27.65°** | **−8.99°** | 4.25 N·m (21%) |

발 간격은 목표 `31.94 cm`에 계속 붙어 있는데 발이 27.6°까지 돌아간다. `stance`가 발바닥 기준"점"만 보기 때문에, 다리를 비틀고 다른 관절로 보정하면 점수가 같다. 정적 토크도 `torque` 항이 최대 보상의 0.03%밖에 벌하지 않아 사실상 무료였다.

v2는 Gaussian 골격과 양수 가중치 합 1.0을 유지하고, 위 세 구멍(자세 민감도, 자세 자유도, 정적 토크)을 닫는다.

## 보상

`source/robonex_balancing/robonex_balancing/tasks/manager_based/robonex_balancing/robonex_balancing_env_cfg.py`의 `RewardsCfg`가 보상 설정의 원본이다. Isaac Lab 템플릿대로 모든 `*Cfg` configclass는 이 한 파일에 있고, `mdp/`에는 함수만 둔다.

`G(e; t) = exp(-ln(2) * sum((e / t)^2))`. 각 오차가 단독으로 `t`에 도달하면 보상은 0.5다. 목표항 가중치 합은 1이며, Isaac reward manager가 매 정책 스텝의 보상률에 `dt`를 곱한다. 시간 제한 종료는 실패 비용에서 제외한다.

| 항목 | 가중치 | v1 | 기준 |
|---|---:|---:|---|
| upright | 0.22 | 0.35 | 전체 몸통 기울기, **반감 오차 3°**; 뒤집힘은 π |
| height | 0.14 | 0.20 | `[1.0510, 1.0710] m` 무오차 구간, 구간 밖 반감 오차 0.03 m |
| linear_stillness | 0.14 | 0.20 | 세계 XYZ 속도, 반감 오차 `(0.15, 0.15, 0.10) m/s` |
| angular_stillness | 0.09 | 0.10 | 몸통 XYZ 각속도의 축별 Gaussian 평균, **반감 오차 `(0.20, 0.20, 0.15) rad/s`** |
| stance | 0.09 | 0.15 | yaw 정렬 수평 프레임에서 L−R 발바닥 중심 XY; 반감 오차 `(0.03, 0.04) m` |
| **foot_yaw** | 0.09 | — | 각 발의 몸통 대비 yaw, 축별 Gaussian 평균, 반감 오차 6° |
| **pelvis** | 0.08 | — | yaw 정렬 프레임에서 `root − 양발 중점` XY, 반감 오차 `(0.05, 0.02) m` |
| **symmetry** | 0.05 | — | 좌우 대응 6쌍의 `q_L + q_R`, 축별 Gaussian 평균, 반감 오차 3° |
| **torque** | 0.04 | −0.01 | 12개 모터 `추정 토크 / 정격`의 축별 Gaussian 평균, 반감 오차 0.30 |
| **target_delta** | 0.02 | −0.01 | 관절 limit 적용 전 목표각 변화의 축별 Gaussian 평균, 반감 오차 0.03 rad |
| **foot_grip** | 0.02 | −0.02 (`foot_slip`) | 접촉한 발의 바닥 중심 XY 속도 Gaussian, 반감 0.1 m/s. 뜬 발은 0.0; 양발 평균 |
| **joint_limit** | 0.02 | −0.02 (`action_excess`) | soft limit(가동범위의 90%)을 넘은 관절각 초과분의 합(AND) Gaussian, 반감 0.05 rad |
| terminating | −1.0 | −1.0 | 함수에서 `/dt` 보정하여 실패 순간 정확히 −1점 |

## action 매핑 (2026-09-08, robonex-common 0.3.0)

`target = 서있는 자세 + action_scale × a`. 이전에는 `target = 가동범위 중점 + (가동범위/2) × a`였다.

| | 0.2.0 | 0.3.0 |
|---|---|---|
| `a = 0`이 명령하는 것 | 가동범위의 중점 | 서있는 자세 |
| `\|a\| = 1`이 덮는 범위 | 관절 전체 | 관절의 7~25% |
| 관절 한계에 닿는 `a` | 항상 1.00 | 1.78 ~ 13.73 (관절별) |
| `clip_actions` | 3.0 | 14.0 (`RUNNER_ACTION_CLIP`) |

`action_scale`은 `robonex_common.limits.ACTION_SCALE_RAD`이며 hip_yaw 0.12, hip_pitch/roll/knee 0.25,
ankle 0.15 rad이다. hip_yaw만 작은 이유는 RS02(peak 17 N·m)가 kp=40에서 24.4°에 토크 포화하기 때문이고,
ankle이 0.15인 이유는 가동범위가 ±35° 수준이라 0.25면 `|a|=1`이 범위의 43%를 덮어 정적 균형의 주 제어축
해상도가 거칠어지기 때문이다.

`RUNNER_ACTION_CLIP = 14.0`은 임의값이 아니라 `action_limit_reach()`의 최대 크기(hip_yaw 13.73)를 올림한
값이다. 이보다 작으면 도달 불가능한 관절 한계가 생긴다. 테스트가 이 부등식을 검사한다.

바뀌지 않은 것: `ACTION_CLIPS`(목표각 hard clip, 한계에서 0.01 rad 안쪽), 42-D observation, kp=40/kd=2,
`sim.dt=1/250` + `decimation=5`. 이 매핑은 `robonex-walking`도 같은 함수로 공유한다.

측정(2026-09-08, 30 iteration, 256 env): `ActionLimitUpper/l_hip_roll_joint = 0.0000`
(0.2.0에서 같은 관절이 13%였다), `Balance/joint_limit_excess_max_deg = 0.0033`,
`Balance/target_delta_rms_rad = 0.0908`. 학습 성공의 증거는 아니고 매핑이 의도대로 동작한다는 확인이다.

## running penalty를 전부 제거한 이유 (v3)

Gaussian 커널을 쓴다고 음수 보상이 금지되는 것은 아니다. 여기서는 설계 선택으로 양수 running 보상 12개와 실패 비용 1개를 사용한다. 양수화가 모든 조기 종료 유인이나 action 포화를 해결한다는 보장은 없다. 특히 Gaussian regularizer로의 변경은 원래 quadratic에 단순 상수를 더한 것과 다르며 정책의 선호도도 바뀐다.

v3.1에서는 보상 가중치·반감 오차·PPO를 유지하고, `foot_grip`의 비접촉 만점만 제거했다. 정지 상태에서 두 발 접촉은 1.0, 한 발만 접촉은 0.5, 두 발이 뜨면 0.0이다. 접촉 이력 3 sample을 쓰므로 순간 접촉 소실에 최대 약 12 ms 지연이 있을 수 있다. `roll_abs_deg`, `pitch_abs_deg`, `stance_width_m`, `both_feet_contact_fraction` 로그를 추가해 좌우 평균 상쇄와 발 지지 상태를 확인한다. 실물 접촉 안전성이나 강제 양발 접촉 제약을 보증하지 않는다.

v2까지는 네 개의 규제항이 **음의 running cost**였다. 그 최악값은 다음과 같았다.

```
음수 최대 = 0.02×25 + 0.03×25 + 0.02×25 + 0.02×100 = 3.75 /s
양수 최대 =                                          1.00 /s
```

자세가 무너진 상태에서는 양수 목표항이 모두 0에 가까워지므로 순 보상이 `−3.75/s`까지 내려갈 수 있었다. `gamma=0.99`, 정책 주기 50 Hz에서:

```
V(그 상태를 유지) = −0.075/step ÷ 0.01 = −7.5
V(즉시 종료)      = −1.0
```

이 계산은 최악의 보상률이 무한히 유지된다는 가정하의 경계 예시다. 실제로 그 상태가 도달·유지 가능한지, 유한 episode와 회복 가능성을 고려한 정책이 조기 종료를 선택하는지는 별도 검증이 필요하다. 음의 running cost만으로 자살 정책이 발생했다고 단정할 수 없다.

고정 길이 궤적에서 매 스텝 동일 상수를 더하면 순위는 유지된다. 그러나 v3의 quadratic→Gaussian 전환은 상수 이동이 아니다. 소오차에서는 quadratic으로 근사되지만 큰 오차에서는 포화되므로 규제 강도와 정책 선호도가 달라진다. v3의 보장은 running 합이 음수가 되지 않는다는 것까지다.

`terminating`은 실패 종료 시 한 번 부과되는 비용이다. 회귀 테스트는 이 항 외의 모든 항이 `[0, 1]`에 있고, 망가진 상태에서도 running 합이 음수가 되지 않는지를 검사한다. 이는 학습 성공이나 전이 성공의 증명은 아니다.

인정하는 trade-off: 유계 보상은 무한 압력을 줄 수 없다. `joint_limit`은 초과분이 0.5 rad이든 5 rad이든 같은 0을 준다. 기계적 한계 방어의 1차 방어선은 보상이 아니라 `ACTION_CLIPS`의 목표각 hard clip(한계에서 0.01 rad 안쪽)이며, 보상은 그 앞에서 미리 감속시키는 역할만 한다. `joint_limit`만 축별 평균이 아니라 합(AND) 형태를 써서 한 관절만 경계를 넘어도 항 전체가 떨어지도록 했다.

### 새 항의 기준값 근거

- `PELVIS_OFFSET_XY = (-0.053545, 0.0002) m`. MuJoCo에서 구한 굽힌 32-DOF 기본자세의 `root − 양발 중점`이며, `STANCE_DELTA_XY`와 같은 발바닥 기준점을 사용한다.
- `MIRROR_JOINT_PAIRS`는 `ACTUATED_JOINTS`에서 `l_`/`r_` 접두사로 유도한 6쌍이다. 좌우 규약은 모두 `q_L = −q_R`이며, 테스트가 이를 `action_normalization`의 offset·clip 부호 대칭으로 검증한다.
- `joint_limit`의 반감 오차 0.05 rad은 soft limit과 기계적 한계 사이의 가장 좁은 간격(발목 0.0593 rad)보다 작다. 즉 반감 지점이 항상 기계적 한계 안쪽에 있다. 테스트가 이 부등식을 검사한다.
- `target_delta`의 반감 오차 0.03 rad은 50 Hz에서 목표각 속도 1.5 rad/s에 해당하며, 가장 느린 액추에이터의 기본 속도 한계(RS02 4.0 rad/s)보다 작다. 이전 값 0.15 rad은 7.5 rad/s로 이 한계를 넘어 규제가 사실상 걸리지 않았다. 테스트가 이 부등식을 검사한다.
- `foot_yaw`는 발 링크와 몸통의 전방축을 XY로 투영해 구한 부호 있는 각이다. 몸통 전체가 yaw로 돌아간 경우에는 0이며, 다리만 비틀렸을 때만 커진다.

## 물리 기준과 한계

- 질량: URDF 합계 `20.513910 kg`, 체중 기준 `201.2414571 N`. 접촉 판정은 3 physics sample 이력의 최대 수직 반력 `> 0.01 × 체중`이며 약 2.01 N이다. 실제 접촉면적이나 하중 분배를 판정하지 않는다.
- 발바닥 중심: description의 `scripts/robonex_data.py::COLLISION_BOX` 발 박스 아래면 중심을 기하 기준점으로 사용한다. mesh 접촉 패치의 실측 중심이라는 뜻은 아니다. L `(0.0416, 0.0200, -0.0654)`, R `(0.0416, 0.0215, -0.0654) m`.
- URDF 영점 FK에서 위 기준점의 L−R 목표는 `(0.000002, 0.3194) m`. 이전 `0.321 m`는 다른 기준점의 폭이며 새 계산에 재사용하지 않았다. 이 값은 평형 자세를 실험으로 확인한 결과가 아니다. 기하가 바뀌면 계약값과 FK 회귀 테스트를 함께 재검토한다.
- 발바닥 점속도는 링크 원점 속도에 `ω × offset`을 더한다. COM 속도와 링크 원점 속도를 혼합하지 않는다.
- 토크는 `ImplicitActuator`의 clipping된 PD 추정치(`applied_torque`)다. PhysX의 직접 측정 토크가 아니다. RS02/RS03 정격 6/20 N·m를 정규화에 사용하며, 특히 RS03 정지 연속 토크 및 방열판 없는 실물의 열 한계를 보증하지 않는다.
- 모터·크랭크 12개만 명령 및 토크 규제 대상이다. 폐루프 수동 관절에는 목표를 추가하지 않는다. 기존 조립 상태 reset, hard limit, 250/50 Hz, 42차원 관측, 12차원 action, PPO 설정을 유지한다.
- `BalanceJointPositionAction`은 기존 offset/scale/clip 변환을 유지하며 관절 limit 적용 전후 목표 이력을 저장한다. `target_delta`는 limit 적용 전 목표를 사용해 경계 밖 출력이 변화량 규제를 회피하지 못한다. 부분 reset은 해당 환경 이력만 기본 목표로 복원한다. 비정상 action은 이전 목표를 유지하고 해당 환경을 `invalid_state`로 종료한다.
- `BalancingVecEnvWrapper`가 runner clip 전 action을 기록한다. 학습·play에서 모두 사용한다. 직접 환경을 호출하는 random/zero agent에서는 환경이 받은 action을 기록한다.
- `invalid_state`는 유한성/루트 회전 정상성 검사다. 기존 100 rad/s 전 관절 속도 guard와 낙상 높이 0.6 m를 유지한다. 발목 2-D 작업영역과 폐루프 동적 안정성 검증을 대체하지 않는다.

## 확인 방법

`tests/test_balance_rewards.py`는 시뮬레이터 없이 CPU tensor와 합성 asset으로 검증한다. 설치된 Isaac의 `JointAction`/`JointPositionAction` 계산 코드를 읽어 최소 ActionTerm 대역 위에서 실행하므로, 상속된 정규화·clip 계산과 새 이력을 함께 비교한다. PhysX 통합 테스트는 아니다.

```bash
conda activate isaacsim
python -m unittest discover -s tests -v
```

GUI 첫 검증 run은 `--max_iterations 200 --run_name gaussian_v1_smoke`로 시작한다. 장기 run은 동일 seed와 학습 예산에서 기존 보상과 비교한다. 보상 변경 이전 체크포인트는 새 실험의 초기화에 사용하지 않는다.

TensorBoard의 `Balance/*`에는 기울기·**roll·pitch 분리**·높이·수평속도·발배치 오차·**골반 오프셋 XY**·**발 yaw 절대값**·**좌우 대칭 RMS**·limit 적용 전 목표각 변화·추정 토크 비율·비정상 상태 비율을 기록한다. `Symmetry/<joint>_deg`는 좌우 6쌍 각각의 `q_L + q_R`이고, `FootYaw/l_foot_deg`·`FootYaw/r_foot_deg`는 발별 부호 있는 yaw다. `ActionLimitLower/<joint>`와 `ActionLimitUpper/<joint>`는 runner clip 전 정책 action이 그 관절의 목표각 clip에 닿는 action 값(`action_limit_reach`)을 넘은 비율이고, `Balance/joint_limit_excess_max_deg`는 soft limit 초과분의 관절 최대값이며, `ActionMean/<joint>`는 탐색이 포함된 batch 평균이다. 이 값들은 낙상률이나 deterministic policy 평가를 대체하지 않는다.

`Balance/tilt_deg`는 스칼라라 방향을 알 수 없어 v1에서 진단이 막혔다. v2에서 `roll_deg`/`pitch_deg`를 따로 기록하는 이유가 이것이다. 보상 자체는 뒤집힘을 π로 구분하는 스칼라 `tilt`를 계속 쓴다.

### 학습 종료 판단

v1 run에서 `Episode_Termination/fall_down`은 it 2500 `1.87%` → it 4000 `3.34%` → it 4999 `2.82%`로 **후반에 나빠졌고**, 같은 구간에서 `Balance/tilt_deg`도 `0.59° → 0.85°`로 나빠졌다. 총 보상은 it 4000에서 정점 후 감소했다. iteration 수를 목표로 삼지 말고 위 체크리스트가 함께 개선되는 동안만 계속한다.

이번 비교 실험은 v1 체크포인트를 resume하지 않고 새 학습으로 시작한다. 기존 critic은 이전 보상에 맞춰져 있어 재적응이 필요하며, 무조건 재사용 불가능하다는 뜻은 아니다. 보상 변경 효과를 구분하기 위해 같은 seed·학습 예산으로 비교한다.

학습 시 `source_snapshot/`에 실제 import된 로컬 task 패키지와 RSL-RL 실행 스크립트를 복사한다. 아직 Git에 추가되지 않은 보상 파일도 보존된다. 이 복사는 외부 SDK나 USD 자산 전체를 보존하는 것은 아니다.

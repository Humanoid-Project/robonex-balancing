# Project memory

Shared knowledge lives in `/home/polygon/humanoid_project/memory` (`rules.md` auto-loads via `.Codex/rules/rules.md`).

For RoboNex domain tasks, read `robonex.md` first for project context, then only the specific file(s) below relevant to the task. Skip all of this for tasks unrelated to the robot itself.

| File | Covers |
|---|---|
| `robonex.md` | Master overview — config, hardware, current status, priorities |
| `project-open-items.md` | Open items only; resolved ones get deleted or moved into a domain file |
| `hw-motors.md` | RS02/RS03 — measurement-methodology pitfalls, gravity-compensation sign convention, decision history. Numeric tables live in repo docs, not here |
| `hw-robstride-protocol.md` | 3 CAN protocol types, zeroing convention, position-domain pitfall (real past incident) |
| `hw-canbus.md` | Wiring/setup, ID 1-12 mapping, feedback-loss bug, reliability issues |
| `hw-power.md` | 48V PSU (OVP 56.6V), ED250 E-stop (don't leave it switched on) |
| `hw-imu-n100.md` | N100 IMU C++ SDK done; remaining: case + bias re-measurement |
| `sw-repo-map.md` | Map of the 5 repos, Robstride-Motor-Test structure, atom01 reference |
| `sw-robonex-description.md` | Single-source URDF + 4 sim targets, closed-loop handling, joint naming |
| `sw-robonex-balancing.md` | Isaac Lab PPO 2-track setup, action vector order, divergence incident |

Repo docs (not memory) are the source of truth for raw physical values — see `robonex.md` for exact paths.

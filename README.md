# RoboNex Balancing

## Setup
```bash
# Example
cd ~/humanoid_project
git clone https://github.com/Humanoid-Project/robonex-balancing.git
git clone https://github.com/Humanoid-Project/robonex-description.git
git clone https://github.com/Humanoid-Project/robonex-common.git
source ./robonex-common/setup/setup_isaacsim.sh
python -m pip install -e ./robonex-balancing/source/robonex_balancing
```

Runs on the `isaacsim` conda env, not a `.venv`. `robonex-common` is pinned in
`source/robonex_balancing/setup.py` — see [`robonex-common/setup/SETUP.md`](https://github.com/Humanoid-Project/robonex-common/blob/main/setup/SETUP.md).

| Variable | Required | Default | Description |
| --- | :---: | --- | --- |
| `ROBONEX_DESCRIPTION_ROOT` | No | Sibling `robonex-description` | Description checkout |

<br>

## Structure

```text
robonex-balancing/
├── README.md
├── scripts/
│   ├── list_envs.py
│   ├── zero_agent.py
│   ├── random_agent.py
│   ├── export_policy_manifest.py
│   └── rsl_rl/
│       ├── train.py
│       ├── play.py
│       └── cli_args.py
└── source/robonex_balancing/
```

| Task | USD |
| --- | --- |
| `RoboNex-Balancing-v0` | `robonex-description/isaac/closed_loop_mesh/robonex_closed_loop_mesh.usd` |

<br>

## Train

### `scripts/rsl_rl/train.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--task` | `Required` | Gym task id |
| - | `--num_envs` | cfg (`512`) | Parallel environments |
| - | `--max_iterations` | cfg | PPO iterations |
| - | `--seed` | - | Environment seed |
| - | `--resume` | Off | Resume from a checkpoint |
| - | `--load_run` | - | Run folder to resume |
| - | `--checkpoint` | - | Checkpoint file to resume |
| - | `--experiment_name` | - | Parsed but currently not applied; the task config still determines the log folder name |
| - | `--run_name` | - | Run-name suffix |
| - | `--logger` | - | `wandb`, `tensorboard`, or `neptune` |

```bash
# Example
cd ~/humanoid_project/robonex-balancing
conda activate isaacsim

python scripts/rsl_rl/train.py \
  --task RoboNex-Balancing-v0 \
  --num_envs 512
```

<br>

## Play

### `scripts/rsl_rl/play.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--task` | `Required` | Gym task id |
| - | `--num_envs` | cfg | Parallel environments |
| - | `--load_run` | - | Run folder to load |
| - | `--checkpoint` | - | Checkpoint file |
| - | `--real-time` | Off | Pace playback to wall clock |
| - | `--video` | Off | Record a video |
| - | `--video_length` | `200` | Recorded steps |
| - | `--export-only` | Off | Export the policy and exit |
| - | `--export-dir` | Checkpoint `exported/` | Export directory |
| - | `--eval-steps` | `0` | Stop after this many policy steps; `0` keeps playback open |
| - | `--eval-baseline` | Off | Evaluate without observation noise, pushes, or reset randomization |
| - | `--eval-output` | - | Save observation, action, and robot state traces as NPZ |
| - | `--eval-zero-joint-friction` | Off | Disable joint friction for a diagnostic comparison |

```bash
# Example
python scripts/rsl_rl/play.py \
  --task RoboNex-Balancing-v0 \
  --num_envs 1
```

```bash
# Example
python scripts/rsl_rl/play.py \
  --task RoboNex-Balancing-v0 \
  --num_envs 1 \
  --headless \
  --export-only \
  --checkpoint logs/rsl_rl/robonex_balancing_closed_loop/2026-09-06_04-26-18/model_1500.pt \
  --export-dir ../robonex-deploy/policies/2026-09-06_04-26-18_iter1500
```

<br>

### `scripts/list_envs.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--keyword` | - | Filter registered tasks |

```bash
# Example
python scripts/list_envs.py
```

<br>

### `scripts/zero_agent.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `--task` | `Required` | Gym task id |
| - | `--num_envs` | - | Parallel environments |

```bash
# Example
python scripts/zero_agent.py \
  --task RoboNex-Balancing-v0 \
  --num_envs 1
```

<br>

## Manifest

### `scripts/export_policy_manifest.py`

| Command | Option | Default | Description |
| --- | --- | --- | --- |
| - | `policy` | `Required` | ONNX policy path |
| - | `--output` | `<policy_dir>/policy_manifest.json` | Manifest path |
| - | `--description-root` | Sibling checkout | `robonex-description` path |
| - | `--common-root` | Sibling checkout | `robonex-common` path |
| - | `--description-model` | `mujoco/robot/scene.xml` | Model path stored in the manifest |

```bash
# Example
python scripts/export_policy_manifest.py /path/to/policy.onnx
```

Schema 2 fingerprints the policy, MuJoCo XML/mesh bundle, `robonex-common` source, and
training source. MuJoCo deployment rejects a changed model bundle or common runtime even
when the recorded Git commit is unchanged.

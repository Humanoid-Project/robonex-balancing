# RoboNex Balancing

Full execution-argument inventory: [scripts/README.md](scripts/README.md), including inherited launcher options and known ineffective flags.

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

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--task` | Yes | - | Gym task id |
| `--num_envs` | No | cfg (`512`) | Parallel environments |
| `--max_iterations` | No | cfg | PPO iterations |
| `--seed` | No | - | Environment seed |
| `--resume` | No | Off | Resume from a checkpoint |
| `--load_run` | No | - | Run folder to resume |
| `--checkpoint` | No | - | Checkpoint file to resume |
| `--experiment_name` | No | - | Parsed but currently not applied; the task config still determines the log folder name |
| `--run_name` | No | - | Run-name suffix |
| `--logger` | No | - | `wandb`, `tensorboard`, or `neptune` |

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

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--task` | Yes | - | Gym task id |
| `--num_envs` | No | - | Parallel environments |
| `--load_run` | No | - | Run folder to load |
| `--checkpoint` | No | - | Checkpoint file |
| `--real-time` | No | Off | Pace playback to wall clock |
| `--video` | No | Off | Record a video |
| `--video_length` | No | `200` | Recorded steps |

```bash
# Example
python scripts/rsl_rl/play.py \
  --task RoboNex-Balancing-v0 \
  --num_envs 1
```

<br>

### `scripts/list_envs.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--keyword` | No | - | Filter registered tasks |

```bash
# Example
python scripts/list_envs.py
```

<br>

### `scripts/zero_agent.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `--task` | Yes | - | Gym task id |
| `--num_envs` | No | - | Parallel environments |

```bash
# Example
python scripts/zero_agent.py \
  --task RoboNex-Balancing-v0 \
  --num_envs 1
```

<br>

## Manifest

### `scripts/export_policy_manifest.py`

| Option | Required | Default | Description |
| --- | :---: | --- | --- |
| `policy` | Yes | - | ONNX policy path |
| `--output` | No | `<policy_dir>/policy_manifest.json` | Manifest path |
| `--description-root` | No | Sibling checkout | `robonex-description` path |
| `--common-root` | No | Sibling checkout | `robonex-common` path |
| `--description-model` | No | `mujoco/basic/scene.xml` | Model path stored in the manifest |

```bash
# Example
python scripts/export_policy_manifest.py /path/to/policy.onnx
```

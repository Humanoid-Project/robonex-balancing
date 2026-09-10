# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--export-only", action="store_true", default=False, help="Export the policy and exit.")
parser.add_argument("--export-dir", type=str, default=None, help="Directory for exported policy files.")
parser.add_argument("--eval-steps", type=int, default=0, help="Stop after this many policy steps; zero keeps playback open.")
parser.add_argument("--eval-baseline", action="store_true", help="Evaluate without observation noise, pushes, or reset randomization.")
parser.add_argument("--eval-output", type=str, help="Save runtime observation, action, and robot state traces as NPZ.")
parser.add_argument("--eval-zero-joint-friction", action="store_true", help="Disable joint friction for a diagnostic comparison.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, export_policy_as_jit, export_policy_as_onnx
from balancing_wrapper import BalancingVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robonex_balancing.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.eval_baseline:
        env_cfg.observations.policy.enable_corruption = False
        for name in ("randomize_actuator_gains", "push_robot", "randomize_friction", "randomize_joint_friction", "randomize_mass"):
            setattr(env_cfg.events, name, None)
        env_cfg.events.reset_base.params["pose_range"] = {}
        env_cfg.events.reset_base.params["velocity_range"] = {}
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = BalancingVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = (
        os.path.abspath(os.path.expanduser(args_cli.export_dir))
        if args_cli.export_dir is not None
        else os.path.join(os.path.dirname(resume_path), "exported")
    )
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    if args_cli.export_only:
        env.close()
        return

    dt = env.unwrapped.step_dt
    if args_cli.eval_zero_joint_friction:
        robot = env.unwrapped.scene["robot"]
        robot.write_joint_friction_coefficient_to_sim(torch.zeros_like(robot.data.joint_pos))

    # reset environment
    obs = env.get_observations()
    if args_cli.eval_steps:
        print("EVAL_OBS", obs["policy"][0].tolist(), flush=True)
        print("EVAL_JOINTS", env.unwrapped.action_manager.get_term("joint_pos")._joint_names, flush=True)
    timestep = 0
    eval_timestep = 0
    eval_trace = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if args_cli.eval_output:
                robot = env.unwrapped.scene["robot"]
                term = env.unwrapped.action_manager.get_term("joint_pos")
                eval_trace.append((obs["policy"][0].cpu().numpy().copy(), actions[0].cpu().numpy().copy(), robot.data.joint_pos[0, term._joint_ids].cpu().numpy().copy(), robot.data.root_state_w[0].cpu().numpy().copy()))
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.eval_steps:
            from robonex_balancing.tasks.manager_based.robonex_balancing.mdp.balance_metrics import balance_metrics, point_state, yaw_frame
            metrics = balance_metrics(env.unwrapped)
            robot = env.unwrapped.scene["robot"]
            feet, _ = point_state(robot.data.body_link_pos_w[:, metrics.foot_ids], robot.data.body_link_quat_w[:, metrics.foot_ids], robot.data.body_link_lin_vel_w[:, metrics.foot_ids], robot.data.body_link_ang_vel_w[:, metrics.foot_ids], metrics.sole_offsets)
            delta = yaw_frame(feet[:, 0] - feet[:, 1], robot.data.root_quat_w)
            if eval_timestep % 250 == 0:
                print("EVAL", eval_timestep, "width", delta[:, 1].mean().item(), "tilt", metrics.log["Balance/tilt_deg"].item(), "joint_names", metrics.joint_names, "q", robot.data.joint_pos[:, metrics.joint_ids].mean(0).tolist(), flush=True)
            eval_timestep += 1
            if eval_timestep >= args_cli.eval_steps:
                break
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.eval_output:
        import numpy as np
        robot = env.unwrapped.scene["robot"]
        term = env.unwrapped.action_manager.get_term("joint_pos")
        trace_obs, trace_actions, trace_q, trace_root = zip(*eval_trace)
        np.savez(args_cli.eval_output, obs=trace_obs, actions=trace_actions, q=trace_q, root=trace_root, joint_names=term._joint_names, body_names=robot.body_names, mass=robot.data.default_mass.cpu().numpy(), inertia=robot.data.default_inertia.cpu().numpy(), stiffness=robot.data.joint_stiffness.cpu().numpy(), damping=robot.data.joint_damping.cpu().numpy(), all_joint_names=robot.joint_names)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

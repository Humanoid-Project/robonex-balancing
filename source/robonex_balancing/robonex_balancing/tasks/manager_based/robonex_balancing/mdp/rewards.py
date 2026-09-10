import torch

from ..robot_contract import ROBOT_WEIGHT_N
from .balance_metrics import (
    balance_metrics,
    contact_grip_reward,
    gaussian_axis_mean,
    gaussian_half_error,
    height_error,
)


def _live(env, values, reward):
    return torch.where(values["valid"] & ~env.termination_manager.terminated, reward, 0.0)


def balance_tracking(env, metric, half_error):
    values = balance_metrics(env).values
    return _live(env, values, gaussian_half_error(values[metric], half_error))


def balance_axis_tracking(env, metric, half_error):
    values = balance_metrics(env).values
    return _live(env, values, gaussian_axis_mean(values[metric], half_error))


def balance_height(env, target_height, lower_tolerance, half_error):
    values = balance_metrics(env).values
    error = height_error(values["height"], target_height, lower_tolerance).unsqueeze(-1)
    return _live(env, values, gaussian_half_error(error, (half_error,)))


def balance_symmetry(env, half_error):
    values = balance_metrics(env).values
    scale = torch.full_like(values["symmetry_error"][0], half_error)
    return _live(env, values, gaussian_axis_mean(values["symmetry_error"], scale))


def balance_effort(env, metric, reference):
    values = balance_metrics(env).values
    scale = torch.full_like(values[metric][0], reference)
    return _live(env, values, gaussian_axis_mean(values[metric], scale))


def balance_joint_limit(env, half_error):
    values = balance_metrics(env).values
    return _live(env, values, gaussian_half_error(values["joint_limit_excess"], (half_error,)))


def balance_foot_grip(env, reference_velocity, contact_weight_fraction):
    values = balance_metrics(env).values
    grip = contact_grip_reward(
        values["sole_velocity_xy"], values["contact_force_z"],
        ROBOT_WEIGHT_N * contact_weight_fraction, reference_velocity,
    )
    return _live(env, values, grip)


def balance_termination_cost(env):
    return env.termination_manager.terminated.float() / env.step_dt

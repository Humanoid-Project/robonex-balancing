import torch

from .balance_metrics import balance_metrics


def balance_invalid_state(env):
    return ~balance_metrics(env).values["valid"]


def unstable_joint_vel(env, limit):
    velocity = env.scene["robot"].data.joint_vel
    return (~torch.isfinite(velocity) | (velocity.abs() > limit)).any(dim=1)

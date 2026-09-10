import math

import torch


def gaussian_half_error(error, half_error):
    scale = torch.as_tensor(half_error, dtype=error.dtype, device=error.device)
    normalized = error / scale
    valid = torch.isfinite(normalized).all(dim=-1)
    normalized = torch.nan_to_num(normalized, nan=100.0, posinf=100.0, neginf=-100.0)
    squared_error = normalized.clamp(-100.0, 100.0).square().sum(dim=-1)
    return torch.where(valid, torch.exp(-math.log(2.0) * squared_error), 0.0)


def gaussian_axis_mean(error, half_error):
    scale = torch.as_tensor(half_error, dtype=error.dtype, device=error.device)
    normalized = error / scale
    valid = torch.isfinite(normalized).all(dim=-1)
    normalized = torch.nan_to_num(normalized, nan=100.0, posinf=100.0, neginf=-100.0)
    reward = torch.exp(-math.log(2.0) * normalized.clamp(-100.0, 100.0).square()).mean(dim=-1)
    return torch.where(valid, reward, 0.0)


def bounded_mean_square(value, reference, max_penalty):
    normalized = value / reference
    normalized = torch.nan_to_num(normalized, nan=1.0e3, posinf=1.0e3, neginf=-1.0e3)
    return normalized.clamp(-1.0e3, 1.0e3).square().mean(dim=-1).clamp(max=max_penalty)


def height_error(height, target_height, lower_tolerance):
    return (height - target_height).clamp(min=0.0) + (target_height - lower_tolerance - height).clamp(min=0.0)


def tilt_angle(gravity):
    return torch.atan2(torch.linalg.vector_norm(gravity[..., :2], dim=-1), -gravity[..., 2])


def roll_pitch(gravity):
    roll = torch.atan2(-gravity[..., 1], -gravity[..., 2])
    pitch = torch.atan2(gravity[..., 0], torch.linalg.vector_norm(gravity[..., 1:], dim=-1))
    return torch.stack((roll, pitch), dim=-1)


def rotate_vector(quat, vector):
    xyz = quat[..., 1:]
    uv = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + torch.linalg.cross(xyz, uv, dim=-1))


def yaw_frame(vector, quat):
    w, x, y, z = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * vector[..., 0] + s * vector[..., 1],
                        -s * vector[..., 0] + c * vector[..., 1], vector[..., 2]), dim=-1)


def forward_axis(quat, shape):
    axis = quat.new_zeros(shape)
    axis[..., 0] = 1.0
    return rotate_vector(quat, axis)


def relative_yaw(quat, reference_quat):
    shape = torch.broadcast_shapes(quat.shape, reference_quat.shape)[:-1] + (3,)
    forward = forward_axis(quat, shape)
    base = forward_axis(reference_quat, shape)
    cross = base[..., 0] * forward[..., 1] - base[..., 1] * forward[..., 0]
    dot = base[..., 0] * forward[..., 0] + base[..., 1] * forward[..., 1]
    return torch.atan2(cross, dot)


def point_state(position, quat, linear_velocity, angular_velocity, local_offset):
    offset = rotate_vector(quat, local_offset)
    return position + offset, linear_velocity + torch.linalg.cross(angular_velocity, offset, dim=-1)


def contact_grip_reward(velocity, contact_force_z, threshold, reference):
    contacts = contact_force_z > threshold
    normalized = torch.nan_to_num(velocity / reference, nan=1.0e3, posinf=1.0e3, neginf=-1.0e3)
    speed_sq = normalized.clamp(-1.0e3, 1.0e3).square().sum(dim=-1).clamp(max=100.0)
    grip = torch.exp(-math.log(2.0) * speed_sq)
    return torch.where(contacts, grip, torch.zeros_like(grip)).mean(dim=-1)


class BalanceMetrics:
    def __init__(self, env):
        from ..robot_contract import (
            FEET,
            FOOT_SOLE_OFFSETS,
            LEG_JOINTS,
            MIRROR_JOINT_PAIRS,
            PELVIS_OFFSET_XY,
            RATED_TORQUE_BY_JOINT,
            ROBOT_WEIGHT_N,
            STANCE_DELTA_XY,
        )

        self.robot = env.scene["robot"]
        self.contact_threshold = 0.01 * ROBOT_WEIGHT_N
        self.sensor = env.scene.sensors["contact_forces"]
        self.joint_ids, self.joint_names = self.robot.find_joints(LEG_JOINTS, preserve_order=True)
        self.foot_ids, _ = self.robot.find_bodies(FEET, preserve_order=True)
        self.sensor_ids, _ = self.sensor.find_bodies(FEET, preserve_order=True)
        self.sole_offsets = torch.tensor(FOOT_SOLE_OFFSETS, device=env.device).expand(env.num_envs, -1, -1)
        self.stance_delta = torch.tensor(STANCE_DELTA_XY, device=env.device)
        self.pelvis_offset = torch.tensor(PELVIS_OFFSET_XY, device=env.device)
        order = list(self.joint_names)
        self.mirror_left = [order.index(pair[0]) for pair in MIRROR_JOINT_PAIRS]
        self.mirror_right = [order.index(pair[1]) for pair in MIRROR_JOINT_PAIRS]
        self.mirror_names = tuple(pair[0][2:] for pair in MIRROR_JOINT_PAIRS)
        self.rated_torque = torch.tensor([RATED_TORQUE_BY_JOINT[name] for name in self.joint_names], device=env.device)
        self.action = env.action_manager.get_term("joint_pos")
        clip = self.action._clip
        offset = torch.as_tensor(self.action._offset, dtype=clip.dtype, device=clip.device)
        scale = torch.as_tensor(self.action._scale, dtype=clip.dtype, device=clip.device)
        self.action_fence = (clip - offset.unsqueeze(-1)) / scale.unsqueeze(-1)
        self.step = None
        self.values = {}
        self.log = {}

    def update(self, env):
        if self.step == env.common_step_counter:
            return self
        self.step = env.common_step_counter
        data = self.robot.data
        feet = self.foot_ids
        sole_pos, sole_vel = point_state(
            data.body_link_pos_w[:, feet], data.body_link_quat_w[:, feet],
            data.body_link_lin_vel_w[:, feet], data.body_link_ang_vel_w[:, feet], self.sole_offsets,
        )
        delta = yaw_frame(sole_pos[:, 0] - sole_pos[:, 1], data.root_quat_w)
        raw = self.action.policy_actions
        valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        force_history = self.sensor.data.net_forces_w_history[:, :, self.sensor_ids]
        for value in (data.root_pos_w, data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_b,
                      data.projected_gravity_b, data.joint_pos, data.joint_vel,
                      data.body_link_pos_w, data.body_link_quat_w, data.body_link_lin_vel_w,
                      data.body_link_ang_vel_w, data.applied_torque[:, self.joint_ids], force_history, raw):
            valid &= torch.isfinite(value).reshape(env.num_envs, -1).all(dim=1)
        valid &= (torch.linalg.vector_norm(data.root_quat_w, dim=-1) - 1.0).abs() < 0.1
        valid &= (torch.linalg.vector_norm(data.projected_gravity_b, dim=-1) - 1.0).abs() < 0.1
        target_delta = self.action.unclipped_targets - self.action.previous_unclipped_targets
        contact_z = force_history[..., 2].amax(dim=1)
        sole_mid = sole_pos.mean(dim=1)
        pelvis = yaw_frame(data.root_pos_w - sole_mid, data.root_quat_w)
        joint_pos = data.joint_pos[:, self.joint_ids]
        soft_limits = data.soft_joint_pos_limits[:, self.joint_ids]
        joint_limit_excess = ((soft_limits[..., 0] - joint_pos).clamp(min=0.0)
                              + (joint_pos - soft_limits[..., 1]).clamp(min=0.0))
        self.values = {
            "valid": valid,
            "tilt": tilt_angle(data.projected_gravity_b).unsqueeze(-1),
            "height": data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
            "linear_velocity": data.root_lin_vel_w,
            "angular_velocity": data.root_ang_vel_b,
            "stance_error": delta[:, :2] - self.stance_delta,
            "pelvis_error": pelvis[:, :2] - self.pelvis_offset,
            "foot_yaw": relative_yaw(data.body_link_quat_w[:, feet], data.root_quat_w.unsqueeze(1)),
            "symmetry_error": joint_pos[:, self.mirror_left] + joint_pos[:, self.mirror_right],
            "target_delta": target_delta,
            "torque_ratio": data.applied_torque[:, self.joint_ids] / self.rated_torque,
            "sole_velocity_xy": sole_vel[..., :2],
            "contact_force_z": contact_z,
            "joint_limit_excess": joint_limit_excess,
        }
        angles = torch.rad2deg(roll_pitch(data.projected_gravity_b))
        self.log = {
            "Balance/tilt_deg": torch.rad2deg(self.values["tilt"]).mean(),
            "Balance/roll_deg": angles[:, 0].mean(),
            "Balance/pitch_deg": angles[:, 1].mean(),
            "Balance/roll_abs_deg": angles[:, 0].abs().mean(),
            "Balance/pitch_abs_deg": angles[:, 1].abs().mean(),
            "Balance/stance_width_m": delta[:, 1].mean(),
            "Balance/both_feet_contact_fraction": (contact_z > self.contact_threshold).all(dim=-1).float().mean(),
            "Balance/height_m": self.values["height"].mean(),
            "Balance/speed_xy_m_s": torch.linalg.vector_norm(data.root_lin_vel_w[:, :2], dim=-1).mean(),
            "Balance/stance_error_m": torch.linalg.vector_norm(self.values["stance_error"], dim=-1).mean(),
            "Balance/pelvis_error_x_m": self.values["pelvis_error"][:, 0].mean(),
            "Balance/pelvis_error_y_m": self.values["pelvis_error"][:, 1].mean(),
            "Balance/foot_yaw_abs_deg": torch.rad2deg(self.values["foot_yaw"].abs()).mean(),
            "Balance/symmetry_rms_deg": torch.rad2deg(self.values["symmetry_error"].square().mean().sqrt()),
            "Balance/invalid_fraction": (~valid).float().mean(),
            "Balance/target_delta_rms_rad": target_delta.square().mean().sqrt(),
            "Balance/torque_rated_rms_estimate": self.values["torque_ratio"].square().mean().sqrt(),
            "Balance/joint_limit_excess_max_deg": torch.rad2deg(joint_limit_excess.amax(dim=-1)).mean(),
        }
        for index, name in enumerate(self.mirror_names):
            self.log[f"Symmetry/{name}_deg"] = torch.rad2deg(self.values["symmetry_error"][:, index]).mean()
        for index, name in enumerate(("l_foot", "r_foot")):
            self.log[f"FootYaw/{name}_deg"] = torch.rad2deg(self.values["foot_yaw"][:, index]).mean()
        for index, name in enumerate(self.action._joint_names):
            self.log[f"ActionLimitLower/{name}"] = (raw[:, index] < self.action_fence[:, index, 0]).float().mean()
            self.log[f"ActionLimitUpper/{name}"] = (raw[:, index] > self.action_fence[:, index, 1]).float().mean()
            self.log[f"ActionMean/{name}"] = raw[:, index].mean()
        self.log = {key: torch.nan_to_num(value.detach(), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
                    for key, value in self.log.items()}
        return self


def balance_metrics(env):
    if not hasattr(env, "_balance_metrics"):
        env._balance_metrics = BalanceMetrics(env)
    return env._balance_metrics.update(env)

import ast
import importlib
import importlib.util
import inspect
import math
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_balancing/robonex_balancing/tasks/manager_based/robonex_balancing"
for name, path in (("balance_test_task", TASK), ("balance_test_task.mdp", TASK / "mdp")):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

metrics = importlib.import_module("balance_test_task.mdp.balance_metrics")
contract = importlib.import_module("balance_test_task.robot_contract")
rewards = importlib.import_module("balance_test_task.mdp.rewards")
terminations = importlib.import_module("balance_test_task.mdp.terminations")

from robonex_common.joints import DEFAULT_JOINT_POS, JOINT_LIMITS_BY_NAME
from robonex_common.limits import ACTION_SCALE_RAD, RUNNER_ACTION_CLIP, action_limit_reach
from robonex_common.motors import DEFAULT_VELOCITY_LIMIT_CURRENT

SOFT_JOINT_POS_LIMIT_FACTOR = 0.9


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Asset:
    def __init__(self, num_envs=3):
        self.joint_names = list(reversed(contract.LEG_JOINTS)) + ["passive_joint"]
        self.num_joints = len(self.joint_names)
        limits = torch.tensor([JOINT_LIMITS_BY_NAME.get(name, (-100., 100.))
                               for name in self.joint_names])
        default = torch.tensor([DEFAULT_JOINT_POS.get(name, 0.) for name in self.joint_names])
        mean = limits.mean(dim=-1, keepdim=True)
        half = SOFT_JOINT_POS_LIMIT_FACTOR * (limits[:, 1:] - limits[:, :1]) / 2.
        soft = torch.cat((mean - half, mean + half), dim=-1)
        self.data = types.SimpleNamespace(
            default_joint_pos=default.repeat(num_envs, 1),
            soft_joint_pos_limits=soft.unsqueeze(0).repeat(num_envs, 1, 1),
        )

    def find_joints(self, names, preserve_order=False):
        selected = list(names) if preserve_order else [name for name in self.joint_names if name in names]
        return [self.joint_names.index(name) for name in selected], selected


class ActionTerm:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._asset = env.scene["robot"]
        self.num_envs = env.num_envs
        self.device = env.device


def load_action_class():
    isaac_root = Path(importlib.util.find_spec("isaaclab").submodule_search_locations[0])
    string_utils = load_module("balance_test_string", isaac_root / "utils/string.py")
    path = isaac_root / "envs/mdp/actions/joint_actions.py"
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"JointAction", "JointPositionAction"}]
    namespace = {"ActionTerm": ActionTerm, "torch": torch, "string_utils": string_utils,
                 "logger": types.SimpleNamespace(info=lambda *args: None)}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *classes], type_ignores=[])), str(path), "exec"), namespace)
    stub = types.ModuleType("isaaclab.envs.mdp.actions.joint_actions")
    stub.JointPositionAction = namespace["JointPositionAction"]
    with patch.dict(sys.modules, {stub.__name__: stub}):
        return importlib.import_module("balance_test_task.mdp.actions").BalanceJointPositionAction


BalanceAction = load_action_class()


def make_action():
    asset = Asset()
    env = types.SimpleNamespace(scene={"robot": asset}, num_envs=3, device="cpu")
    cfg = types.SimpleNamespace(joint_names=contract.LEG_JOINTS, preserve_order=False,
                                scale=contract.ACTION_SCALES, offset=contract.ACTION_OFFSETS,
                                clip=contract.ACTION_CLIPS, use_default_offset=False)
    return BalanceAction(cfg, env)


def make_env():
    action = make_action()
    robot = action._asset
    data = robot.data
    bodies = ["r_foot", "base_link", "l_foot"]
    robot.find_bodies = lambda names, preserve_order: ([bodies.index(name) for name in names], list(names))
    data.root_pos_w = torch.tensor([[0., 0., contract.BASE_HEIGHT]]).repeat(3, 1)
    data.root_quat_w = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
    data.root_lin_vel_w = torch.zeros(3, 3)
    data.root_ang_vel_b = torch.zeros(3, 3)
    data.projected_gravity_b = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
    data.joint_pos = data.default_joint_pos.clone()
    data.joint_vel = torch.zeros(3, 13)
    data.applied_torque = torch.zeros(3, 13)
    data.body_link_pos_w = torch.tensor([[[0.011944, -0.1814, 0.06545], [0., 0., contract.BASE_HEIGHT],
                                         [0.011946, 0.1395, 0.06545]]]).repeat(3, 1, 1)
    data.body_link_quat_w = torch.tensor([[[1., 0., 0., 0.]]]).repeat(3, 3, 1)
    data.body_link_lin_vel_w = torch.zeros(3, 3, 3)
    data.body_link_ang_vel_w = torch.zeros(3, 3, 3)
    force = torch.zeros(3, 3, 2, 3)
    force[..., 2] = 100.
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w_history=force))
    sensor.find_bodies = lambda names, preserve_order: ([1, 0], list(names))

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.sensors = {"contact_forces": sensor}
    scene.env_origins = torch.zeros(3, 3)
    return types.SimpleNamespace(scene=scene, action_manager=types.SimpleNamespace(get_term=lambda name: action),
                                 num_envs=3, device="cpu", step_dt=0.02, common_step_counter=1,
                                 termination_manager=types.SimpleNamespace(terminated=torch.zeros(3, dtype=torch.bool)))


def rewards_cfg_class():
    tree = ast.parse((TASK / "robonex_balancing_env_cfg.py").read_text())
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg")


def literal(node):
    if isinstance(node, ast.Name) and node.id == "BASE_HEIGHT":
        return contract.BASE_HEIGHT
    if isinstance(node, ast.Call) and node.func.id == "radians":
        return math.radians(literal(node.args[0]))
    if isinstance(node, ast.Tuple):
        return tuple(literal(item) for item in node.elts)
    return ast.literal_eval(node)


def reward_terms():
    cfg = rewards_cfg_class()
    for node in cfg.body:
        if not isinstance(node, ast.Assign):
            continue
        kwargs = {kw.arg: kw.value for kw in node.value.keywords}
        params = kwargs.get("params")
        values = {} if params is None else {k.value: literal(v) for k, v in zip(params.keys, params.values)}
        yield node.targets[0].id, kwargs["func"].attr, ast.literal_eval(kwargs["weight"]), values


class BalanceRewardTests(unittest.TestCase):
    def test_gaussian_half_error_and_vector_composition(self):
        result = metrics.gaussian_half_error(torch.tensor([[0., 0.], [2., 0.], [2., 3.]]), (2., 3.))
        torch.testing.assert_close(result, torch.tensor([1., 0.5, 0.25]))

    def test_gaussian_invalid_and_extreme_errors(self):
        result = metrics.gaussian_half_error(torch.tensor([[float("nan")], [float("inf")], [1.e30]]), (0.1,))
        torch.testing.assert_close(result, torch.zeros(3))

    def test_axis_gaussian_preserves_other_axis_signals(self):
        error = torch.tensor([[0., 0., 0.], [2., 0., 0.], [2., 3., 0.]])
        result = metrics.gaussian_axis_mean(error, (2., 3., 4.))
        torch.testing.assert_close(result, torch.tensor([1., 5. / 6., 2. / 3.]))

    def test_axis_gaussian_rejects_nonfinite_vector(self):
        result = metrics.gaussian_axis_mean(torch.tensor([[0., float("nan"), 0.]]), (1., 1., 1.))
        torch.testing.assert_close(result, torch.zeros(1))

    def test_tilt_distinguishes_inversion(self):
        gravity = torch.tensor([[0., 0., -1.], [1., 0., 0.], [0., 0., 1.]])
        torch.testing.assert_close(metrics.tilt_angle(gravity), torch.tensor([0., math.pi / 2, math.pi]))

    def test_height_band(self):
        result = metrics.height_error(torch.tensor([1.05, 1.06, 1.07, 1.08, 1.09]), 1.08, 0.02)
        torch.testing.assert_close(result, torch.tensor([0.01, 0., 0., 0., 0.01]), atol=1.e-6, rtol=1.e-5)

    def test_yaw_invariance(self):
        q = torch.tensor([[math.cos(0.7), 0., 0., math.sin(0.7)]])
        delta = torch.tensor([[0.000002, 0.3194, 0.]])
        torch.testing.assert_close(metrics.yaw_frame(metrics.rotate_vector(q, delta), q), delta)

    def test_yaw_frame_ignores_pure_roll(self):
        q = torch.tensor([[math.cos(0.2), math.sin(0.2), 0., 0.]])
        delta = torch.tensor([[0., 0.3194, 0.]])
        torch.testing.assert_close(metrics.yaw_frame(delta, q), delta)

    def test_point_velocity_includes_rotation(self):
        q = torch.tensor([[[1., 0., 0., 0.]]])
        zero = torch.zeros(1, 1, 3)
        position, velocity = metrics.point_state(zero, q, zero, torch.tensor([[[0., 0., 2.]]]),
                                                 torch.tensor([[[0.1, 0., 0.]]]))
        torch.testing.assert_close(position, torch.tensor([[[0.1, 0., 0.]]]))
        torch.testing.assert_close(velocity, torch.tensor([[[0., 0.2, 0.]]]))

    def test_grip_only_scores_supporting_feet(self):
        velocity = torch.tensor([[[0.1, 0.], [3., 0.]]])
        force = torch.tensor([[100., 0.]])
        torch.testing.assert_close(metrics.contact_grip_reward(velocity, force, 2., 0.1), torch.tensor([0.25]))

    def test_grip_requires_contact_even_at_zero_velocity(self):
        velocity = torch.zeros(3, 2, 2)
        force = torch.tensor([[0., 0.], [100., 0.], [100., 100.]])
        torch.testing.assert_close(metrics.contact_grip_reward(velocity, force, 2., 0.1), torch.tensor([0., 0.5, 1.]))

    def test_grip_is_finite_for_extreme_velocity(self):
        velocity = torch.tensor([[[float("nan"), 0.], [1.e30, 0.]]])
        force = torch.tensor([[100., 100.]])
        grip = metrics.contact_grip_reward(velocity, force, 2., 0.1)
        self.assertTrue(torch.isfinite(grip).all())
        torch.testing.assert_close(grip, torch.zeros(1))

    def test_action_mapping_matches_existing_contract(self):
        action = make_action()
        for magnitude in (-20., -3., -1., 0., 1., 3., 20.):
            action.process_actions(torch.full((3, 12), magnitude))
            expected = [min(max(contract.ACTION_OFFSETS[name] + magnitude * contract.ACTION_SCALES[name],
                                contract.ACTION_CLIPS[name][0]), contract.ACTION_CLIPS[name][1])
                        for name in action._joint_names]
            torch.testing.assert_close(action.processed_actions, torch.tensor(expected).expand(3, -1))

    def test_raw_policy_action_is_retained_before_the_runner_clip(self):
        action = make_action()
        raw = torch.full((3, 12), 2. * RUNNER_ACTION_CLIP)
        action.capture_policy_actions(raw)
        action.process_actions(raw.clamp(-RUNNER_ACTION_CLIP, RUNNER_ACTION_CLIP))
        torch.testing.assert_close(action.policy_actions, raw)
        torch.testing.assert_close(action.raw_actions, torch.full((3, 12), RUNNER_ACTION_CLIP))

    def test_partial_reset_resets_target_history(self):
        action = make_action()
        action.process_actions(torch.ones(3, 12))
        previous = action.processed_actions.clone()
        default = action._asset.data.default_joint_pos[:, action._joint_ids]
        action.reset(torch.tensor([1]))
        torch.testing.assert_close(action.processed_actions[1], default[1])
        torch.testing.assert_close(action.unclipped_targets[1], default[1])
        torch.testing.assert_close(action.processed_actions[0], previous[0])
        action.process_actions(torch.zeros(3, 12))
        torch.testing.assert_close(action.previous_targets[1], default[1])
        torch.testing.assert_close(action.previous_unclipped_targets[1], default[1])
        torch.testing.assert_close(action.previous_targets[0], previous[0])

    def test_unclipped_target_delta_prevents_limit_dead_zone(self):
        action = make_action()
        action.process_actions(torch.full((3, 12), 20.))
        action.process_actions(torch.full((3, 12), 21.))
        torch.testing.assert_close(action.processed_actions - action.previous_targets, torch.zeros(3, 12))
        torch.testing.assert_close(
            action.unclipped_targets - action.previous_unclipped_targets,
            torch.tensor([contract.ACTION_SCALES[name] for name in action._joint_names]).expand(3, -1),
        )
        reach = action_limit_reach(0.01)
        self.assertTrue(all(21. > reach[name][1] for name in action._joint_names))

    def test_nonfinite_action_holds_previous_target(self):
        action = make_action()
        action.process_actions(torch.zeros(3, 12))
        previous = action.processed_actions.clone()
        action.capture_policy_actions(torch.full((3, 12), float("inf")))
        action.process_actions(torch.full((3, 12), 3.))
        torch.testing.assert_close(action.processed_actions, previous)
        self.assertFalse(torch.isfinite(action.policy_actions).all())

    def test_zero_action_commands_the_standing_pose(self):
        action = make_action()
        action.process_actions(torch.zeros(3, 12))
        default = action._asset.data.default_joint_pos[:, action._joint_ids]
        torch.testing.assert_close(action.processed_actions, default)
        for name in action._joint_names:
            self.assertEqual(contract.ACTION_OFFSETS[name], DEFAULT_JOINT_POS[name])

    def test_runner_clip_reaches_every_mechanical_limit(self):
        reach = action_limit_reach(0.01)
        worst = max(max(abs(low), abs(high)) for low, high in reach.values())
        self.assertLessEqual(worst, RUNNER_ACTION_CLIP)
        action = make_action()
        for sign in (-1., 1.):
            action.process_actions(torch.full((3, 12), sign * RUNNER_ACTION_CLIP))
            index = 0 if sign < 0. else 1
            expected = [contract.ACTION_CLIPS[name][index] for name in action._joint_names]
            torch.testing.assert_close(action.processed_actions, torch.tensor(expected).expand(3, -1))

    def test_action_fence_log_uses_the_contract_reach_not_unit_actions(self):
        env = make_env()
        state = metrics.balance_metrics(env)
        reach = action_limit_reach(0.01)
        term = env.action_manager.get_term("joint_pos")
        for index, name in enumerate(term._joint_names):
            torch.testing.assert_close(state.action_fence[0, index],
                                       torch.tensor(reach[name]), atol=1.e-5, rtol=0.)
        self.assertGreater(float(state.action_fence[0, term._joint_names.index("l_hip_yaw_joint"), 1]), 13.)
        near = min(reach[name][1] for name in term._joint_names)
        self.assertGreater(near, 1.7)
        logged = make_env()
        logged.common_step_counter = 11
        term = logged.action_manager.get_term("joint_pos")
        term.policy_actions = torch.full((3, 12), 5.)
        log = metrics.balance_metrics(logged).log
        for name in term._joint_names:
            expected = 1. if 5. > reach[name][1] else 0.
            self.assertEqual(float(log[f"ActionLimitUpper/{name}"]), expected, name)
        self.assertEqual(float(log["ActionLimitUpper/l_hip_yaw_joint"]), 0.)
        self.assertEqual(float(log["ActionLimitUpper/l_hip_roll_joint"]), 1.)

    def test_hip_roll_near_fence_moved_away_from_one_action_unit(self):
        reach = action_limit_reach(0.01)
        self.assertAlmostEqual(reach["l_hip_roll_joint"][1], 1.7752, places=3)
        self.assertAlmostEqual(reach["r_hip_roll_joint"][0], -1.7752, places=3)

    def test_joint_limit_reward_halves_one_half_error_past_the_soft_limit(self):
        env = make_env()
        robot = env.scene["robot"]
        torch.testing.assert_close(rewards.balance_joint_limit(env, 0.05), torch.ones(3))
        beyond = make_env()
        beyond.common_step_counter = 7
        robot = beyond.scene["robot"]
        index = robot.joint_names.index("l_hip_roll_joint")
        soft_upper = robot.data.soft_joint_pos_limits[0, index, 1]
        robot.data.joint_pos[:, index] = soft_upper + 0.05
        state = metrics.balance_metrics(beyond)
        column = contract.LEG_JOINTS.index("l_hip_roll_joint")
        self.assertAlmostEqual(float(state.values["joint_limit_excess"][0, column]), 0.05, places=6)
        torch.testing.assert_close(rewards.balance_joint_limit(beyond, 0.05),
                                   torch.full((3,), 0.5), atol=1.e-6, rtol=0.)
        self.assertAlmostEqual(float(state.log["Balance/joint_limit_excess_max_deg"]),
                               math.degrees(0.05), places=4)

    def test_joint_limit_reward_sees_the_lower_stop_too(self):
        env = make_env()
        robot = env.scene["robot"]
        index = robot.joint_names.index("l_hip_roll_joint")
        soft_lower = robot.data.soft_joint_pos_limits[0, index, 0]
        robot.data.joint_pos[:, index] = soft_lower - 0.05
        column = contract.LEG_JOINTS.index("l_hip_roll_joint")
        state = metrics.balance_metrics(env)
        self.assertAlmostEqual(float(state.values["joint_limit_excess"][0, column]), 0.05, places=6)
        torch.testing.assert_close(rewards.balance_joint_limit(env, 0.05),
                                   torch.full((3,), 0.5), atol=1.e-6, rtol=0.)

    def test_joint_limit_reward_is_full_inside_the_soft_band(self):
        env = make_env()
        robot = env.scene["robot"]
        index = robot.joint_names.index("l_knee_pitch_joint")
        soft = robot.data.soft_joint_pos_limits[0, index]
        robot.data.joint_pos[:, index] = soft[1] - 0.01
        torch.testing.assert_close(rewards.balance_joint_limit(env, 0.05), torch.ones(3))

    def test_termination_cost_is_independent_of_dt(self):
        for dt in (0.02, 0.01):
            env = types.SimpleNamespace(step_dt=dt, termination_manager=types.SimpleNamespace(terminated=torch.tensor([True, False])))
            torch.testing.assert_close(-dt * rewards.balance_termination_cost(env), torch.tensor([-1., 0.]))

    def test_passive_joint_divergence_is_detected(self):
        robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_vel=torch.zeros(3, 32)))
        robot.data.joint_vel[0, 31] = 101.
        robot.data.joint_vel[1, 30] = float("nan")
        env = types.SimpleNamespace(scene={"robot": robot})
        torch.testing.assert_close(terminations.unstable_joint_vel(env, 100.), torch.tensor([True, True, False]))

    def test_invalid_or_terminated_states_receive_no_tracking_reward(self):
        state = types.SimpleNamespace(values={"valid": torch.tensor([True, False, True]), "tilt": torch.zeros(3, 1)})
        env = types.SimpleNamespace(termination_manager=types.SimpleNamespace(terminated=torch.tensor([False, False, True])))
        with patch.object(rewards, "balance_metrics", return_value=state):
            torch.testing.assert_close(rewards.balance_tracking(env, "tilt", (0.1,)), torch.tensor([1., 0., 0.]))

    def test_stance_reference_matches_current_urdf(self):
        root = ET.parse(contract.DESCRIPTION_ROOT / "urdf/robonex.urdf").getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}

        def transform(link):
            if link not in joints:
                return torch.eye(4, dtype=torch.float64)
            joint = joints[link]
            origin = joint.find("origin")
            r, p, y = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            rx = torch.tensor([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]], dtype=torch.float64)
            ry = torch.tensor([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]], dtype=torch.float64)
            rz = torch.tensor([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]], dtype=torch.float64)
            local = torch.eye(4, dtype=torch.float64)
            local[:3, :3] = rz @ ry @ rx
            local[:3, 3] = torch.tensor([float(v) for v in origin.get("xyz", "0 0 0").split()], dtype=torch.float64)
            return transform(joint.find("parent").get("link")) @ local

        sole = [(transform(name) @ torch.tensor((*offset, 1.), dtype=torch.float64))[:3]
                for name, offset in zip(contract.FEET, contract.FOOT_SOLE_OFFSETS)]
        torch.testing.assert_close((sole[0] - sole[1])[:2], torch.tensor(contract.STANCE_DELTA_XY, dtype=torch.float64), atol=1.e-9, rtol=0.)
        mass = sum(float(link.find("inertial/mass").get("value")) for link in root.findall("link"))
        self.assertAlmostEqual(mass * 9.81, contract.ROBOT_WEIGHT_N, places=6)

    def test_closed_loop_default_pose_matches_action_contract(self):
        pose = contract.CLOSED_LOOP_DEFAULT_JOINT_POS
        self.assertEqual(len(pose), 32)
        self.assertEqual(set(contract.LEG_JOINTS), set(DEFAULT_JOINT_POS))
        for name in contract.LEG_JOINTS:
            self.assertAlmostEqual(pose[name], DEFAULT_JOINT_POS[name])
            self.assertAlmostEqual(pose[name], contract.ACTION_OFFSETS[name])
        self.assertEqual(len(set(pose) - set(contract.LEG_JOINTS)), 20)
        self.assertAlmostEqual(contract.BASE_HEIGHT, 1.0710)
        self.assertEqual(contract.PELVIS_OFFSET_XY, (-0.053545, 0.0002))

    def test_joint_limit_half_error_fits_inside_the_soft_to_hard_band(self):
        params = next(values for name, _, _, values in reward_terms() if name == "joint_limit")
        narrowest = min(0.5 * (upper - lower) * (1. - SOFT_JOINT_POS_LIMIT_FACTOR)
                        for lower, upper in (JOINT_LIMITS_BY_NAME[name] for name in contract.LEG_JOINTS))
        self.assertGreater(params["half_error"], 0.)
        self.assertLessEqual(params["half_error"], narrowest)

    def test_target_delta_reference_stays_within_the_slowest_actuator(self):
        params = next(values for name, _, _, values in reward_terms() if name == "target_delta")
        policy_hz = 50.
        slowest = min(DEFAULT_VELOCITY_LIMIT_CURRENT.values())
        self.assertGreater(params["reference"], 0.)
        self.assertLessEqual(params["reference"] * policy_hz, slowest)
        self.assertLess(params["reference"], min(ACTION_SCALE_RAD.values()))

    def test_env_cfg_declares_the_soft_joint_limit_band_used_by_the_reward(self):
        tree = ast.parse((TASK / "robonex_balancing_env_cfg.py").read_text())
        factors = [ast.literal_eval(node.value) for node in ast.walk(tree)
                   if isinstance(node, ast.keyword) and node.arg == "soft_joint_pos_limit_factor"]
        self.assertEqual(factors, [SOFT_JOINT_POS_LIMIT_FACTOR])

    def test_ppo_cfg_takes_the_action_clip_from_the_shared_contract(self):
        tree = ast.parse((TASK / "agents/rsl_rl_ppo_cfg.py").read_text())
        assigned = [node.value.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Name)
                    and any(isinstance(target, ast.Name) and target.id == "clip_actions"
                            for target in node.targets)]
        self.assertEqual(assigned, ["RUNNER_ACTION_CLIP"])

    def test_env_cfg_standing_pose_matches_the_action_offsets(self):
        tree = ast.parse((TASK / "robonex_balancing_env_cfg.py").read_text())
        node = next(kw.value for kw in ast.walk(tree)
                    if isinstance(kw, ast.keyword) and kw.arg == "joint_pos")
        self.assertIsInstance(node, ast.Name)
        self.assertEqual(node.id, "CLOSED_LOOP_DEFAULT_JOINT_POS")
        pose = contract.CLOSED_LOOP_DEFAULT_JOINT_POS
        self.assertEqual(len(pose), 32)
        for name, angle in DEFAULT_JOINT_POS.items():
            self.assertAlmostEqual(angle, contract.ACTION_OFFSETS[name], places=9)
            self.assertAlmostEqual(pose[name], angle, places=9)

    def test_reward_configuration_signatures_and_weights(self):
        cfg = rewards_cfg_class()
        total_positive = 0.
        names = []
        weights = []
        for node in cfg.body:
            if not isinstance(node, ast.Assign):
                continue
            names.append(node.targets[0].id)
            weights.append(ast.literal_eval({kw.arg: kw.value for kw in node.value.keywords}["weight"]))
            kwargs = {kw.arg: kw.value for kw in node.value.keywords}
            function = getattr(rewards, kwargs["func"].attr)
            params = kwargs.get("params")
            param_names = set() if params is None else {key.value for key in params.keys}
            self.assertEqual(set(inspect.signature(function).parameters) - {"env"}, param_names)
            weight = ast.literal_eval(kwargs["weight"])
            total_positive += max(weight, 0.)
        self.assertEqual(len(names), 13)
        self.assertNotIn("alive", names)
        self.assertNotIn("joint_deviation", names)
        for required in ("foot_yaw", "pelvis", "symmetry"):
            self.assertIn(required, names)
        self.assertAlmostEqual(total_positive, 1.)
        self.assertEqual([name for name, weight in zip(names, weights) if weight < 0.], ["terminating"])

    def test_every_running_term_stays_in_the_unit_interval(self):
        env = make_env()
        for name, func, _, params in reward_terms():
            if name == "terminating":
                continue
            value = getattr(rewards, func)(env, **params)
            self.assertTrue((value >= 0.).all(), f"{name} produced a negative reward")
            self.assertTrue((value <= 1.).all(), f"{name} exceeded 1.0")

    def test_running_reward_cannot_be_negative_in_a_broken_state(self):
        env = make_env()
        data = env.scene["robot"].data
        data.projected_gravity_b = torch.tensor([[0.7, 0.5, -0.51]]).repeat(3, 1)
        data.root_lin_vel_w = torch.full((3, 3), 9.)
        data.root_ang_vel_b = torch.full((3, 3), 9.)
        data.applied_torque = torch.full((3, 13), 60.)
        env.action_manager.get_term("joint_pos").policy_actions = torch.full((3, 12), 30.)
        data.joint_pos = torch.full((3, 13), 9.)
        total = 0.
        for name, func, weight, params in reward_terms():
            if name == "terminating":
                continue
            total = total + weight * getattr(rewards, func)(env, **params)
        self.assertTrue((total >= 0.).all())

    def test_metrics_resolve_motor_torque_by_name_and_exclude_passive(self):
        env = make_env()
        robot = env.scene["robot"]
        for index, name in enumerate(robot.joint_names[:-1]):
            robot.data.applied_torque[:, index] = 2. * contract.RATED_TORQUE_BY_JOINT[name]
        robot.data.applied_torque[:, -1] = 1.e6
        state = metrics.balance_metrics(env)
        torch.testing.assert_close(state.values["torque_ratio"], torch.full((3, 12), 2.))
        torch.testing.assert_close(state.values["stance_error"], torch.zeros(3, 2), atol=1.e-7, rtol=0.)

    def test_metrics_contact_sensor_order_matches_foot_order(self):
        env = make_env()
        env.scene.sensors["contact_forces"].data.net_forces_w_history[:, :, 0, 2] = 0.
        env.scene["robot"].data.body_link_lin_vel_w[:, 2, 0] = 0.1
        torch.testing.assert_close(rewards.balance_foot_grip(env, 0.1, 0.01), torch.full((3,), 0.25))

    def test_absolute_roll_log_does_not_cancel_opposite_tilts(self):
        env = make_env()
        angle = torch.tensor([0.1, -0.1, 0.])
        env.scene["robot"].data.projected_gravity_b = torch.stack((torch.zeros(3), -angle.sin(), -angle.cos()), dim=-1)
        state = metrics.balance_metrics(env)
        self.assertAlmostEqual(state.log["Balance/roll_deg"].item(), 0., places=5)
        self.assertGreater(state.log["Balance/roll_abs_deg"].item(), 3.)
        torch.testing.assert_close(state.log["Balance/stance_width_m"], torch.tensor(0.3194))
        torch.testing.assert_close(state.log["Balance/both_feet_contact_fraction"], torch.tensor(1.))

    def test_metrics_cache_updates_on_next_step_and_rejects_nonfinite(self):
        env = make_env()
        state = metrics.balance_metrics(env)
        self.assertTrue(state.values["valid"].all())
        env.scene["robot"].data.root_pos_w[0, 2] = float("nan")
        self.assertIs(metrics.balance_metrics(env), state)
        self.assertTrue(state.values["valid"].all())
        env.common_step_counter += 1
        self.assertTrue(terminations.balance_invalid_state(env)[0])
        torch.testing.assert_close(rewards.balance_height(env, 1.0789, 0.02, 0.03), torch.tensor([0., 1., 1.]))

    def test_crossed_stance_is_not_rewarded(self):
        env = make_env()
        data = env.scene["robot"].data
        data.body_link_pos_w[:, [0, 2], 1] *= -1.
        self.assertTrue((rewards.balance_tracking(env, "stance_error", (0.03, 0.04)) < 1.e-10).all())

    def test_sole_offsets_match_description_geometry(self):
        tree = ast.parse((contract.DESCRIPTION_ROOT / "scripts/robonex_data.py").read_text())
        node = next(node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "COLLISION_BOX" for target in node.targets))
        boxes = ast.literal_eval(node.value)
        for name, offset in zip(contract.FEET, contract.FOOT_SOLE_OFFSETS):
            size, center = boxes[name]
            self.assertEqual(offset, (center[0], center[1], center[2] - size[2] / 2))

    def test_reference_pose_scores_full_on_new_posture_terms(self):
        env = make_env()
        for reward in (rewards.balance_tracking(env, "pelvis_error", (0.03, 0.02)),
                       rewards.balance_axis_tracking(env, "foot_yaw", (0.1, 0.1)),
                       rewards.balance_symmetry(env, 0.05)):
            torch.testing.assert_close(reward, torch.ones(3), atol=1.e-6, rtol=0.)

    def test_foot_yaw_penalises_twist_and_ignores_whole_body_yaw(self):
        env = make_env()
        data = env.scene["robot"].data
        half = math.cos(math.radians(6.) / 2.), math.sin(math.radians(6.) / 2.)
        data.body_link_quat_w[:, 2] = torch.tensor([half[0], 0., 0., half[1]])
        torch.testing.assert_close(metrics.balance_metrics(env).values["foot_yaw"][:, 0],
                                   torch.full((3,), math.radians(6.)), atol=1.e-6, rtol=0.)
        torch.testing.assert_close(rewards.balance_axis_tracking(env, "foot_yaw", (math.radians(6.),) * 2),
                                   torch.full((3,), 0.75), atol=1.e-6, rtol=0.)
        turned = make_env()
        turned.common_step_counter = 2
        spin = math.cos(0.6), math.sin(0.6)
        turned.scene["robot"].data.root_quat_w = torch.tensor([[spin[0], 0., 0., spin[1]]]).repeat(3, 1)
        turned.scene["robot"].data.body_link_quat_w[:] = torch.tensor([spin[0], 0., 0., spin[1]])
        torch.testing.assert_close(rewards.balance_axis_tracking(turned, "foot_yaw", (0.1, 0.1)),
                                   torch.ones(3), atol=1.e-6, rtol=0.)

    def test_mirror_pairs_match_action_offset_convention(self):
        self.assertEqual(len(contract.MIRROR_JOINT_PAIRS), 6)
        for left, right in contract.MIRROR_JOINT_PAIRS:
            self.assertTrue(left.startswith("l_") and right.startswith("r_"))
            self.assertIn(right, contract.LEG_JOINTS)
            self.assertAlmostEqual(contract.ACTION_OFFSETS[left], -contract.ACTION_OFFSETS[right], places=9)
            self.assertAlmostEqual(contract.ACTION_SCALES[left], contract.ACTION_SCALES[right], places=9)
            low, high = contract.ACTION_CLIPS[left]
            self.assertAlmostEqual(low, -contract.ACTION_CLIPS[right][1], places=9)
            self.assertAlmostEqual(high, -contract.ACTION_CLIPS[right][0], places=9)

    def test_symmetry_error_is_the_mirrored_joint_sum(self):
        env = make_env()
        robot = env.scene["robot"]
        left, right = contract.MIRROR_JOINT_PAIRS[0]
        robot.data.joint_pos[:, robot.joint_names.index(left)] = 0.10
        robot.data.joint_pos[:, robot.joint_names.index(right)] = 0.10
        error = metrics.balance_metrics(env).values["symmetry_error"]
        self.assertAlmostEqual(float(error[0, 0]), 0.20, places=6)
        torch.testing.assert_close(error[:, 1:], torch.zeros(3, 5), atol=1.e-7, rtol=0.)
        mirrored = make_env()
        mirrored.common_step_counter = 3
        robot = mirrored.scene["robot"]
        robot.data.joint_pos[:, robot.joint_names.index(left)] = 0.10
        robot.data.joint_pos[:, robot.joint_names.index(right)] = -0.10
        torch.testing.assert_close(rewards.balance_symmetry(mirrored, math.radians(3.)),
                                   torch.ones(3), atol=1.e-6, rtol=0.)

    def test_pelvis_error_tracks_lateral_shift(self):
        env = make_env()
        env.scene["robot"].data.root_pos_w[:, 1] += 0.02
        reward = rewards.balance_tracking(env, "pelvis_error", (0.03, 0.02))
        torch.testing.assert_close(reward, torch.full((3,), 0.5), atol=1.e-6, rtol=0.)

    def test_wrapper_preserves_raw_action_and_does_not_mutate_prior_log(self):
        class BaseWrapper:
            def step(self, actions):
                env = self.unwrapped
                env.action_manager.get_term("joint_pos").process_actions(actions.clamp(-3., 3.))
                env.common_step_counter += 1
                metrics.balance_metrics(env)
                return None, None, None, self.extras

        stub = types.ModuleType("isaaclab_rl.rsl_rl")
        stub.RslRlVecEnvWrapper = BaseWrapper
        with patch.dict(sys.modules, {stub.__name__: stub}):
            module = load_module("balance_test_wrapper", ROOT / "scripts/rsl_rl/balancing_wrapper.py")
        wrapper = module.BalancingVecEnvWrapper()
        wrapper.unwrapped = make_env()
        wrapper.extras = {"log": {"existing": 1.}}
        _, _, _, extras = wrapper.step(torch.full((3, 12), 20.))
        previous_log = extras["log"]
        self.assertEqual(previous_log["ActionLimitLower/l_hip_yaw_joint"], 0.)
        self.assertEqual(previous_log["ActionLimitUpper/l_hip_yaw_joint"], 1.)
        self.assertEqual(previous_log["ActionMean/l_hip_yaw_joint"], 20.)
        torch.testing.assert_close(wrapper.unwrapped.action_manager.get_term("joint_pos").policy_actions,
                                   torch.full((3, 12), 20.))
        _, _, _, extras = wrapper.step(torch.zeros(3, 12))
        self.assertIsNot(extras["log"], previous_log)
        self.assertEqual(previous_log["ActionLimitUpper/l_hip_yaw_joint"], 1.)
        self.assertEqual(extras["log"]["ActionLimitUpper/l_hip_yaw_joint"], 0.)
        self.assertEqual(extras["log"]["ActionMean/l_hip_yaw_joint"], 0.)


if __name__ == "__main__":
    unittest.main()

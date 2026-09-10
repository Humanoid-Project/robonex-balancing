import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class BalanceJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.policy_actions = torch.zeros_like(self.raw_actions)
        self.previous_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self.unclipped_targets = self.previous_targets.clone()
        self.previous_unclipped_targets = self.previous_targets.clone()
        self._processed_actions.copy_(self.previous_targets)
        self._policy_actions_pending = False

    def capture_policy_actions(self, actions):
        self.policy_actions.copy_(actions.detach())
        self._policy_actions_pending = True

    def process_actions(self, actions):
        if not self._policy_actions_pending:
            self.policy_actions.copy_(actions)
        self._policy_actions_pending = False
        self.previous_targets.copy_(self.processed_actions)
        self.previous_unclipped_targets.copy_(self.unclipped_targets)
        valid = torch.isfinite(self.policy_actions).all(dim=1, keepdim=True)
        valid &= torch.isfinite(actions).all(dim=1, keepdim=True)
        hold_action = (self.previous_unclipped_targets - self._offset) / self._scale
        safe_actions = torch.where(valid, actions, hold_action)
        self.unclipped_targets.copy_(safe_actions * self._scale + self._offset)
        super().process_actions(safe_actions)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        default_targets = self._asset.data.default_joint_pos[ids][:, self._joint_ids]
        self._processed_actions[ids] = default_targets
        self.previous_targets[ids] = default_targets
        self.unclipped_targets[ids] = default_targets
        self.previous_unclipped_targets[ids] = default_targets
        self.policy_actions[ids] = 0.0

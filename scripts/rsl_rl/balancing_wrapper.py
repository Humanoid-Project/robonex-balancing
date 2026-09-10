from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class BalancingVecEnvWrapper(RslRlVecEnvWrapper):
    def step(self, actions):
        self.unwrapped.action_manager.get_term("joint_pos").capture_policy_actions(actions)
        observations, rewards, dones, extras = super().step(actions)
        metrics = getattr(self.unwrapped, "_balance_metrics", None)
        if metrics is not None:
            extras["log"] = {**extras.get("log", {}), **metrics.log}
        return observations, rewards, dones, extras

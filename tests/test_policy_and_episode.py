import numpy as np
import torch

import rl_train
from rl_train import (
    DOWN,
    MLP,
    SCREEN_H,
    SCREEN_W,
    UP,
    load_policy,
    play_episode,
    select_action,
)


class ConstantPolicy(torch.nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        return torch.tensor([[self.p]], dtype=torch.float32)


def test_select_action_uses_epsilon_random_branch(monkeypatch):
    obs = np.zeros((SCREEN_H, SCREEN_W), dtype=np.float32)
    policy = ConstantPolicy(1.0)

    monkeypatch.setattr(rl_train.np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(rl_train.np.random, "choice", lambda _: DOWN)

    action = select_action(policy, obs, epsilon=0.5)
    assert action == DOWN


def test_select_action_policy_branch_returns_up_when_prob_exceeds_rand(monkeypatch):
    obs = np.zeros((SCREEN_H, SCREEN_W), dtype=np.float32)
    policy = ConstantPolicy(0.8)

    values = iter([0.9, 0.2])
    monkeypatch.setattr(rl_train.np.random, "rand", lambda: next(values))

    action = select_action(policy, obs, epsilon=0.0)
    assert action == UP


def test_play_episode_returns_traj_and_terminal_reward_when_env_finishes(monkeypatch):
    class TerminalEnv:
        def reset(self):
            return np.zeros((SCREEN_H, SCREEN_W), dtype=np.float32)

        def step(self, *_):
            obs_next = np.ones((SCREEN_H, SCREEN_W), dtype=np.float32)
            return obs_next, 1.0, -1.0, True

    monkeypatch.setattr(rl_train, "select_action", lambda *args, **kwargs: UP)

    traj, terminal_reward, done = play_episode(
        MLP(), MLP(), TerminalEnv(), max_steps=10
    )

    assert len(traj) == 1
    assert traj[0][1] == UP
    assert terminal_reward == 1.0
    assert done is True


def test_play_episode_respects_max_steps_when_no_terminal(monkeypatch):
    class NeverDoneEnv:
        def __init__(self):
            self.steps = 0

        def reset(self):
            return np.zeros((SCREEN_H, SCREEN_W), dtype=np.float32)

        def step(self, *_):
            self.steps += 1
            obs_next = np.zeros((SCREEN_H, SCREEN_W), dtype=np.float32)
            return obs_next, 0.0, 0.0, False

    monkeypatch.setattr(rl_train, "select_action", lambda *args, **kwargs: DOWN)

    env = NeverDoneEnv()
    traj, terminal_reward, done = play_episode(MLP(), MLP(), env, max_steps=7)

    assert len(traj) == 7
    assert env.steps == 7
    assert terminal_reward == 0.0
    assert done is False


def test_play_episode_with_zero_max_steps_returns_empty_traj():
    class AnyEnv:
        def reset(self):
            return np.zeros((SCREEN_H, SCREEN_W), dtype=np.float32)

        def step(self, *_):
            raise AssertionError("step should not be called when max_steps=0")

    traj, terminal_reward, done = play_episode(MLP(), MLP(), AnyEnv(), max_steps=0)

    assert traj == []
    assert terminal_reward == 0.0
    assert done is False


def test_load_policy_roundtrip(tmp_path):
    model = MLP()
    policy_path = tmp_path / "policy.pth"
    torch.save(model.state_dict(), policy_path)

    loaded = load_policy(str(policy_path))

    assert loaded.training is False

    source_state = model.state_dict()
    loaded_state = loaded.state_dict()
    assert source_state.keys() == loaded_state.keys()
    for key in source_state:
        assert torch.allclose(source_state[key], loaded_state[key])

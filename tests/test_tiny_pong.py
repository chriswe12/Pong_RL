import numpy as np

from rl_train import (
    BALL_SIZE,
    SCREEN_H,
    SCREEN_W,
    STAY,
    TRAIL_LEN,
    UP,
)


def test_reset_returns_normalized_observation_shape_and_type(env):
    obs = env.reset()
    assert obs.shape == (SCREEN_H, SCREEN_W)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() <= 1.0
    assert 0.0 <= obs.max() <= 1.0


def test_step_keeps_paddles_in_bounds_for_repeated_up_down_actions(env):
    env.left.top = 0
    env.step(UP, STAY)
    assert env.left.top == 0

    env.right.bottom = SCREEN_H
    env.step(STAY, DOWN)
    assert env.right.bottom == SCREEN_H


def test_top_bottom_wall_collision_flips_ball_dy(env):
    env.ball_x = SCREEN_W / 2
    env.ball_y = 0
    env.ball_dy = -abs(env.ball_dy)

    env.step(STAY, STAY)

    assert env.ball_dy > 0


def test_left_score_terminal_reward_contract(env):
    env.ball_x = 0
    env.ball_y = SCREEN_H / 2
    env.ball_dx = -1
    env.ball_dy = 0

    _, r_left, r_right, done = env.step(STAY, STAY)

    assert done is True
    assert r_left == -1.0
    assert r_right == 1.0


def test_right_score_terminal_reward_contract(env):
    env.ball_x = SCREEN_W - BALL_SIZE
    env.ball_y = SCREEN_H / 2
    env.ball_dx = 1
    env.ball_dy = 0

    _, r_left, r_right, done = env.step(STAY, STAY)

    assert done is True
    assert r_left == 1.0
    assert r_right == -1.0


def test_ball_trail_is_capped_to_trail_len(env):
    env.ball_dx = 0
    env.ball_dy = 0

    for _ in range(TRAIL_LEN + 5):
        env.step(STAY, STAY)

    assert len(env.ball_trail) == TRAIL_LEN


def test_observation_value_range_is_0_to_1(env):
    obs, *_ = env.step(STAY, STAY)
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0

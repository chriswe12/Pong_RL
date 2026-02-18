import os

import numpy as np
import pytest
import torch

# Ensure pygame can initialize in headless test environments.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from rl_train import TinyPong


@pytest.fixture(autouse=True)
def seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def env():
    return TinyPong()

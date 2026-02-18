import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure pygame can initialize in headless test environments.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Ensure project root is importable in CI/pytest importlib modes.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(autouse=True)
def seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def env():
    from rl_train import TinyPong

    return TinyPong()

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from contextlib import nullcontext

# NOTE: We only force the dummy video driver during training (when this file
# is executed as a script). When imported by the game, we do NOT override
# SDL_VIDEODRIVER so that a normal window can appear.
if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
else:
    # Minimal init so Surface creation works when imported.
    if not pygame.get_init():
        pygame.init()

# Constants for the 80x80 game
SCREEN_W = 80
SCREEN_H = 80
PADDLE_W = 1
PADDLE_H = 10
BALL_SIZE = 1
PADDLE_SPEED = 1
BALL_SPEED = 0.5
TRAIL_LEN = 10  # number of past ball positions to overlay

UP = -1
STAY = 0
DOWN = 1


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    obs_next: np.ndarray
    done: bool


class TinyPong:
    """Minimal 80x80 Pong environment for self-play training.
    Observation is the raw 80x80 grayscale image (uint8 -> float32 [0,1]).
    Two paddles: left uses external action, right mirrors left action (self-play baseline).
    """

    def __init__(self):
        self.surface = pygame.Surface((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.left = pygame.Rect(5, SCREEN_H // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
        self.right = pygame.Rect(
            SCREEN_W - 5 - PADDLE_W, SCREEN_H // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H
        )
        self.ball_x = SCREEN_W // 2 - BALL_SIZE // 2
        self.ball_y = SCREEN_H // 2 - BALL_SIZE // 2
        # randomize initial direction a bit
        self.ball_dx = BALL_SPEED * (1 if np.random.rand() < 0.5 else -1)
        self.ball_dy = BALL_SPEED * (1 if np.random.rand() < 0.5 else -1)
        # trail holds tuples (x,y)
        self.ball_trail = [(int(self.ball_x), int(self.ball_y))]
        return self._get_obs()

    def step(self, action_left: int, action_right: int):
        # Apply actions
        if action_left == UP and self.left.top > 0:
            self.left.y -= PADDLE_SPEED
        elif action_left == DOWN and self.left.bottom < SCREEN_H:
            self.left.y += PADDLE_SPEED

        if action_right == UP and self.right.top > 0:
            self.right.y -= PADDLE_SPEED
        elif action_right == DOWN and self.right.bottom < SCREEN_H:
            self.right.y += PADDLE_SPEED

        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        ball_rect = pygame.Rect(
            int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE
        )

        # Update trail
        self.ball_trail.append((ball_rect.x, ball_rect.y))
        if len(self.ball_trail) > TRAIL_LEN:
            self.ball_trail.pop(0)

        # Collisions
        if ball_rect.top <= 0 or ball_rect.bottom >= SCREEN_H:
            self.ball_dy *= -1

        if ball_rect.colliderect(self.left) or ball_rect.colliderect(self.right):
            self.ball_dx *= -1

        reward_left = 0.0
        reward_right = 0.0
        done = False

        # Scoring
        if ball_rect.left <= 0:
            reward_left = -1.0
            reward_right = 1.0
            done = True
        elif ball_rect.right >= SCREEN_W:
            reward_left = 1.0
            reward_right = -1.0
            done = True

        obs = self._get_obs()
        return obs, reward_left, reward_right, done

    def _get_obs(self) -> np.ndarray:
        # Render to surface (black background, white objects)
        self.surface.fill((0, 0, 0))
        # Draw trail in dim gray (directional cue)
        for tx, ty in self.ball_trail[:-1]:  # exclude current position
            pygame.draw.rect(
                self.surface, (128, 128, 128), (tx, ty, BALL_SIZE, BALL_SIZE)
            )
        pygame.draw.rect(self.surface, (255, 255, 255), self.left)
        pygame.draw.rect(self.surface, (255, 255, 255), self.right)
        pygame.draw.rect(
            self.surface,
            (255, 255, 255),
            (int(self.ball_x), int(self.ball_y), BALL_SIZE, BALL_SIZE),
        )
        # Convert to numpy 80x80 grayscale
        arr = pygame.surfarray.array3d(self.surface)  # (W,H,3)
        arr = arr.transpose(1, 0, 2)  # (H,W,3)
        gray = arr[..., 0]  # channel is identical (B/W)
        gray = gray.astype(np.float32) / 255.0
        return gray


class MLP(nn.Module):
    def __init__(self, input_dim=SCREEN_W * SCREEN_H, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def select_action(
    policy: MLP,
    obs: np.ndarray,
    epsilon: float = 0.0,
    device: torch.device | None = None,
    use_amp: bool = False,
) -> int:
    """Return an action using epsilon-greedy over the policy's Bernoulli output.
    With probability epsilon choose a random action (exploration); otherwise sample
    according to the policy's probability of moving UP.
    """
    if np.random.rand() < epsilon:
        return np.random.choice([UP, DOWN])
    with torch.no_grad():
        x = torch.from_numpy(obs.reshape(1, -1)).to(device or "cpu")
        # AMP not necessary for inference here, but safe to leave disabled
        prob_up = policy(x).item()  # in [0,1]
    return UP if np.random.rand() < prob_up else DOWN


def play_episode(
    policy_left: MLP,
    policy_right: MLP,
    env: TinyPong,
    max_steps=500,
    epsilon_left: float = 0.0,
    epsilon_right: float = 0.0,
    device: torch.device | None = None,
    use_amp: bool = False,
):
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a_left = select_action(
            policy_left, obs, epsilon=epsilon_left, device=device, use_amp=use_amp
        )
        a_right = select_action(
            policy_right, obs, epsilon=epsilon_right, device=device, use_amp=use_amp
        )
        obs_next, r_left, r_right, done = env.step(a_left, a_right)
        traj.append((obs, a_left, r_left))
        obs = obs_next
        if done:
            break
    return traj, r_left  # return terminal reward as return signal


def train_self_play(
    episodes=2000,
    lr=1e-3,
    gamma=0.99,
    save_path="policy.pth",
    seed=0,
    eps_start: float = 0.3,
    eps_end: float = 0.01,
    device: str = "auto",
    amp: bool = False,
    compile_model: bool = False,
    max_steps: int = 500,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    env = TinyPong()
    # Resolve device
    if device == "auto":
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(device)

    policy_left = MLP().to(dev)
    policy_right = MLP().to(dev)

    if compile_model and hasattr(torch, "compile"):
        try:
            policy_left = torch.compile(policy_left)
            policy_right = torch.compile(policy_right)
        except Exception:
            pass

    optimizer_left = optim.Adam(policy_left.parameters(), lr=lr)

    # AMP setup
    use_amp = amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    for ep in range(episodes):
        # Linear annealing of epsilon
        frac = ep / max(1, episodes - 1)
        epsilon = eps_start + (eps_end - eps_start) * frac
        traj, R = play_episode(
            policy_left,
            policy_right,
            env,
            max_steps=max_steps,
            epsilon_left=epsilon,
            epsilon_right=0.0,
            device=dev,
            use_amp=use_amp,
        )
        # Simple REINFORCE using terminal reward R for all steps
        if len(traj) == 0:
            continue
        returns = R  # same for all steps (sparse)

        # Compute loss
        logps = []
        for obs, action, _ in traj:
            x = torch.from_numpy(obs.reshape(1, -1)).to(dev)
            with autocast_ctx():
                prob_up = policy_left(x)
                prob_up = torch.clamp(prob_up, 1e-5, 1 - 1e-5)
                # Bernoulli logprob
                if action == UP:
                    logp = torch.log(prob_up)
                else:
                    logp = torch.log(1 - prob_up)
            logps.append(logp)
        logps = torch.cat(logps)
        with autocast_ctx():
            loss = -(returns * logps).mean()

        optimizer_left.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer_left)
            scaler.update()
        else:
            loss.backward()
            optimizer_left.step()

        if (ep + 1) % 100 == 0:
            if dev.type == "cuda":
                dev_name = torch.cuda.get_device_name(0)
            else:
                dev_name = "cpu"
            print(
                f"Episode {ep + 1}/{episodes} | R={R:.2f} | loss={loss.item():.4f} | eps={epsilon:.4f} | device={dev.type}:{dev_name}"
            )

    # Save left policy
    torch.save(policy_left.state_dict(), save_path)
    print(f"Saved policy to {save_path}")


def load_policy(path: str) -> MLP:
    policy = MLP()
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save", type=str, default="policy.pth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps_start", type=float, default=0.3)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument(
        "--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable CUDA AMP mixed precision"
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model if available"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    args = parser.parse_args()

    # Ensure pygame is initialized for headless rendering in training
    if not pygame.get_init():
        pygame.init()

    train_self_play(
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        save_path=args.save,
        seed=args.seed,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        device=args.device,
        amp=args.amp,
        compile_model=args.compile,
        max_steps=args.max_steps,
    )

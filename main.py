import pygame
import sys
import torch
import numpy as np
from rl_train import load_policy, TinyPong, UP, DOWN, STAY
import os

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Screen dimensions
SCREEN_WIDTH = 80
SCREEN_HEIGHT = 80
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 800
screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption("Pong Clone")

# Use the exact same environment as training
env = TinyPong()

# Scores
left_score = 0
right_score = 0

# Font
font = pygame.font.Font(None, 36)

# Load policy if available
policy_path = "policy.pth"
policy = None
if policy_path and os.path.exists(policy_path):
    try:
        policy = load_policy(policy_path)
        print("Loaded policy for right paddle.")
    except Exception as e:
        print(f"Could not load policy: {e}")

# Game loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Observation for policy (same as training)
    obs = env.get_obs()

    # Player controls -> action
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        a_left = UP
    elif keys[pygame.K_s]:
        a_left = DOWN
    else:
        a_left = STAY

    # Policy action for right paddle (if loaded)
    if policy is not None:
        with torch.no_grad():
            x = torch.from_numpy(obs.reshape(1, -1))
            prob_up = policy(x).item()
        a_right = UP if np.random.rand() < prob_up else DOWN
    else:
        a_right = STAY

    # Step environment (applies movement & physics identical to training)
    obs_next, r_left, r_right, done = env.step(a_left, a_right)
    if done:
        if r_left > r_right:
            left_score += 1
        elif r_right > r_left:
            right_score += 1
        env.reset()

    # Draw everything
    # env.get_obs() already rendered onto env.surface. Scale it for display.
    scaled_surface = pygame.transform.scale(
        env.surface, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )
    screen.blit(scaled_surface, (0, 0))

    # Draw scores outside the game area
    left_text = font.render(str(left_score), True, WHITE)
    right_text = font.render(str(right_score), True, WHITE)
    screen.blit(left_text, (DISPLAY_WIDTH // 4, 20))
    screen.blit(right_text, (3 * DISPLAY_WIDTH // 4, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()

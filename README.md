# Pong RL (80x80 Pixel Environment)

Minimal Pong clone plus self-play reinforcement learning with a tiny MLP:

Architecture: 80*80 grayscale input -> Linear(6400->200) -> ReLU -> Linear(200->1) -> Sigmoid (probability of UP).

Features:
- Shared TinyPong environment for both training and gameplay (identical physics).
- Pixel-level low-fidelity 80x80 with ball trail for directional cue.
- Epsilon-greedy exploration with linear annealing.
- Deterministic or stochastic play sampling Bernoulli in the game.

## Files
- `rl_train.py`: Environment + training loop + model + policy load.
- `main.py`: Uses the same `TinyPong` env for human vs policy.
- `policy.pth`: Saved weights after training (ignored by git).

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train
```powershell
python rl_train.py --episodes 2000 --save policy.pth --eps_start 0.3 --eps_end 0.01
```

## Play
```powershell
python main.py
```
Controls: W/S for left paddle.

## Create and Push GitHub Repo (if repo doesn't exist yet)
Replace YOURUSER and REPO_NAME accordingly.
```powershell
git init
git add .
git commit -m "Initial commit: Pong RL"
git branch -M main
git remote add origin https://github.com/YOURUSER/REPO_NAME.git
git push -u origin main
```
If the remote doesn't exist yet, create it first on GitHub (web UI: New Repository) or via `gh` CLI:
```powershell
# Install GitHub CLI if needed: winget install --id GitHub.cli
gh auth login
gh repo create YOURUSER/REPO_NAME --public --source . --remote origin --push
```

## Reinforcement Learning Notes
- Reward: terminal +/-1 only. Can add shaping (distance to ball, paddle hits) for faster learning.
- Exploration: epsilon linearly annealed from start to end.
- Potential upgrades: entropy bonus, advantage baseline, CNN instead of MLP, frame stacking.

## License
MIT (add a LICENSE file if you wish).

## Pre-commit Hooks
Install and enable hooks locally:
```powershell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The repo includes standard checks for Python and C++:
- General hygiene: YAML validity, trailing whitespace, EOF newline, merge conflict markers, large files.
- Python: `ruff` lint + auto-fix and `ruff-format`.
- C++: `clang-format` for `*.c, *.cc, *.cpp, *.cxx, *.h, *.hh, *.hpp, *.hxx`.

## CI
GitHub Actions runs on every push to `main` and on pull requests:
- `pre-commit` across the full repo.
- Python smoke check (`python -m compileall`).

## Makefile Targets
Common local commands:
```powershell
make install-dev
make precommit-install
make precommit
make check
make train
make play
```

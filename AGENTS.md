# Repository Guidelines

## Project Structure & Module Organization
- Core RL and environment logic lives in `rl_train.py` (TinyPong env, model, training loop, policy I/O).
- Interactive gameplay entrypoint is `main.py` (human left paddle vs optional trained right policy).
- Unit tests are in `tests/`:
  - `tests/test_tiny_pong.py` for environment physics/reward contracts.
  - `tests/test_policy_and_episode.py` for action selection, episode flow, and policy loading.
  - `tests/conftest.py` for shared fixtures and deterministic seeds.
- CI is defined in `.github/workflows/ci.yml`.

## Build, Test, and Development Commands
- `make install-dev`: install runtime deps plus dev tools (`pre-commit`, `ruff`).
- `make check`: run lint, byte-compile smoke checks, and unit tests.
- `make test`: run `pytest -q`.
- `make train`: start RL training and write `policy.pth`.
- `make play`: run the game loop locally.
- `make precommit`: run all pre-commit hooks exactly as CI does.

## Coding Style & Naming Conventions
- Python style is enforced with `ruff` (`ruff check .`) and `ruff format .`.
- Use 4-space indentation and keep functions focused and testable.
- Naming:
  - `snake_case` for functions/variables.
  - `UPPER_SNAKE_CASE` for constants (`SCREEN_W`, `BALL_SPEED`).
  - `PascalCase` for classes (`TinyPong`, `MLP`).
- Keep module-level side effects minimal, except required `pygame` init behavior.

## Testing Guidelines
- Framework: `pytest`.
- Add unit tests under `tests/` named `test_*.py`; test functions should be named `test_*`.
- Prefer deterministic tests: seed RNGs and monkeypatch randomness where needed.
- Cover edge behavior (bounds, collisions, terminal rewards), not only happy paths.
- Run locally with `make test` (or `python -m pytest -q`).

## Commit & Pull Request Guidelines
- Existing history uses short, imperative messages (examples: `Add pre-commit hooks...`, `added gpu support`).
- Prefer: `<Verb> <what changed>` with clear scope (e.g., `Add unit tests for TinyPong collisions`).
- PRs should include:
  - concise summary of behavior changes,
  - test evidence (`make check` output or equivalent),
  - linked issue/context when applicable.
- Ensure CI jobs pass before requesting review.

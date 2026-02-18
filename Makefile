.PHONY: help install-dev precommit-install precommit lint format check smoke test train play

PYTHON ?= python3
PIP ?= pip3

help:
	@echo "Available targets:"
	@echo "  install-dev        Install project + dev tooling deps"
	@echo "  precommit-install  Install git pre-commit hooks"
	@echo "  precommit          Run pre-commit on all files"
	@echo "  lint               Run Python lint checks (ruff)"
	@echo "  format             Format Python code (ruff-format)"
	@echo "  check              Run lint + smoke checks"
	@echo "  smoke              Byte-compile Python sources"
	@echo "  test               Run unit tests (pytest)"
	@echo "  train              Run RL training script"
	@echo "  play               Run game"

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pre-commit ruff

precommit-install:
	pre-commit install

precommit:
	pre-commit run --all-files --show-diff-on-failure --color=always

lint:
	ruff check .

format:
	ruff format .

smoke:
	$(PYTHON) -m compileall -q .

test:
	$(PYTHON) -m pytest -q

check: lint smoke test

train:
	$(PYTHON) rl_train.py --episodes 2000 --save policy.pth --eps_start 0.3 --eps_end 0.01

play:
	$(PYTHON) main.py

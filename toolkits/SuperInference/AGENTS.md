# Repository Guidelines

## Project Structure & Module Organization
- **manager/** orchestrates action execution (`base_policy_runner.py`, `utils/` chunk managers).
- **consumer/** contains policy connectors, data savers, and replayers; entry points live here (e.g., `diffusion_policy_infer.py`).
- **policy/** houses learnable policy adapters such as `diffusion_policy_umi.py`.
- **devices/** exposes hardware/shm abstractions, while **utils/** provides shared helpers (rerun visualization, logging, calibration).
- **configs/** and root `config_*.yaml` files store runtime presets; **scripts/** includes one-off converters and visualizers.
- Logs, assets, and recordings land in `logs/`, `assests/`, and `recordings/`.

## Build, Test, and Development Commands
- `python consumer/diffusion_policy_infer.py --config configs/...yaml` – run inference against SHM streams.
- `python manager/base_policy_runner.py --config config_policy_*.yaml` – start the action executor/robot bridge.
- `pytest tests` – execute automated test suites (create `tests/` mirroring modules; hook into CI).
- `ruff check .` / `black .` – lint and format Python sources before commits.

## Coding Style & Naming Conventions
- Python modules follow PEP 8 with 4-space indentation; prefer descriptive snake_case for variables/functions and UpperCamelCase for classes.
- Keep SHM/device names consistent with config entries (e.g., `RizonRobot_1_control`).
- Align logging via `utils.logger_config.logger`; include context like device/timestamps.
- Place shared constants/utilities in `utils/`; avoid duplicating helper logic across consumer/manager layers.

## Testing Guidelines
- Add pytest modules under `tests/<package>/test_*.py`; mirror directory names (e.g., `tests/consumer/test_policy_connector.py`).
- Mock SHM handles and rerun logging when possible to avoid hardware dependencies.
- Target coverage for new features: ≥80% of touched lines/functions. Document edge-case scenarios in test docstrings.

## Commit & Pull Request Guidelines
- Commit messages follow short imperative summaries (e.g., `Add rerun logging for raw predictions`) with optional scope prefix.
- Group related changes per commit; avoid mixing policy and device updates unless tightly coupled.
- PRs should include: purpose summary, config/testing instructions (`python …` commands), screenshots of rerun panels when UI changes occur, and linked issue/bug IDs.
- Ensure lint (`ruff`, `black`) and tests pass before requesting review; rebase onto latest `main` to keep SHM schema changes in sync.

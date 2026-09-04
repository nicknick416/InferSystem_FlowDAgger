# Repository Guidelines

## Project Structure & Module Organization

InferSystem is a Python control and inference scaffold organized by hardware boundary:

- `Core/`: shared dataclasses, config loading, logging, and registries.
- `Robot/`: robot abstractions and platform drivers such as Flexiv and ARX5.
- `Sensor/`: camera and tactile sensor interfaces, plus `Sensor/manager.py`.
- `Inference/`: ZMQ/msgpack client, server, and action dispatch utilities.
- `Config/`: example YAML system configurations.
- `Example/`: runnable hardware, calibration, replay, and inference scripts.
- `scripts/`: shell helpers such as CAN startup for ARX5.

Add reusable code to the relevant package, not to `Example/`. Keep examples configuration driven.

## Build, Test, and Development Commands

- `conda create -n infersystem python=3.10 -y`: create the recommended runtime.
- `pip install -r requirements.txt`: install shared dependencies.
- `pip install -e .`: install this repository in editable mode for imports.
- `python Example/test_inference_server.py --port 5555`: start a mock echo inference server.
- `python Example/test_inference_client.py Config/rizon4_example.yaml --dry-run --server 127.0.0.1:5555`: test the client loop without hardware.
- `python -m pytest`: run automated tests when tests are present under `tests/`.

Install robot SDKs separately as documented in `README.md`; hardware scripts require vendor drivers, device access, and correct YAML settings.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax, 4-space indentation, type hints, and focused functions. Existing code favors `dataclass(slots=True)`, `Path`, module-level loggers, and explicit lifecycle methods such as `connect()`, `open_all()`, and `close_all()`.

Use `PascalCase` for classes and `snake_case` for functions, variables, YAML keys, and modules. Keep comments brief and useful, especially around hardware safety or protocol behavior.

## Testing Guidelines

Put new unit tests in `tests/` so they match `pyproject.toml`. Name files `test_<feature>.py` and prefer mock sensors, mock robots, or dry-run paths over real hardware. Treat `Example/test_*.py` as integration checks for inference networking and hardware loops.

## Commit & Pull Request Guidelines

History uses Conventional Commit style with optional scopes, for example `feat(inference): ...`, `refactor(robot): ...`, and `docs(config): ...`. Keep commits focused.

Pull requests should describe the behavior change, list commands run, note affected hardware or configs, and link related issues. Include screenshots or logs for calibration, visualization, or inference behavior when useful.

## Security & Configuration Tips

Do not commit private network addresses, credentials, calibration captures, or site-specific robot settings unless they are sanitized examples. Prefer adding new sample configs under `Config/` with safe placeholder values.

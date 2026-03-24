# Repository Guidelines

## Project Structure & Module Organization
- `tensor_cpu/` contains the stable compiler/runtime path: tensor API, dispatcher, JIT, ABI, and runtime glue.
- `experimental/` holds promotion candidates and prototype implementations. Keep unstable changes isolated here until they are ready for `tensor_cpu/`.
- `examples/test/` contains the pytest suite for regression, integration, and training paths.
- `examples/benchmarks/` contains the benchmark harness and checked-in baselines such as `baseline.quick.json`.
- `docs/` stores architecture, support policy, stable API, and promotion guidance.

## Build, Test, and Development Commands
- `pip install -e .[dev] --no-build-isolation` installs the project in editable mode with the local dev toolchain.
- `pytest` runs the default test suite under `examples/test/`.
- `python examples/benchmarks/run_bench.py --quick --output benchmark-results.json` runs the smoke benchmark and writes a JSON report.
- `python examples/benchmarks/run_bench.py --quick --compare examples/benchmarks/baseline.quick.json --threshold 0.10` checks quick benchmark regressions against the baseline.
- `pre-commit run --all-files` applies the repo’s formatting, lint, type, and hygiene hooks.

## Coding Style & Naming Conventions
- Use Python 3.10+ with 4-space indentation and a 100-character line limit.
- Format code with `black` and `isort`; lint with `ruff`; type-check with `mypy`.
- Prefer `snake_case` for functions, modules, and test files; use `PascalCase` for classes.
- Keep experimental code in `experimental/` unless it is intentionally promoted into the stable path.

## Testing Guidelines
- Pytest discovers files named `test_*.py` under `examples/test/`.
- Add focused regression tests alongside the feature area they cover.
- Use markers consistently: `integration` for end-to-end paths and `benchmark` for lightweight performance checks.
- For benchmark-sensitive changes, compare against the checked-in baseline and note any meaningful runtime or compile-time deltas.

## Commit & Pull Request Guidelines
- The git history in this checkout only shows an initial commit, so there is no project-specific commit convention yet.
- Use short, imperative commit subjects, such as `fix matmul lowering` or `add benchmark baseline`.
- Pull requests should explain the change, list validation commands run, and call out benchmark or API impact.
- Include screenshots or logs only when they help verify a UI, benchmark, or runtime behavior change.

## Security & Configuration Tips
- Do not commit secrets, machine-specific paths, or large generated artifacts.
- Keep local benchmark outputs and scratch files out of version control unless they are intended as checked-in baselines.

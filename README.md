# Tensor CPU

Research/teaching-oriented CPU tensor compiler prototype with one stable execution path:

`Tensor -> TraceContext -> Graph -> optimize_graph -> CppCodegen -> JITEngine/runtime`

## Status

- Stable path: eager tensor API, tracing, graph passes, C++ code generation, ctypes-based runtime, graph-mode training helpers.
- Experimental path: control flow capture, layered codegen experiments, pybind runtime, symbolic helpers, lazy wrappers.
- Current stable constraints: single-output graphs only, no control-flow lowering on the default path, HPC 2D matmul requires exact traced shapes.

## Quick Start

```bash
pip install -e .[dev] --no-build-isolation
pytest
python examples/benchmarks/run_bench.py --quick --output benchmark-results.json
```

## Project Layout

- `tensor_cpu/`: stable package
- `experimental/`: experiments and promotion candidates
- `examples/test/`: regression and integration tests
- `examples/benchmarks/`: benchmark harness
- `docs/`: architecture, execution-path, stable API, support policy

## Docs

- `docs/architecture.md`
- `docs/execution-path.md`
- `docs/stable-api.md`
- `docs/support-policy.md`
- `docs/experimental.md`


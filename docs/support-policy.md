# Support Policy

This repository currently supports one stable execution path:

`Tensor -> TraceContext -> ir.Graph -> passes.optimize_graph -> backend.CppCodegen -> runtime.JITEngine`

## Supported

- Eager tensor execution on NumPy-backed tensors
- Trace-based graph capture
- Graph optimization pipeline on the stable IR
- Single-output graph compilation
- ctypes-based JIT runtime
- Graph-mode training-step compilation on the stable IR
- Dynamic shape reuse for non-HPC symbolic kernels when input ranks match

## Supported With Guardrails

- HPC 2D matmul kernels only for exact traced input shapes
- Benchmarking via `examples/benchmarks/run_bench.py`
- Experimental lazy training only through explicit `experimental.lazy` imports

## Not Supported On The Stable Path

- Control-flow lowering
- Multiple graph outputs
- Experimental layered codegen as the default backend
- pybind runtime as the default runtime
- ABI compatibility guarantees across releases

## Rules For New Work

- New features must enter behind a clear boundary: stable path or `experimental/`.
- Experimental modules cannot be re-exported from `tensor_cpu/__init__.py`.
- Any promoted feature must add tests, documentation, and a statement of runtime constraints.

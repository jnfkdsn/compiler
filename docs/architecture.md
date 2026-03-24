# Architecture

`tensor_cpu` currently has one stable execution path:

`Tensor -> frontend.tracer -> ir.Graph -> passes.optimize_graph -> backend.codegen -> runtime.JITEngine`

The package is organized around that path:

- `tensor_cpu/tensor.py`: user-facing eager tensor API and trace-aware operator entrypoints.
- `tensor_cpu/frontend/tracer.py`: trace context and graph construction helpers.
- `tensor_cpu/ir/`: graph IR, op enums, and shape inference.
- `tensor_cpu/passes/`: graph-level optimization pipeline.
- `tensor_cpu/backend/codegen.py`: Graph IR to C++ kernel emission.
- `tensor_cpu/runtime.py`: local compilation, ABI marshalling, and execution.
- `tensor_cpu/autodiff/`: VJP registry and training-step graph construction.
- `tensor_cpu/nn/`, `tensor_cpu/optim/`: higher-level model and optimizer APIs.

Stable-path constraints today:

- Single output only.
- No control-flow lowering on the default path.
- The default runtime is the ctypes-based JIT runtime.
- HPC 2D matmul kernels specialize literal input sizes and therefore require exact traced shapes at runtime.

Experimental modules live outside the default path and should not be treated as production-ready interfaces.

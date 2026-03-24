# Execution Path

The stable compiler/runtime path is intentionally narrow:

1. `Tensor` ops run eagerly with NumPy and, when tracing is enabled, also append nodes into `ir.Graph`.
2. `TraceContext` collects inputs, constants, and operator nodes and records a single graph output.
3. `passes.optimize_graph` optionally rewrites the graph, primarily for fusion and dead-code elimination.
4. `backend.CppCodegen` lowers the graph to one C++ kernel plus metadata:
   - symbolic output shape expressions
   - required input ranks
   - workspace layout
   - exact input shapes when a kernel uses literal-size specialization
5. `runtime.JITEngine` compiles the generated kernel with the local C++ toolchain and returns a `JITModule`.
6. `JITModule.run()` validates inputs, allocates output/workspace buffers, calls the ABI entrypoint, and returns a NumPy array.

Important runtime rules:

- Stable JIT compilation rejects graphs with more than one output.
- Rank mismatches are caught in Python before entering the native kernel.
- HPC-specialized 2D matmul kernels are guarded with exact traced input shapes to avoid silent wrong answers on dynamic-shape reuse.
- Non-HPC symbolic kernels still support dynamic shape reuse when ranks match the traced graph.

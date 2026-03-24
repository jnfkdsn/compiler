# Stable API

This document defines the stage-0 stable surface of the repository.

## Stable Top-Level Exports

The following top-level imports are part of the current stable path:

- `tensor_cpu.Tensor`
- `tensor_cpu.TraceContext`
- `tensor_cpu.CppCodegen`
- `tensor_cpu.GeneratedKernel`
- `tensor_cpu.JITEngine`
- `tensor_cpu.StaticGraph`
- `tensor_cpu.SymbolicTensor`
- `tensor_cpu.compile_training_step`
- `tensor_cpu.compile_sgd_update_kernel`
- `tensor_cpu.compile_adam_update_kernels`
- `tensor_cpu.enable_jit_matmul`
- `tensor_cpu.disable_jit_matmul`
- `tensor_cpu.optimize_graph`
- `tensor_cpu.AbiStatus`
- `tensor_cpu.decode_abi_status`

## Stable Subpackages

These subpackages are stable import roots:

- `tensor_cpu.ir`
- `tensor_cpu.frontend`
- `tensor_cpu.passes`
- `tensor_cpu.backend`
- `tensor_cpu.autodiff`
- `tensor_cpu.jit`
- `tensor_cpu.nn`
- `tensor_cpu.optim`

## Explicitly Not Stable

The following are intentionally outside the stable API:

- Any module under `experimental/`
- Control-flow lowering
- Multi-output graph compilation
- pybind runtime as the default runtime
- Lazy wrappers as a top-level `tensor_cpu` export

Promotion rule:

- A module is not stable unless it is documented here or re-exported from `tensor_cpu/__init__.py`.

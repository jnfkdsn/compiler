# Experimental Modules

The following modules are currently experimental and are not part of the stable execution path:

- `experimental/tracing_v2.py`
- `experimental/control_flow.py`
- `experimental/codegen_v2.py`
- `experimental/runtime_pybind.py`
- `experimental/symbolic.py`
- `experimental/lazy.py`

What “experimental” means in this repository:

- The module may expose useful ideas or partial implementations.
- It is not guaranteed to be wired into `tensor_cpu.runtime.JITEngine`.
- API shape and internal IR are allowed to change without compatibility guarantees.
- Test coverage is incomplete compared with the stable path.

Current guidance:

- Use the stable path for tracing, code generation, and runtime execution.
- Treat experimental modules as design branches or prototypes.
- Import experimental features explicitly from `experimental.*` rather than re-exporting them through `tensor_cpu`.
- `experimental.lazy` remains available for lazy-training prototypes, but `LazyTensor` is no longer part of the stable `tensor_cpu` top-level API.
- When promoting an experimental module, wire it into the default execution path, add tests, and document its constraints before advertising it as supported.
- Before promotion, walk through `docs/promotion-checklist.md` and satisfy every item there.

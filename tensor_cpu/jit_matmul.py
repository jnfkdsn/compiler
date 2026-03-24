"""JIT-compiled matmul kernel with shape-based caching.

Separated from tensor.py to avoid frontend-backend coupling.
Installs itself into the dispatcher's eager matmul slot when enabled;
tensor.py never needs to know about C++ compilation or dynamic libraries.
"""

from __future__ import annotations

import numpy as np

from . import dispatcher
from .ir.ops import OpType


_JIT_MATMUL_CACHE: dict[tuple[int, int, int, bool], object] = {}
_JIT_MATMUL_USE_HPC: bool = True
_JIT_MATMUL_BUILDING: bool = False


def enable_jit_matmul(use_hpc_template: bool = True) -> None:
    """Enable JIT-backed 2D matmul in eager mode with kernel cache."""
    global _JIT_MATMUL_USE_HPC
    _JIT_MATMUL_USE_HPC = bool(use_hpc_template)
    dispatcher.set_eager_binary(OpType.MATMUL, _jit_matmul_dispatch)


def disable_jit_matmul() -> None:
    """Disable JIT-backed matmul and clear cached kernels."""
    dispatcher.reset_eager_binary(OpType.MATMUL)
    _JIT_MATMUL_CACHE.clear()


def _jit_matmul_dispatch(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Dispatch matmul: use JIT for 2D non-tracing, fallback to NumPy otherwise."""
    if lhs.ndim != 2 or rhs.ndim != 2:
        return lhs @ rhs
    # During tracing the graph recorder must see plain numpy matmul
    if dispatcher.is_tracing():
        return lhs @ rhs
    return _jit_matmul(lhs, rhs)


def _jit_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"MatMul shape mismatch: {lhs.shape} @ {rhs.shape}")

    key = (int(lhs.shape[0]), int(lhs.shape[1]), int(rhs.shape[1]), bool(_JIT_MATMUL_USE_HPC))
    module = _JIT_MATMUL_CACHE.get(key)
    if module is None:
        global _JIT_MATMUL_BUILDING
        if _JIT_MATMUL_BUILDING:
            return lhs @ rhs
        _JIT_MATMUL_BUILDING = True
        try:
            # Deferred imports avoid circular dependency at package init time.
            from .frontend.tracer import TraceContext
            from .runtime import JITEngine
            from .tensor import Tensor

            m, k, n, _ = key
            a0 = np.zeros((m, k), dtype=np.float32)
            b0 = np.zeros((k, n), dtype=np.float32)
            with TraceContext() as tc:
                ta = Tensor.from_numpy(a0, name="jit_a")
                tb = Tensor.from_numpy(b0, name="jit_b")
                _ = (ta @ tb).mark_as_output()
                graph = tc.graph
            module = JITEngine(use_hpc_template=_JIT_MATMUL_USE_HPC).compile_graph(graph)
            _JIT_MATMUL_CACHE[key] = module
        finally:
            _JIT_MATMUL_BUILDING = False
    return module.run(lhs, rhs)

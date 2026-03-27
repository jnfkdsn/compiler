"""Unified operation dispatcher with thread-local trace state and gradient rules.

Centralizes three concerns previously scattered across tensor.py and tracer.py:

1. Thread-safe trace context (replaces module-level _TRACE_STATE)
   — Uses threading.local() so concurrent traces won't corrupt each other.

2. Eager op implementations (NumPy kernels, pluggable for JIT overrides)
   — JIT matmul installs itself here; tensor.py never sees compiler details.

3. Gradient rules (delegated to vjp.py — single source of truth)
   — Both eager backward (Tensor.backward) and graph-level AD
     (train_jit.build_backward_graph) consume the same VJPRule
     definitions from vjp.py.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from .autodiff.vjp import get_vjp_rule
from .ir.ops import OpType

# ---------------------------------------------------------------------------
# Thread-local trace state  (replaces global _TRACE_STATE in old tracer.py)
# ---------------------------------------------------------------------------


class _TraceState:
    __slots__ = ("enabled", "graph")

    def __init__(self) -> None:
        self.enabled: bool = False
        self.graph: Any = None  # Optional[Graph]


_tls = threading.local()


def _get_trace_state() -> _TraceState:
    state = getattr(_tls, "_trace_state", None)
    if state is None:
        state = _TraceState()
        _tls._trace_state = state
    return state


def is_tracing() -> bool:
    s = _get_trace_state()
    return s.enabled and s.graph is not None


def current_graph() -> Any:
    s = _get_trace_state()
    if s.graph is None:
        raise RuntimeError("No active trace graph. Use TraceContext().")
    return s.graph


def set_tracing(enabled: bool, graph: Any = None) -> None:
    s = _get_trace_state()
    s.enabled = enabled
    s.graph = graph


# ---------------------------------------------------------------------------
# Eager op registry  (pluggable — JIT matmul overrides the MATMUL slot)
# ---------------------------------------------------------------------------

_DEFAULT_BINARY: dict[OpType, Callable[..., np.ndarray]] = {
    OpType.ADD: lambda a, b: a + b,
    OpType.SUB: lambda a, b: a - b,
    OpType.MUL: lambda a, b: a * b,
    OpType.DIV: lambda a, b: a / b,
    OpType.MATMUL: lambda a, b: a @ b,
    OpType.RELU_GRAD: lambda x, grad: np.where(x > 0, grad, 0.0).astype(grad.dtype, copy=False),
}

_DEFAULT_UNARY: dict[OpType, Callable[..., np.ndarray]] = {
    OpType.RELU: lambda x: np.maximum(x, 0.0),
    OpType.EXP: np.exp,
    OpType.LOG: np.log,
    OpType.SIGMOID: lambda x: 1.0 / (1.0 + np.exp(-x)),
    OpType.TRANSPOSE: lambda x: x.T,
}

_EAGER_BINARY: dict[OpType, Callable[..., np.ndarray]] = dict(_DEFAULT_BINARY)
_EAGER_UNARY: dict[OpType, Callable[..., np.ndarray]] = dict(_DEFAULT_UNARY)


def eager_binary(op_type: OpType, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    fn = _EAGER_BINARY.get(op_type)
    if fn is None:
        raise RuntimeError(f"No eager implementation for binary op: {op_type}")
    return fn(lhs, rhs)


def eager_unary(op_type: OpType, data: np.ndarray) -> np.ndarray:
    fn = _EAGER_UNARY.get(op_type)
    if fn is None:
        raise RuntimeError(f"No eager implementation for unary op: {op_type}")
    return fn(data)


def set_eager_binary(op_type: OpType, fn: Callable[..., np.ndarray]) -> None:
    """Override the eager implementation for a binary op (e.g., JIT matmul)."""
    _EAGER_BINARY[op_type] = fn


def reset_eager_binary(op_type: OpType) -> None:
    """Reset a binary op to its default NumPy implementation."""
    default = _DEFAULT_BINARY.get(op_type)
    if default is not None:
        _EAGER_BINARY[op_type] = default


# ---------------------------------------------------------------------------
# Gradient rules — delegated to vjp.py (single source of truth)
# ---------------------------------------------------------------------------
def binary_grad_rule(op_type: OpType) -> Callable | None:
    rule = get_vjp_rule(op_type)
    return rule.eager if rule else None


def unary_grad_rule(op_type: OpType) -> Callable | None:
    rule = get_vjp_rule(op_type)
    return rule.eager if rule else None


def reduce_grad_rule(op_type: OpType) -> Callable | None:
    """Return the eager reduce gradient function, or ``None``."""
    rule = get_vjp_rule(op_type)
    return rule.eager if rule else None

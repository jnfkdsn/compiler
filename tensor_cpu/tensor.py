"""Tensor object with eager execution, autograd, and trace mode support.

Delegates computation dispatch, gradient rules, and trace state management
to the ``dispatcher`` module, keeping this file focused on the Tensor API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Set

import numpy as np

from . import dispatcher
from .frontend.tracer import (
    add_binary_node,
    add_const_node,
    add_input_node,
    add_reduce_node,
    add_transpose_node,
    add_unary_node,
    mark_output,
)
from .ir.ops import OpType
from .ir.shape_inference import normalize_reduce_axes


# ---------------------------------------------------------------------------
# Unified dispatch helpers (eager + trace + backward via dispatcher)
# ---------------------------------------------------------------------------

def _dispatch_binary(op_type: OpType, lhs: "Tensor", rhs: "Tensor") -> "Tensor":
    if lhs.data is None or rhs.data is None:
        raise RuntimeError(f"Eager tensor data is missing for {op_type.value}.")
    result_data = dispatcher.eager_binary(op_type, lhs.data, rhs.data)

    traced_node = None
    if dispatcher.is_tracing():
        if lhs.node is None or rhs.node is None:
            raise RuntimeError("Both operands must own nodes while tracing.")
        traced_node = add_binary_node(op_type, lhs.node, rhs.node)

    out = Tensor(
        data=result_data,
        node=traced_node,
        requires_grad=lhs.requires_grad or rhs.requires_grad,
        _prev={lhs, rhs},
    )

    grad_fn = dispatcher.binary_grad_rule(op_type)
    if grad_fn and (lhs.requires_grad or rhs.requires_grad):
        _lhs_data, _rhs_data = lhs.data, rhs.data

        def _backward() -> None:
            if out.grad is None:
                return
            lg, rg = grad_fn(out.grad, _lhs_data, _rhs_data, out.data)
            if lhs.requires_grad:
                lhs.grad = lg if lhs.grad is None else lhs.grad + lg
            if rhs.requires_grad:
                rhs.grad = rg if rhs.grad is None else rhs.grad + rg

        out._backward = _backward
    return out


def _dispatch_unary(op_type: OpType, src: "Tensor") -> "Tensor":
    if src.data is None:
        raise RuntimeError(f"Eager tensor data is missing for {op_type.value}.")
    result_data = dispatcher.eager_unary(op_type, src.data)

    traced_node = None
    if dispatcher.is_tracing():
        if src.node is None:
            raise RuntimeError("Operand must own a node while tracing.")
        if op_type == OpType.TRANSPOSE:
            traced_node = add_transpose_node(src.node)
        else:
            traced_node = add_unary_node(op_type, src.node)

    out = Tensor(
        data=result_data,
        node=traced_node,
        requires_grad=src.requires_grad,
        _prev={src},
    )

    grad_fn = dispatcher.unary_grad_rule(op_type)
    if grad_fn and src.requires_grad:
        _src_data = src.data

        def _backward() -> None:
            if out.grad is None or not src.requires_grad:
                return
            g = grad_fn(out.grad, _src_data, out.data)
            src.grad = g if src.grad is None else src.grad + g

        out._backward = _backward
    return out


def _dispatch_reduce(
    op_type: OpType,
    src: "Tensor",
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> "Tensor":
    if src.data is None:
        raise RuntimeError(f"Eager tensor data is missing for {op_type.value}.")

    if op_type == OpType.SUM:
        eager_result = np.asarray(src.data.sum(axis=axis, keepdims=keepdims), dtype=src.data.dtype)
    elif op_type == OpType.MEAN:
        eager_result = np.asarray(src.data.mean(axis=axis, keepdims=keepdims), dtype=src.data.dtype)
    elif op_type == OpType.MAX:
        eager_result = np.asarray(src.data.max(axis=axis, keepdims=keepdims), dtype=src.data.dtype)
    else:
        raise RuntimeError(f"Unsupported reduce op: {op_type}")

    axes = normalize_reduce_axes(axis=axis, ndim=src.data.ndim)

    traced_node = None
    if dispatcher.is_tracing():
        if src.node is None:
            raise RuntimeError("Operand must own a node while tracing.")
        traced_node = add_reduce_node(op_type, src.node, axis=axis, keepdims=keepdims)

    out = Tensor(
        data=eager_result,
        node=traced_node,
        requires_grad=src.requires_grad,
        _prev={src},
    )

    if src.requires_grad:
        _src_data = src.data
        _out_data = eager_result
        _axes = axes
        _keepdims = keepdims
        _reduce_grad_fn = dispatcher.reduce_grad_rule(op_type)

        if _reduce_grad_fn is not None:
            def _backward() -> None:
                if out.grad is None or not src.requires_grad:
                    return
                g = _reduce_grad_fn(out.grad, _src_data, _out_data, _axes, _keepdims)
                src.grad = g if src.grad is None else src.grad + g

            out._backward = _backward
    return out


@dataclass(slots=True)
class Tensor:
    """A lightweight tensor with eager execution and optional graph node."""

    data: Optional[np.ndarray]   #eager 
    node: Optional[object]      #tracing
    name: Optional[str] = None
    requires_grad: bool = False
    grad: Optional[np.ndarray] = None
    _prev: Set["Tensor"] = field(default_factory=set)
    _backward: Callable[[], None] = lambda: None

    def __hash__(self) -> int:
        return id(self)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        name: Optional[str] = None,
        requires_grad: bool = False,
    ) -> "Tensor":
        arr = np.asarray(array, dtype=np.float32)
        node = None
        if dispatcher.is_tracing():
            node = add_input_node(
                name=name or "input",
                shape=tuple(arr.shape),
                dtype=str(arr.dtype),
            )
        return Tensor(data=arr, node=node, name=name, requires_grad=requires_grad)

    @staticmethod
    def scalar(value: float, dtype: str = "float32", requires_grad: bool = False) -> "Tensor":
        arr = np.asarray(value, dtype=dtype)
        node = None
        if dispatcher.is_tracing():
            node = add_const_node(
                name="const",
                shape=tuple(arr.shape),
                dtype=str(arr.dtype),
                value=float(arr),
            )
        return Tensor(data=arr, node=node, name="const", requires_grad=requires_grad)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.data is None:
            if self.node is None:
                return ()
            return self.node.shape
        return tuple(self.data.shape)

    @property
    def dtype(self) -> str:
        if self.data is None:
            if self.node is None:
                return "float32"
            return self.node.dtype
        return str(self.data.dtype)

    # --- Binary ops (delegated to dispatcher) ---

    def __add__(self, other: "Tensor | float") -> "Tensor":
        return _dispatch_binary(OpType.ADD, self, self._ensure_tensor(other))

    def __radd__(self, other: "Tensor | float") -> "Tensor":
        return self.__add__(other)

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        return _dispatch_binary(OpType.MUL, self, self._ensure_tensor(other))

    def __rmul__(self, other: "Tensor | float") -> "Tensor":
        return self.__mul__(other)

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        return _dispatch_binary(OpType.SUB, self, self._ensure_tensor(other))

    def __rsub__(self, other: "Tensor | float") -> "Tensor":
        return self._ensure_tensor(other).__sub__(self)

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        return _dispatch_binary(OpType.DIV, self, self._ensure_tensor(other))

    def __rtruediv__(self, other: "Tensor | float") -> "Tensor":
        return self._ensure_tensor(other).__truediv__(self)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return _dispatch_binary(OpType.MATMUL, self, other)

    # --- Unary ops ---

    def transpose(self) -> "Tensor":
        return _dispatch_unary(OpType.TRANSPOSE, self)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def relu(self) -> "Tensor":
        return _dispatch_unary(OpType.RELU, self)

    def relu_grad(self, grad: "Tensor | float") -> "Tensor":
        return _dispatch_binary(OpType.RELU_GRAD, self, self._ensure_tensor(grad))

    def exp(self) -> "Tensor":
        return _dispatch_unary(OpType.EXP, self)

    def log(self) -> "Tensor":
        return _dispatch_unary(OpType.LOG, self)

    def sigmoid(self) -> "Tensor":
        return _dispatch_unary(OpType.SIGMOID, self)

    # --- Reduce ops ---

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        return _dispatch_reduce(OpType.SUM, self, axis, keepdims)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        return _dispatch_reduce(OpType.MEAN, self, axis, keepdims)

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        return _dispatch_reduce(OpType.MAX, self, axis, keepdims)

    # --- Compound ops ---

    def softmax(self, axis: int = -1) -> "Tensor":
        x_shifted = self - self.max(axis=axis, keepdims=True)
        exp_x = x_shifted.exp()
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    # --- Autograd ---

    def backward(self) -> None:
        if self.data is None:
            raise RuntimeError("Cannot backpropagate without eager data.")

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build(v: "Tensor") -> None:
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for tensor in reversed(topo):
            tensor._backward()

    def zero_grad(self) -> None:
        self.grad = None

    def mark_as_output(self) -> "Tensor":
        if dispatcher.is_tracing() and self.node is not None:
            mark_output(self.node)
        return self

    @staticmethod
    def _ensure_tensor(value: "Tensor | float") -> "Tensor":
        if isinstance(value, Tensor):
            return value
        return Tensor.scalar(value)

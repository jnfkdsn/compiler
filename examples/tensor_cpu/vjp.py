"""Unified VJP (Vector-Jacobian Product) rules for reverse-mode AD.

This module is the **single source of truth** for all gradient math.
Every operation defines both:

  - ``eager``: a NumPy-based function used by ``Tensor.backward()``
  - ``graph``: a Graph-node builder used by ``build_backward_graph()``

Both ``dispatcher.py`` (eager backward) and ``train_jit.py`` (graph
backward) consume rules from the unified registry here.  Adding a new
differentiable op requires **one** entry in ``_VJP_REGISTRY`` — no
duplicate definitions elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .graph import Graph, Node
from .ops import OpType
from .shape_inference import infer_binary, infer_reduce, infer_unary


# ====================================================================
# Unified rule dataclass
# ====================================================================

@dataclass(frozen=True, slots=True)
class VJPRule:
    """A gradient rule with both eager (NumPy) and graph execution paths."""
    eager: Callable
    graph: Callable


# ====================================================================
# Eager helpers (moved from dispatcher.py — single location now)
# ====================================================================

def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce broadcasted gradients back to operand shape."""
    if grad.shape == shape:
        return grad
    out = grad
    while len(out.shape) > len(shape):
        out = out.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and out.shape[axis] != 1:
            out = out.sum(axis=axis, keepdims=True)
    return out


def _expand_reduce_grad(
    grad: np.ndarray,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    keepdims: bool,
) -> np.ndarray:
    if not axes:
        return grad
    out = grad
    if not keepdims:
        for axis in axes:
            out = np.expand_dims(out, axis=axis)
    return np.broadcast_to(out, input_shape)


# ====================================================================
# Graph construction helpers
# ====================================================================

def add_const_scalar(graph: Graph, value: float, dtype: str = "float32") -> Node:
    return graph.add_node(
        op_type=OpType.CONST,
        name=f"const_{len(list(graph.nodes()))}",
        shape=(),
        dtype=dtype,
        attrs={"value": float(value)},
    )


def add_binary(graph: Graph, op: OpType, lhs: Node, rhs: Node, name: str) -> Node:
    out_shape, out_dtype = infer_binary(op, lhs.shape, rhs.shape, lhs.dtype, rhs.dtype)
    return graph.add_node(op_type=op, name=name, inputs=[lhs.id, rhs.id], shape=out_shape, dtype=out_dtype)


def add_unary(graph: Graph, op: OpType, src: Node, name: str) -> Node:
    out_shape, out_dtype = infer_unary(op, src.shape, src.dtype)
    return graph.add_node(op_type=op, name=name, inputs=[src.id], shape=out_shape, dtype=out_dtype)


def reduce_to_shape(graph: Graph, grad: Node, target_shape: tuple[int, ...], name_prefix: str) -> Node:
    """Reduce broadcasted *grad* back to *target_shape* using SUM reduce nodes."""
    if grad.shape == target_shape:
        return grad

    src_shape = grad.shape
    rank_src = len(src_shape)
    rank_tgt = len(target_shape)
    aligned_tgt = (1,) * (rank_src - rank_tgt) + target_shape

    axes: List[int] = []
    for axis, (sd, td) in enumerate(zip(src_shape, aligned_tgt)):
        if td == 1 and sd != 1:
            axes.append(axis)

    out = grad
    if axes:
        out_shape = tuple(dim for i, dim in enumerate(src_shape) if i not in set(axes))
        out = graph.add_node(
            op_type=OpType.SUM,
            name=f"{name_prefix}_sum",
            inputs=[grad.id],
            shape=out_shape,
            dtype=grad.dtype,
            attrs={"axis": tuple(axes), "keepdims": False},
        )

    if out.shape != target_shape:
        out = graph.add_node(
            op_type=OpType.BROADCAST_TO,
            name=f"{name_prefix}_reshape_like",
            inputs=[out.id],
            shape=target_shape,
            dtype=out.dtype,
            attrs={"target_shape": tuple(target_shape)},
        )
    return out


def broadcast_to(graph: Graph, grad: Node, target_shape: tuple[int, ...], name: str) -> Node:
    if grad.shape == target_shape:
        return grad
    return graph.add_node(
        op_type=OpType.BROADCAST_TO,
        name=name,
        inputs=[grad.id],
        shape=target_shape,
        dtype=grad.dtype,
        attrs={"target_shape": tuple(target_shape)},
    )


# ---- VJP rules  ----
# Signature: (graph, gout, input_nodes, out_node, attrs) -> list[Node | None]
# Each element in the returned list is the gradient w.r.t. the corresponding input.

def _vjp_add(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    lhs, rhs = inputs
    return [
        reduce_to_shape(graph, gout, lhs.shape, f"add_lhs_{out.id}"),
        reduce_to_shape(graph, gout, rhs.shape, f"add_rhs_{out.id}"),
    ]


def _vjp_sub(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    lhs, rhs = inputs
    gl = reduce_to_shape(graph, gout, lhs.shape, f"sub_lhs_{out.id}")
    minus1 = add_const_scalar(graph, -1.0, dtype=gout.dtype)
    gr = add_binary(graph, OpType.MUL, gout, minus1, f"sub_neg_{out.id}")
    gr = reduce_to_shape(graph, gr, rhs.shape, f"sub_rhs_{out.id}")
    return [gl, gr]


def _vjp_mul(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    lhs, rhs = inputs
    gl = add_binary(graph, OpType.MUL, gout, rhs, f"mul_gl_{out.id}")
    gr = add_binary(graph, OpType.MUL, gout, lhs, f"mul_gr_{out.id}")
    return [
        reduce_to_shape(graph, gl, lhs.shape, f"mul_lhs_{out.id}"),
        reduce_to_shape(graph, gr, rhs.shape, f"mul_rhs_{out.id}"),
    ]


def _vjp_div(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    lhs, rhs = inputs
    gl = add_binary(graph, OpType.DIV, gout, rhs, f"div_gl_{out.id}")
    rr = add_binary(graph, OpType.MUL, rhs, rhs, f"div_rr_{out.id}")
    num = add_binary(graph, OpType.MUL, gout, lhs, f"div_num_{out.id}")
    gr = add_binary(graph, OpType.DIV, num, rr, f"div_gr_{out.id}")
    minus1 = add_const_scalar(graph, -1.0, dtype=gout.dtype)
    gr = add_binary(graph, OpType.MUL, gr, minus1, f"div_gr_neg_{out.id}")
    return [
        reduce_to_shape(graph, gl, lhs.shape, f"div_lhs_{out.id}"),
        reduce_to_shape(graph, gr, rhs.shape, f"div_rhs_{out.id}"),
    ]


def _vjp_matmul(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    a, b = inputs
    bt = add_unary(graph, OpType.TRANSPOSE, b, f"matmul_bt_{out.id}")
    at = add_unary(graph, OpType.TRANSPOSE, a, f"matmul_at_{out.id}")
    ga = add_binary(graph, OpType.MATMUL, gout, bt, f"matmul_ga_{out.id}")
    gb = add_binary(graph, OpType.MATMUL, at, gout, f"matmul_gb_{out.id}")
    return [ga, gb]


def _vjp_transpose(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    return [add_unary(graph, OpType.TRANSPOSE, gout, f"tr_grad_{out.id}")]


def _vjp_exp(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    src = inputs[0]
    ex = add_unary(graph, OpType.EXP, src, f"exp_re_{out.id}")
    g = add_binary(graph, OpType.MUL, gout, ex, f"exp_grad_{out.id}")
    return [g]


def _vjp_log(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    src = inputs[0]
    g = add_binary(graph, OpType.DIV, gout, src, f"log_grad_{out.id}")
    return [g]


def _vjp_sigmoid(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    src = inputs[0]
    y = add_unary(graph, OpType.SIGMOID, src, f"sig_y_{out.id}")
    one = add_const_scalar(graph, 1.0, dtype=y.dtype)
    one_minus = add_binary(graph, OpType.SUB, one, y, f"sig_1my_{out.id}")
    term = add_binary(graph, OpType.MUL, y, one_minus, f"sig_term_{out.id}")
    g = add_binary(graph, OpType.MUL, gout, term, f"sig_grad_{out.id}")
    return [g]


def _vjp_relu(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    src = inputs[0]
    g = add_binary(graph, OpType.RELU_GRAD, src, gout, f"relu_grad_{out.id}")
    return [g]


def _vjp_reduce(graph: Graph, gout: Node, inputs: List[Node], out: Node, attrs: dict) -> List[Optional[Node]]:
    """VJP for SUM, MEAN, and MAX reduce ops."""
    src = inputs[0]
    op_type = out.op_type
    axes = tuple(int(v) for v in attrs.get("axis", ()))
    keepdims = bool(attrs.get("keepdims", False))

    g = gout
    if not keepdims and axes:
        exp_shape = list(g.shape)
        for axis in sorted(axes):
            exp_shape.insert(axis, 1)
        g = broadcast_to(graph, g, tuple(exp_shape), f"reduce_expand_{out.id}")
    g = broadcast_to(graph, g, src.shape, f"reduce_bcast_{out.id}")

    if op_type == OpType.MEAN:
        reduce_sz = 1.0
        for ax in axes:
            reduce_sz *= float(src.shape[ax])
        inv = add_const_scalar(graph, 1.0 / max(reduce_sz, 1.0), dtype=g.dtype)
        g = add_binary(graph, OpType.MUL, g, inv, f"mean_scale_{out.id}")
    elif op_type == OpType.MAX:
        # Sub-gradient: broadcast grad to positions matching the max value.
        # Tie handling matches eager mode (all maxima receive full upstream grad).
        max_ref = out
        if not keepdims and axes:
            exp_shape = list(max_ref.shape)
            for axis in sorted(axes):
                exp_shape.insert(axis, 1)
            max_ref = broadcast_to(graph, max_ref, tuple(exp_shape), f"max_expand_{out.id}")

        max_b = broadcast_to(graph, max_ref, src.shape, f"max_out_bcast_{out.id}")
        mask = add_binary(graph, OpType.EQ, src, max_b, f"max_mask_{out.id}")
        g = add_binary(graph, OpType.MUL, g, mask, f"max_grad_{out.id}")

    return [g]


# ====================================================================
# Eager backward rules (NumPy)
# ====================================================================

# -- Binary: (grad_out, lhs_data, rhs_data, out_data) -> (lhs_grad, rhs_grad)

def _eager_add(g: np.ndarray, l: np.ndarray, r: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _sum_to_shape(g, l.shape), _sum_to_shape(g, r.shape)

def _eager_sub(g: np.ndarray, l: np.ndarray, r: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _sum_to_shape(g, l.shape), _sum_to_shape(-g, r.shape)

def _eager_mul(g: np.ndarray, l: np.ndarray, r: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _sum_to_shape(g * r, l.shape), _sum_to_shape(g * l, r.shape)

def _eager_div(g: np.ndarray, l: np.ndarray, r: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _sum_to_shape(g / r, l.shape), _sum_to_shape(-g * l / (r * r), r.shape)

def _eager_matmul(g: np.ndarray, l: np.ndarray, r: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return g @ r.T, l.T @ g

# -- Unary: (grad_out, src_data, out_data) -> src_grad

def _eager_relu(g: np.ndarray, x: np.ndarray, o: np.ndarray) -> np.ndarray:
    return g * (x > 0)

def _eager_exp(g: np.ndarray, x: np.ndarray, o: np.ndarray) -> np.ndarray:
    return g * o

def _eager_log(g: np.ndarray, x: np.ndarray, o: np.ndarray) -> np.ndarray:
    return g / x

def _eager_sigmoid(g: np.ndarray, x: np.ndarray, o: np.ndarray) -> np.ndarray:
    return g * o * (1.0 - o)

def _eager_transpose(g: np.ndarray, x: np.ndarray, o: np.ndarray) -> np.ndarray:
    return g.T

# -- Reduce: (grad_out, src_data, out_data, axes, keepdims) -> src_grad

def _eager_sum(g: np.ndarray, x: np.ndarray, o: np.ndarray, axes: tuple, keepdims: bool) -> np.ndarray:
    expanded = _expand_reduce_grad(g, x.shape, axes, keepdims)
    return expanded * np.ones(x.shape, dtype=g.dtype)

def _eager_mean(g: np.ndarray, x: np.ndarray, o: np.ndarray, axes: tuple, keepdims: bool) -> np.ndarray:
    reduce_size = np.float32(1.0)
    for a in axes:
        reduce_size *= np.float32(x.shape[a])
    expanded = _expand_reduce_grad(g, x.shape, axes, keepdims)
    return (expanded / reduce_size) * np.ones(x.shape, dtype=g.dtype)

def _eager_max(g: np.ndarray, x: np.ndarray, o: np.ndarray, axes: tuple, keepdims: bool) -> np.ndarray:
    expanded_out = _expand_reduce_grad(o, x.shape, axes, keepdims)
    mask = (x == expanded_out).astype(g.dtype)
    expanded_grad = _expand_reduce_grad(g, x.shape, axes, keepdims)
    return expanded_grad * mask


# ====================================================================
# Unified registry  (single source of truth)
# ====================================================================

_VJP_REGISTRY: Dict[OpType, VJPRule] = {
    # Binary
    OpType.ADD:       VJPRule(eager=_eager_add,       graph=_vjp_add),
    OpType.SUB:       VJPRule(eager=_eager_sub,       graph=_vjp_sub),
    OpType.MUL:       VJPRule(eager=_eager_mul,       graph=_vjp_mul),
    OpType.DIV:       VJPRule(eager=_eager_div,       graph=_vjp_div),
    OpType.MATMUL:    VJPRule(eager=_eager_matmul,    graph=_vjp_matmul),
    # Unary
    OpType.RELU:      VJPRule(eager=_eager_relu,      graph=_vjp_relu),
    OpType.EXP:       VJPRule(eager=_eager_exp,       graph=_vjp_exp),
    OpType.LOG:       VJPRule(eager=_eager_log,       graph=_vjp_log),
    OpType.SIGMOID:   VJPRule(eager=_eager_sigmoid,   graph=_vjp_sigmoid),
    OpType.TRANSPOSE: VJPRule(eager=_eager_transpose, graph=_vjp_transpose),
    # Reduce
    OpType.SUM:       VJPRule(eager=_eager_sum,       graph=_vjp_reduce),
    OpType.MEAN:      VJPRule(eager=_eager_mean,      graph=_vjp_reduce),
    OpType.MAX:       VJPRule(eager=_eager_max,        graph=_vjp_reduce),
}


# ====================================================================
# Public API
# ====================================================================

def get_vjp_rule(op_type: OpType) -> Optional[VJPRule]:
    """Return the unified VJP rule for *op_type*, or ``None``."""
    return _VJP_REGISTRY.get(op_type)


def register_vjp(op_type: OpType, *, eager: Callable, graph: Callable) -> None:
    """Register (or override) a VJP rule for a custom op."""
    _VJP_REGISTRY[op_type] = VJPRule(eager=eager, graph=graph)


def apply_vjp(
    graph: Graph,
    op_type: OpType,
    gout: Node,
    inputs: List[Node],
    out_node: Node,
    attrs: dict,
) -> List[Optional[Node]]:
    """Look up and apply the graph-mode VJP rule for *op_type*.

    Returns a list of gradient nodes, one per input (``None`` if that
    input has no gradient contribution).
    """
    rule = _VJP_REGISTRY.get(op_type)
    if rule is None:
        raise NotImplementedError(f"No VJP rule registered for {op_type}")
    return rule.graph(graph, gout, inputs, out_node, attrs)

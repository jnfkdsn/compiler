"""Trace mode controller and global graph context.

Delegates thread-local state management to the ``dispatcher`` module,
making trace context inherently thread-safe.
"""

from __future__ import annotations

from .. import dispatcher
from ..ir.graph import Graph, Node
from ..ir.ops import OpType
from ..ir.shape_inference import infer_binary, infer_reduce, infer_unary


class TraceContext:
    """Context manager that enables graph tracing."""

    def __init__(self) -> None:
        self.graph = Graph()

    def __enter__(self) -> TraceContext:
        dispatcher.set_tracing(True, self.graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        dispatcher.set_tracing(False)


def is_tracing() -> bool:
    return dispatcher.is_tracing()


def current_graph() -> Graph:
    g = dispatcher.current_graph()
    if not isinstance(g, Graph):
        raise RuntimeError("Trace graph is not a Graph instance.")
    return g


def add_input_node(name: str, shape: tuple[int, ...], dtype: str) -> Node:
    graph = current_graph()
    return graph.add_node(op_type=OpType.INPUT, name=name, shape=shape, dtype=dtype)


def add_const_node(name: str, shape: tuple[int, ...], dtype: str, value: float) -> Node:
    graph = current_graph()
    return graph.add_node(
        op_type=OpType.CONST,
        name=name,
        shape=shape,
        dtype=dtype,
        attrs={"value": float(value)},
    )


def add_binary_node(op_type: OpType, lhs: Node, rhs: Node) -> Node:  # 二元节点
    graph = current_graph()
    out_shape, out_dtype = infer_binary(op_type, lhs.shape, rhs.shape, lhs.dtype, rhs.dtype)
    return graph.add_node(
        op_type=op_type,
        name=f"{op_type.value}_{lhs.id}_{rhs.id}",
        inputs=[lhs.id, rhs.id],
        shape=out_shape,
        dtype=out_dtype,
    )


def add_unary_node(op_type: OpType, src: Node) -> Node:  # 一元节点
    graph = current_graph()
    out_shape, out_dtype = infer_unary(op_type, src.shape, src.dtype)
    return graph.add_node(
        op_type=op_type,
        name=f"{op_type.value}_{src.id}",
        inputs=[src.id],
        shape=out_shape,
        dtype=out_dtype,
    )


def add_transpose_node(src: Node) -> Node:  # 转置节点
    graph = current_graph()
    out_shape, out_dtype = infer_unary(OpType.TRANSPOSE, src.shape, src.dtype)
    return graph.add_node(
        op_type=OpType.TRANSPOSE,
        name=f"transpose_{src.id}",
        inputs=[src.id],
        shape=out_shape,
        dtype=out_dtype,
    )


def add_reduce_node(
    op_type: OpType, src: Node, axis: int | tuple[int, ...] | None, keepdims: bool
) -> Node:  # 归约节点
    graph = current_graph()
    out_shape, out_dtype, axes = infer_reduce(
        op_type, src.shape, src.dtype, axis=axis, keepdims=keepdims
    )
    return graph.add_node(
        op_type=op_type,
        name=f"{op_type.value}_{src.id}",
        inputs=[src.id],
        shape=out_shape,
        dtype=out_dtype,
        attrs={"axis": axes, "keepdims": bool(keepdims)},
    )


def mark_output(node: Node) -> None:
    graph = current_graph()
    graph.mark_output(node.id)

"""Operator fusion passes for graph IR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ir.ops import OpType

if TYPE_CHECKING:
    from ..ir.graph import Graph


def fuse_matmul_bias_relu(graph: Graph) -> int:
    """Fuse MatMul + Add (+ ReLU) into one node when pattern matches.

    Returns the number of fused subgraphs.
    """
    fused_count = 0
    # Snapshot because graph is mutated during traversal.
    ordered_nodes = list(graph.topological_sort())

    for relu_or_add in ordered_nodes:
        if not graph.has_node(relu_or_add.id):
            continue

        with_relu = relu_or_add.op_type == OpType.RELU
        add_node = relu_or_add
        relu_node = None

        if with_relu:
            if len(relu_or_add.inputs) != 1:
                continue
            add_id = relu_or_add.inputs[0]
            if not graph.has_node(add_id):
                continue
            add_node = graph.get_node(add_id)
            relu_node = relu_or_add

        if add_node.op_type != OpType.ADD or len(add_node.inputs) != 2:
            continue

        add_use_count = _count_uses(graph, add_node.id)
        if relu_node is None and add_use_count > 1:
            continue
        if relu_node is not None and (_count_uses(graph, relu_node.id) > 0):
            continue

        lhs = graph.get_node(add_node.inputs[0])
        rhs = graph.get_node(add_node.inputs[1])

        if lhs.op_type == OpType.MATMUL:
            matmul_node = lhs
            bias_node = rhs
        elif rhs.op_type == OpType.MATMUL:
            matmul_node = rhs
            bias_node = lhs
        else:
            continue

        if _count_uses(graph, matmul_node.id) != 1:
            continue

        if len(bias_node.shape) != 1:
            continue
        if len(matmul_node.shape) != 2 or matmul_node.shape[1] != bias_node.shape[0]:
            continue

        fused_op = OpType.FUSED_MATMUL_BIAS_RELU if with_relu else OpType.FUSED_MATMUL_BIAS
        fused = graph.add_node(
            op_type=fused_op,
            name=f"{fused_op.value}_{matmul_node.id}_{bias_node.id}",
            inputs=[matmul_node.inputs[0], matmul_node.inputs[1], bias_node.id],
            shape=matmul_node.shape,
            dtype=matmul_node.dtype,
        )

        replace_target = relu_node.id if relu_node is not None else add_node.id
        graph.replace_all_uses(replace_target, fused.id)

        to_remove = [matmul_node.id, add_node.id]
        if relu_node is not None:
            to_remove.append(relu_node.id)
        graph.erase_nodes(to_remove)
        fused_count += 1

    return fused_count


def _count_uses(graph: Graph, node_id: int) -> int:
    count = 0
    for node in graph.nodes():
        for src in node.inputs:
            if src == node_id:
                count += 1
    return count

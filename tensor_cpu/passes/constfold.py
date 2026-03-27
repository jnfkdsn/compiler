"""Constant folding pass: evaluate ops with constant inputs."""

from __future__ import annotations

import math

from ..ir.graph import Graph
from ..ir.ops import OpType


def constant_fold(graph: Graph) -> int:
    """Fold nodes whose inputs are all constant into a single CONST node.

    Returns the number of folded nodes.
    """
    folded: list[int] = []

    def is_const(n_id: int) -> bool:
        if not graph.has_node(n_id):
            return False
        return graph.get_node(n_id).op_type == OpType.CONST

    for node in list(graph.topological_sort()):
        # skip constants and inputs
        if node.op_type in (OpType.CONST, OpType.INPUT, OpType.OUTPUT):
            continue

        # Only fold simple elementwise scalar ops where all inputs are const scalars
        if any(not is_const(i) for i in node.inputs):
            continue

        # gather input values
        vals = [graph.get_node(i).attrs.get("value") for i in node.inputs]
        if any(v is None for v in vals):
            continue

        try:
            if node.op_type == OpType.ADD:
                out = vals[0] + vals[1]
            elif node.op_type == OpType.SUB:
                out = vals[0] - vals[1]
            elif node.op_type == OpType.MUL:
                out = vals[0] * vals[1]
            elif node.op_type == OpType.DIV:
                out = vals[0] / vals[1]
            elif node.op_type == OpType.EXP:
                out = math.exp(vals[0])
            elif node.op_type == OpType.LOG:
                out = math.log(vals[0])
            elif node.op_type == OpType.RELU:
                out = vals[0] if vals[0] > 0 else 0.0
            else:
                # unsupported op for folding
                continue
        except Exception:
            continue

        # insert a new const node and replace uses
        new = graph.add_node(
            op_type=OpType.CONST,
            name=f"const_fold_{node.id}",
            shape=(),
            dtype=node.dtype,
            attrs={"value": float(out)},
        )
        graph.replace_all_uses(node.id, new.id)
        folded.append(node.id)

    # remove old nodes that were folded
    if folded:
        graph.erase_nodes(folded)
    return len(folded)

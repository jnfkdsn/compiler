"""Common subexpression elimination (CSE) pass."""

from __future__ import annotations

from typing import Dict, Tuple

from ..ir.graph import Graph
from ..ir.ops import OpType


def _node_key(node) -> Tuple:
    # Create a hashable key representing node semantics: op_type, inputs, attrs, shape, dtype
    return (
        node.op_type,
        tuple(node.inputs),
        tuple(sorted(node.attrs.items())),
        tuple(node.shape),
        node.dtype,
    )


def cse(graph: Graph) -> int:
    """Eliminate duplicate nodes computing the same value.

    Returns number of nodes removed.
    """
    seen: Dict[Tuple, int] = {}
    removed = []

    for node in list(graph.topological_sort()):
        # Skip inputs, consts and outputs
        if node.op_type in (OpType.INPUT, OpType.CONST, OpType.OUTPUT):
            continue

        key = _node_key(node)
        if key in seen:
            prev_id = seen[key]
            graph.replace_all_uses(node.id, prev_id)
            removed.append(node.id)
        else:
            seen[key] = node.id

    if removed:
        graph.erase_nodes(removed)
    return len(removed)

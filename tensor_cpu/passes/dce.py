"""Dead code elimination pass."""

from __future__ import annotations

from typing import Set


def dead_code_elimination(graph) -> int:
    """Remove nodes not reachable from graph outputs."""
    live: Set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in live or not graph.has_node(node_id):
            return
        live.add(node_id)
        node = graph.get_node(node_id)
        for src in node.inputs:
            visit(src)

    for out_id in graph.output_ids:
        visit(out_id)

    all_ids = [node.id for node in graph.nodes()]
    dead = [node_id for node_id in all_ids if node_id not in live]
    graph.erase_nodes(dead)
    return len(dead)

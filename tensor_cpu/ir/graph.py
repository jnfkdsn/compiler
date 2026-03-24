"""Graph IR data structures and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ops import OpType


Shape = Tuple[int, ...]


@dataclass(slots=True)
class Node:
    """A node in the traced computation graph."""

    id: int
    op_type: OpType
    name: str
    inputs: List[int]
    shape: Shape
    dtype: str
    attrs: Dict[str, object] = field(default_factory=dict)
    rank: int = 0
    numel: int = 1
    strides: Shape = ()


class Graph:
    """Directed acyclic graph used by the compiler frontend."""

    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}
        self._order: List[int] = [] #图优化
        self._next_id = 0
        self.output_ids: List[int] = []

    def add_node(
        self,
        *,
        op_type: OpType,
        name: str,
        inputs: Optional[Sequence[int]] = None,
        shape: Shape = (),
        dtype: str = "float32",
        attrs: Optional[Dict[str, object]] = None,
    ) -> Node:
        """Create and register a new graph node."""
        normalized_shape = tuple(shape)
        node = Node(
            id=self._next_id,
            op_type=op_type,
            name=name,
            inputs=list(inputs or []),
            shape=normalized_shape,
            dtype=dtype,
            attrs=dict(attrs or {}),
            rank=len(normalized_shape),
            numel=_numel(normalized_shape),
            strides=_contiguous_strides(normalized_shape),
        )
        self._nodes[node.id] = node
        self._order.append(node.id)
        self._next_id += 1
        return node

    def get_node(self, node_id: int) -> Node:
        return self._nodes[node_id]

    def has_node(self, node_id: int) -> bool:
        return node_id in self._nodes

    def nodes(self) -> Iterable[Node]:
        for node_id in self._order:
            if node_id in self._nodes:
                yield self._nodes[node_id]

    def mark_output(self, node_id: int) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node id: {node_id}")
        if node_id not in self.output_ids:
            self.output_ids.append(node_id)

    def replace_all_uses(self, old_id: int, new_id: int) -> None:
        for node in self.nodes():
            node.inputs = [new_id if node_id == old_id else node_id for node_id in node.inputs]
        self.output_ids = [new_id if node_id == old_id else node_id for node_id in self.output_ids]

    def erase_nodes(self, node_ids: Sequence[int]) -> None:
        to_remove = set(node_ids)
        for node_id in to_remove:
            self._nodes.pop(node_id, None)
        self._order = [node_id for node_id in self._order if node_id not in to_remove]

    def topological_sort(self) -> List[Node]: #node拓扑排序
        """Return nodes in topological order using Kahn's algorithm."""
        indegree: Dict[int, int] = {node.id: 0 for node in self.nodes()}
        outgoing: Dict[int, List[int]] = {node.id: [] for node in self.nodes()}

        for node in self.nodes():
            for src_id in node.inputs:
                indegree[node.id] += 1
                outgoing[src_id].append(node.id)

        ready: List[int] = [node_id for node_id, deg in indegree.items() if deg == 0]
        ordered: List[Node] = []

        while ready:
            current = ready.pop(0)
            ordered.append(self._nodes[current])

            for dst in outgoing[current]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)

        if len(ordered) != len(self._nodes):
            raise ValueError("Graph contains a cycle; expected a DAG.")
        return ordered

    def to_debug_string(self) -> str:
        """Human-readable DAG printout for quick verification."""
        lines = []
        for node in self.topological_sort():
            lines.append(
                " | ".join(
                    [
                        f"id={node.id}",
                        f"name={node.name}",
                        f"op={node.op_type.value}",
                        f"inputs={node.inputs}",
                        f"shape={node.shape}",
                        f"numel={node.numel}",
                        f"strides={node.strides}",
                        f"dtype={node.dtype}",
                    ]
                )
            )
        return "\n".join(lines)


def _numel(shape: Shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _contiguous_strides(shape: Shape) -> Shape:
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)

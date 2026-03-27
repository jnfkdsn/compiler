"""Control Flow Operators for Graph IR.

This module extends the Graph IR with control flow nodes:
- IfNode: Conditional branching with true/false subgraphs
- LoopNode: Iterative execution with body subgraph
- WhileNode: Conditional loop with condition and body subgraphs

These nodes enable capturing Python control flow in the computation graph,
which can then be lowered to C++ conditionals and loops during codegen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from tensor_cpu.ir.graph import Graph, Node
from tensor_cpu.ir.ops import OpType


class ControlFlowType(Enum):
    """Types of control flow operations."""

    IF = auto()
    ELSE = auto()
    ELIF = auto()
    WHILE = auto()
    FOR = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()


@dataclass(slots=True)
class SubGraph:
    """A subgraph representing a branch or loop body."""

    nodes: list[Node] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def get_node(self, node_id: int) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None


@dataclass(slots=True)
class IfNode(Node):
    """Conditional branch node with true and false subgraphs.

    Attributes:
        condition_node_id: ID of the node producing the condition value
        true_graph: Subgraph executed when condition is true
        false_graph: Subgraph executed when condition is false (else branch)
    """

    condition_node_id: int = -1
    true_graph: SubGraph | None = None
    false_graph: SubGraph | None = None

    def __post_init__(self):
        self.op_type = OpType.CONTROL_FLOW
        self.attrs["cf_type"] = ControlFlowType.IF


@dataclass(slots=True)
class WhileNode(Node):
    """While loop node with condition and body subgraphs.

    Attributes:
        condition_graph: Subgraph producing the loop condition
        body_graph: Subgraph executed on each iteration
    """

    condition_graph: SubGraph | None = None
    body_graph: SubGraph | None = None

    def __post_init__(self):
        self.op_type = OpType.CONTROL_FLOW
        self.attrs["cf_type"] = ControlFlowType.WHILE


@dataclass(slots=True)
class ForNode(Node):
    """For loop node with iterator and body subgraphs.

    Attributes:
        iterable_node_id: ID of the node producing the iterable
        loop_var_name: Name of the loop variable
        body_graph: Subgraph executed on each iteration
        loop_var_node_id: ID of the node representing the loop variable
    """

    iterable_node_id: int = -1
    loop_var_name: str = ""
    body_graph: SubGraph | None = None
    loop_var_node_id: int = -1

    def __post_init__(self):
        self.op_type = OpType.CONTROL_FLOW
        self.attrs["cf_type"] = ControlFlowType.FOR


@dataclass(slots=True)
class BreakNode(Node):
    """Break statement node."""

    def __post_init__(self):
        self.op_type = OpType.CONTROL_FLOW
        self.attrs["cf_type"] = ControlFlowType.BREAK


@dataclass(slots=True)
class ContinueNode(Node):
    """Continue statement node."""

    def __post_init__(self):
        self.op_type = OpType.CONTROL_FLOW
        self.attrs["cf_type"] = ControlFlowType.CONTINUE


class ControlFlowGraph(Graph):
    """Extended Graph with control flow support."""

    def __init__(self) -> None:
        super().__init__()
        self._control_flow_nodes: dict[int, IfNode | WhileNode | ForNode] = {}
        self._loop_stack: list[int] = []

    def add_if_node(
        self,
        condition_node_id: int,
        name: str = "if",
    ) -> IfNode:
        """Add an if node to the graph."""
        node = IfNode(
            id=self._next_id,
            op_type=OpType.CONTROL_FLOW,
            name=name,
            inputs=[condition_node_id],
            shape=(),
            dtype="bool",
            condition_node_id=condition_node_id,
            true_graph=SubGraph(),
            false_graph=SubGraph(),
        )
        self._nodes[node.id] = node
        self._order.append(node.id)
        self._next_id += 1
        self._control_flow_nodes[node.id] = node
        return node

    def add_while_node(
        self,
        name: str = "while",
    ) -> WhileNode:
        """Add a while node to the graph."""
        node = WhileNode(
            id=self._next_id,
            op_type=OpType.CONTROL_FLOW,
            name=name,
            inputs=[],
            shape=(),
            dtype="void",
            condition_graph=SubGraph(),
            body_graph=SubGraph(),
        )
        self._nodes[node.id] = node
        self._order.append(node.id)
        self._next_id += 1
        self._control_flow_nodes[node.id] = node
        self._loop_stack.append(node.id)
        return node

    def add_for_node(
        self,
        iterable_node_id: int,
        loop_var_name: str = "i",
        name: str = "for",
    ) -> ForNode:
        """Add a for node to the graph."""
        node = ForNode(
            id=self._next_id,
            op_type=OpType.CONTROL_FLOW,
            name=name,
            inputs=[iterable_node_id],
            shape=(),
            dtype="void",
            iterable_node_id=iterable_node_id,
            loop_var_name=loop_var_name,
            body_graph=SubGraph(),
        )
        self._nodes[node.id] = node
        self._order.append(node.id)
        self._next_id += 1
        self._control_flow_nodes[node.id] = node
        self._loop_stack.append(node.id)
        return node

    def add_break_node(self, loop_id: int, name: str = "break") -> BreakNode:
        """Add a break node to the graph."""
        node = BreakNode(
            id=self._next_id,
            op_type=OpType.CONTROL_FLOW,
            name=name,
            inputs=[],
            shape=(),
            dtype="void",
        )
        node.attrs["loop_id"] = loop_id
        self._nodes[node.id] = node
        self._order.append(node.id)
        self._next_id += 1
        return node

    def add_continue_node(self, loop_id: int, name: str = "continue") -> ContinueNode:
        """Add a continue node to the graph."""
        node = ContinueNode(
            id=self._next_id,
            op_type=OpType.CONTROL_FLOW,
            name=name,
            inputs=[],
            shape=(),
            dtype="void",
        )
        node.attrs["loop_id"] = loop_id
        self._nodes[node.id] = node
        self._order.append(node.id)
        self._next_id += 1
        return node

    def get_control_flow_node(self, node_id: int) -> IfNode | WhileNode | ForNode | None:
        """Get a control flow node by ID."""
        return self._control_flow_nodes.get(node_id)

    def current_loop(self) -> int | None:
        """Get the current loop ID (top of loop stack)."""
        return self._loop_stack[-1] if self._loop_stack else None

    def pop_loop(self) -> int | None:
        """Pop the current loop from the stack."""
        if self._loop_stack:
            return self._loop_stack.pop()
        return None


class ControlFlowLowering:
    """Lower control flow nodes to C++ AST."""

    def __init__(self, graph: ControlFlowGraph) -> None:
        self.graph = graph

    def lower_if(self, node: IfNode, names: dict[int, str]) -> list[str]:
        """Lower an if node to C++ code."""
        lines = []
        cond_name = names.get(node.condition_node_id, "cond")

        lines.append(f"if ({cond_name}) {{")

        if node.true_graph:
            for sub_node in node.true_graph.nodes:
                lines.extend(self._lower_subgraph_node(sub_node, names))

        if node.false_graph and node.false_graph.nodes:
            lines.append("} else {")
            for sub_node in node.false_graph.nodes:
                lines.extend(self._lower_subgraph_node(sub_node, names))

        lines.append("}")
        return lines

    def lower_while(self, node: WhileNode, names: dict[int, str]) -> list[str]:
        """Lower a while node to C++ code."""
        lines = []

        lines.append("while (true) {")

        if node.condition_graph:
            for sub_node in node.condition_graph.nodes:
                lines.extend(self._lower_subgraph_node(sub_node, names))

            cond_name = names.get(node.condition_graph.output_ids[-1], "cond")
            lines.append(f"if (!{cond_name}) break;")

        if node.body_graph:
            for sub_node in node.body_graph.nodes:
                lines.extend(self._lower_subgraph_node(sub_node, names))

        lines.append("}")
        return lines

    def lower_for(self, node: ForNode, names: dict[int, str]) -> list[str]:
        """Lower a for node to C++ code."""
        lines = []
        iter_name = names.get(node.iterable_node_id, "iter")
        var_name = node.loop_var_name

        lines.append(f"for (auto& {var_name} : {iter_name}) {{")

        if node.body_graph:
            for sub_node in node.body_graph.nodes:
                lines.extend(self._lower_subgraph_node(sub_node, names))

        lines.append("}")
        return lines

    def _lower_subgraph_node(self, node: Node, names: dict[int, str]) -> list[str]:
        """Lower a node within a subgraph."""
        return [f"// Node {node.id}: {node.op_type.value}"]

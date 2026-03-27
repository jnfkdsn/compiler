"""Python AST and Bytecode parsing for control flow capture.

This module provides mechanisms to capture Python control flow (if/while/for)
by parsing the source code AST or bytecode, instead of relying solely on
operator overloading-based tracing.

Two approaches are supported:
- AST-based: Parse source code using Python's ast module
- Bytecode-based: Analyze function bytecode using dis module (recommended)
"""

from __future__ import annotations

import ast
import dis
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ControlFlowOp(Enum):
    """Control flow operation types."""

    IF = "if"
    ELSE = "else"
    WHILE = "while"
    FOR = "for"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"


@dataclass(slots=True)
class ControlFlowNode:
    """A node representing control flow in the graph."""

    op_type: ControlFlowOp
    condition: TracedValue | None = None
    true_branch: list[GraphNode] | None = None
    false_branch: list[GraphNode] | None = None
    body: list[GraphNode] | None = None
    loop_var: str | None = None
    iterable: TracedValue | None = None


@dataclass(slots=True)
class TracedValue:
    """A value tracked during AST/bytecode tracing."""

    name: str
    node_id: int
    shape: tuple[int, ...] = ()
    dtype: str = "float32"
    is_symbolic: bool = False


@dataclass(slots=True)
class GraphNode:
    """A node in the traced computation graph."""

    id: int
    op: str
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    control_flow: ControlFlowNode | None = None


@dataclass
class TracingState:
    """State maintained during AST/bytecode tracing."""

    scope: dict[str, TracedValue] = field(default_factory=dict)
    graph_nodes: list[GraphNode] = field(default_factory=list)
    next_id: int = 0
    current_loop: str | None = None
    loop_vars: set[str] = field(default_factory=set)

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def get_or_create(
        self, name: str, shape: tuple[int, ...] = (), dtype: str = "float32"
    ) -> TracedValue:
        if name not in self.scope:
            node_id = self.new_id()
            value = TracedValue(name, node_id, shape, dtype)
            self.scope[name] = value
            self.graph_nodes.append(
                GraphNode(id=node_id, op="input", attrs={"shape": shape, "dtype": dtype})
            )
        return self.scope[name]

    def create_computed(
        self,
        op: str,
        inputs: list[TracedValue],
        shape: tuple[int, ...] = (),
        dtype: str = "float32",
        attrs: dict[str, Any] | None = None,
    ) -> TracedValue:
        node_id = self.new_id()
        value = TracedValue(f"_{op}_{node_id}", node_id, shape, dtype)
        self.scope[value.name] = value

        node = GraphNode(id=node_id, op=op, inputs=[v.node_id for v in inputs], attrs=attrs or {})
        node.attrs.update({"shape": shape, "dtype": dtype})
        self.graph_nodes.append(node)

        return value


class ASTTracer(ast.NodeVisitor):
    """Trace Python AST to capture control flow and tensor operations."""

    def __init__(self, state: TracingState) -> None:
        self.state = state
        self.current_value: TracedValue | None = None

    def trace_function(self, func: Callable, *args, **kwargs) -> list[GraphNode]:
        """Trace a function and return the captured graph nodes."""
        source = inspect.getsource(func)
        tree = ast.parse(source)

        for arg_name in inspect.signature(func).parameters:
            self.state.get_or_create(arg_name)

        self.visit(tree)
        return self.state.graph_nodes

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value:
            self.visit(node.value)
            self.state.graph_nodes.append(
                GraphNode(
                    id=self.state.new_id(),
                    op="output",
                    inputs=[self.current_value.node_id] if self.current_value else [],
                )
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self.current_value

        for target in node.targets:
            if isinstance(target, ast.Name):
                self.state.scope[target.id] = value
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.state.scope[elt.id] = value

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        target_val = self.current_value
        self.visit(node.value)
        value_val = self.current_value

        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.FloorDiv: "div",
            ast.Mod: "mod",
            ast.Pow: "pow",
        }
        op = op_map.get(type(node.op), "unknown")

        self.current_value = self.state.create_computed(op, [target_val, value_val])

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        cond = self.current_value

        true_nodes_before = len(self.state.graph_nodes)
        for stmt in node.body:
            self.visit(stmt)
        true_nodes = self.state.graph_nodes[true_nodes_before:]

        false_nodes = []
        if node.orelse:
            false_nodes_before = len(self.state.graph_nodes)
            for stmt in node.orelse:
                self.visit(stmt)
            false_nodes = self.state.graph_nodes[false_nodes_before:]

        cf_node = ControlFlowNode(
            op_type=ControlFlowOp.IF,
            condition=cond,
            true_branch=true_nodes,
            false_branch=false_nodes,
        )

        self.state.graph_nodes.append(
            GraphNode(id=self.state.new_id(), op="if", control_flow=cf_node)
        )

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        cond = self.current_value

        body_nodes_before = len(self.state.graph_nodes)
        for stmt in node.body:
            self.visit(stmt)
        body_nodes = self.state.graph_nodes[body_nodes_before:]

        cf_node = ControlFlowNode(
            op_type=ControlFlowOp.WHILE,
            condition=cond,
            body=body_nodes,
        )

        self.state.graph_nodes.append(
            GraphNode(id=self.state.new_id(), op="while", control_flow=cf_node)
        )

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        iterable = self.current_value

        loop_var = None
        if isinstance(node.target, ast.Name):
            loop_var = node.target.id
            self.state.loop_vars.add(loop_var)
            self.state.get_or_create(loop_var)

        body_nodes_before = len(self.state.graph_nodes)
        old_loop = self.state.current_loop
        self.state.current_loop = loop_var

        for stmt in node.body:
            self.visit(stmt)

        self.state.current_loop = old_loop
        body_nodes = self.state.graph_nodes[body_nodes_before:]

        cf_node = ControlFlowNode(
            op_type=ControlFlowOp.FOR,
            loop_var=loop_var,
            iterable=iterable,
            body=body_nodes,
        )

        self.state.graph_nodes.append(
            GraphNode(id=self.state.new_id(), op="for", control_flow=cf_node)
        )

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.visit(node.left)
        left = self.current_value
        self.visit(node.right)
        right = self.current_value

        op_map = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.FloorDiv: "div",
            ast.Mod: "mod",
            ast.Pow: "pow",
            ast.LShift: "lshift",
            ast.RShift: "rshift",
            ast.BitOr: "bitwise_or",
            ast.BitXor: "bitwise_xor",
            ast.BitAnd: "bitwise_and",
        }
        op = op_map.get(type(node.op), "unknown")

        self.current_value = self.state.create_computed(op, [left, right])

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.visit(node.operand)
        operand = self.current_value

        op_map = {
            ast.UAdd: "pos",
            ast.USub: "neg",
            ast.Not: "not",
            ast.Invert: "invert",
        }
        op = op_map.get(type(node.op), "unknown")

        self.current_value = self.state.create_computed(op, [operand])

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        left = self.current_value

        comparators = []
        for comp in node.comparators:
            self.visit(comp)
            comparators.append(self.current_value)

        op_map = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }

        if len(node.ops) == 1:
            op = op_map.get(type(node.ops[0]), "unknown")
            self.current_value = self.state.create_computed(op, [left, comparators[0]])
        else:
            self.current_value = self.state.create_computed(
                "compare",
                [left] + comparators,
                attrs={"ops": [op_map.get(type(o), "unknown") for o in node.ops]},
            )

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            func_name = "unknown"

        args = []
        for arg in node.args:
            self.visit(arg)
            args.append(self.current_value)

        for kw in node.keywords:
            self.visit(kw.value)
            args.append(self.current_value)

        self.current_value = self.state.create_computed(
            func_name, args, attrs={"is_call": True, "func_name": func_name}
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)
        base = self.current_value

        if isinstance(node.slice, ast.Index):
            self.visit(node.slice.value)
            index = self.current_value
        elif isinstance(node.slice, ast.Slice):
            indices = []
            if node.slice.lower:
                self.visit(node.slice.lower)
                indices.append(self.current_value)
            if node.slice.upper:
                self.visit(node.slice.upper)
                indices.append(self.current_value)
            if node.slice.step:
                self.visit(node.slice.step)
                indices.append(self.current_value)
            index = indices[0] if len(indices) == 1 else None
        else:
            self.visit(node.slice)
            index = self.current_value

        self.current_value = self.state.create_computed("index", [base, index] if index else [base])

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.visit(node.value)
        base = self.current_value

        self.current_value = self.state.create_computed(
            "getattr", [base], attrs={"attr": node.attr}
        )

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.state.scope:
            self.current_value = self.state.scope[node.id]
        else:
            self.current_value = self.state.get_or_create(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if isinstance(value, (int, float)):
            self.current_value = self.state.create_computed("const", [], attrs={"value": value})
        else:
            self.current_value = self.state.create_computed("const", [], attrs={"value": value})

    def visit_Num(self, node: ast.Num) -> None:
        self.current_value = self.state.create_computed("const", [], attrs={"value": node.n})

    def visit_Str(self, node: ast.Str) -> None:
        self.current_value = self.state.create_computed("const", [], attrs={"value": node.s})


class BytecodeTracer:
    """Trace Python bytecode to capture control flow and tensor operations.

    This is the recommended approach as it works with compiled code
    and handles edge cases that AST parsing might miss.
    """

    def __init__(self) -> None:
        self.state = TracingState()
        self._stack: list[TracedValue] = []
        self._block_stack: list[tuple[str, int]] = []

    def trace_function(self, func: Callable, *args, **kwargs) -> list[GraphNode]:
        """Trace a function using bytecode analysis."""
        for i, (name, _param) in enumerate(inspect.signature(func).parameters.items()):
            if i < len(args):
                shape = getattr(args[i], "shape", ())
                dtype = getattr(args[i], "dtype", "float32")
            else:
                shape = ()
                dtype = "float32"
            self.state.get_or_create(name, shape, dtype)

        bytecode = dis.Bytecode(func)
        self._analyze_bytecode(bytecode)

        return self.state.graph_nodes

    def _analyze_bytecode(self, bytecode: dis.Bytecode) -> None:
        """Analyze bytecode instructions."""
        instructions = list(bytecode)
        i = 0

        while i < len(instructions):
            instr = instructions[i]
            self._process_instruction(instr, instructions, i)
            i += 1

    def _process_instruction(
        self, instr: dis.Instruction, instructions: list[dis.Instruction], idx: int
    ) -> None:
        """Process a single bytecode instruction."""
        opname = instr.opname

        if opname == "LOAD_FAST" or opname == "LOAD_GLOBAL":
            name = instr.argval
            if name in self.state.scope:
                self._stack.append(self.state.scope[name])
            else:
                self._stack.append(self.state.get_or_create(name))

        elif opname == "LOAD_CONST":
            value = instr.argval
            traced = self.state.create_computed("const", [], attrs={"value": value})
            self._stack.append(traced)

        elif opname == "STORE_FAST" or opname == "STORE_NAME":
            name = instr.argval
            if self._stack:
                value = self._stack.pop()
                self.state.scope[name] = value

        elif opname == "BINARY_ADD":
            self._binary_op("add")
        elif opname == "BINARY_SUBTRACT":
            self._binary_op("sub")
        elif opname == "BINARY_MULTIPLY":
            self._binary_op("mul")
        elif opname == "BINARY_TRUE_DIVIDE":
            self._binary_op("div")
        elif opname == "BINARY_FLOOR_DIVIDE":
            self._binary_op("floor_div")
        elif opname == "BINARY_MODULO":
            self._binary_op("mod")
        elif opname == "BINARY_POWER":
            self._binary_op("pow")

        elif opname == "COMPARE_OP":
            cmp_op = dis.cmp_op[instr.arg]
            op_map = {
                "<": "lt",
                "<=": "le",
                "==": "eq",
                "!=": "ne",
                ">": "gt",
                ">=": "ge",
            }
            self._binary_op(op_map.get(cmp_op, "compare"))

        elif opname == "UNARY_NEGATIVE":
            self._unary_op("neg")
        elif opname == "UNARY_POSITIVE":
            self._unary_op("pos")
        elif opname == "UNARY_NOT":
            self._unary_op("not")

        elif opname == "CALL_FUNCTION" or opname == "CALL_METHOD":
            nargs = instr.arg
            args = []
            for _ in range(nargs):
                if self._stack:
                    args.insert(0, self._stack.pop())

            func = self._stack.pop() if self._stack else None
            func_name = getattr(func, "name", "unknown") if func else "unknown"

            result = self.state.create_computed(func_name, args, attrs={"is_call": True})
            self._stack.append(result)

        elif opname == "RETURN_VALUE":
            if self._stack:
                value = self._stack.pop()
                self.state.graph_nodes.append(
                    GraphNode(id=self.state.new_id(), op="output", inputs=[value.node_id])
                )

        elif opname == "POP_JUMP_IF_FALSE" or opname == "POP_JUMP_IF_TRUE":
            cond = self._stack.pop() if self._stack else None
            target = instr.arg

            cf_node = ControlFlowNode(
                op_type=ControlFlowOp.IF,
                condition=cond,
            )

            self.state.graph_nodes.append(
                GraphNode(
                    id=self.state.new_id(),
                    op="if_branch",
                    control_flow=cf_node,
                    attrs={"target": target, "jump_if_true": opname == "POP_JUMP_IF_TRUE"},
                )
            )

        elif opname == "SETUP_LOOP":
            self._block_stack.append(("loop", instr.arg))

        elif opname == "GET_ITER":
            if self._stack:
                iterable = self._stack.pop()
                result = self.state.create_computed("iter", [iterable])
                self._stack.append(result)

        elif opname == "FOR_ITER":
            if self._stack:
                self._stack[-1]
                loop_var = f"_loop_var_{self.state.new_id()}"
                result = self.state.get_or_create(loop_var)
                self._stack.append(result)

        elif opname == "JUMP_ABSOLUTE":
            target = instr.arg
            self.state.graph_nodes.append(
                GraphNode(id=self.state.new_id(), op="jump", attrs={"target": target})
            )

    def _binary_op(self, op: str) -> None:
        """Process a binary operation."""
        if len(self._stack) >= 2:
            right = self._stack.pop()
            left = self._stack.pop()
            result = self.state.create_computed(op, [left, right])
            self._stack.append(result)

    def _unary_op(self, op: str) -> None:
        """Process a unary operation."""
        if self._stack:
            operand = self._stack.pop()
            result = self.state.create_computed(op, [operand])
            self._stack.append(result)


def trace_function(func: Callable, method: str = "bytecode") -> list[GraphNode]:
    """Trace a function and return the captured graph nodes.

    Args:
        func: The function to trace
        method: Tracing method, either "ast" or "bytecode" (default: "bytecode")

    Returns:
        List of captured graph nodes including control flow
    """
    if method == "ast":
        state = TracingState()
        tracer = ASTTracer(state)
        return tracer.trace_function(func)
    else:
        tracer = BytecodeTracer()
        return tracer.trace_function(func)


def trace_method(obj: Any, method_name: str, method: str = "bytecode") -> list[GraphNode]:
    """Trace a method and return the captured graph nodes."""
    func = getattr(obj, method_name)
    return trace_function(func, method)

"""Strict static-graph frontend: define graph first, compile, then run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .graph import Graph, Node
from .ops import OpType
from .runtime import JITEngine, JITModule
from .shape_inference import infer_binary, infer_reduce, infer_unary


@dataclass
class SymbolicTensor:
    """Symbolic tensor bound to a static graph node.

    This object does not hold eager data. All numerical execution happens only
    after `StaticGraph.compile(...).run(...)`.
    """

    graph: "StaticGraph"
    node: Node

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.node.shape)

    @property
    def dtype(self) -> str:
        return str(self.node.dtype)

    def _ensure(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        if isinstance(other, SymbolicTensor):
            if other.graph is not self.graph:
                raise ValueError("Cannot mix SymbolicTensor from different StaticGraph instances.")
            return other
        return self.graph.const(float(other), dtype=self.dtype)

    def _binary(self, op: OpType, other: "SymbolicTensor | float", name: str) -> "SymbolicTensor":
        rhs = self._ensure(other)
        out_shape, out_dtype = infer_binary(op, self.shape, rhs.shape, self.dtype, rhs.dtype)
        n = self.graph._graph.add_node(
            op_type=op,
            name=f"{name}_{self.node.id}_{rhs.node.id}",
            inputs=[self.node.id, rhs.node.id],
            shape=out_shape,
            dtype=out_dtype,
        )
        return SymbolicTensor(graph=self.graph, node=n)

    def _unary(self, op: OpType, name: str) -> "SymbolicTensor":
        out_shape, out_dtype = infer_unary(op, self.shape, self.dtype)
        n = self.graph._graph.add_node(
            op_type=op,
            name=f"{name}_{self.node.id}",
            inputs=[self.node.id],
            shape=out_shape,
            dtype=out_dtype,
        )
        return SymbolicTensor(graph=self.graph, node=n)

    def __add__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        return self._binary(OpType.ADD, other, "add")

    def __radd__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        return self.__add__(other)

    def __sub__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        return self._binary(OpType.SUB, other, "sub")

    def __rsub__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        lhs = self._ensure(other)
        return lhs.__sub__(self)

    def __mul__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        return self._binary(OpType.MUL, other, "mul")

    def __rmul__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        return self.__mul__(other)

    def __truediv__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        return self._binary(OpType.DIV, other, "div")

    def __rtruediv__(self, other: "SymbolicTensor | float") -> "SymbolicTensor":
        lhs = self._ensure(other)
        return lhs.__truediv__(self)

    def __matmul__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        return self._binary(OpType.MATMUL, other, "matmul")

    def relu(self) -> "SymbolicTensor":
        return self._unary(OpType.RELU, "relu")

    def exp(self) -> "SymbolicTensor":
        return self._unary(OpType.EXP, "exp")

    def log(self) -> "SymbolicTensor":
        return self._unary(OpType.LOG, "log")

    def sigmoid(self) -> "SymbolicTensor":
        return self._unary(OpType.SIGMOID, "sigmoid")

    def transpose(self) -> "SymbolicTensor":
        return self._unary(OpType.TRANSPOSE, "transpose")

    @property
    def T(self) -> "SymbolicTensor":
        return self.transpose()

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "SymbolicTensor":
        out_shape, out_dtype, axes = infer_reduce(OpType.SUM, self.shape, self.dtype, axis=axis, keepdims=keepdims)
        n = self.graph._graph.add_node(
            op_type=OpType.SUM,
            name=f"sum_{self.node.id}",
            inputs=[self.node.id],
            shape=out_shape,
            dtype=out_dtype,
            attrs={"axis": axes, "keepdims": bool(keepdims)},
        )
        return SymbolicTensor(graph=self.graph, node=n)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "SymbolicTensor":
        out_shape, out_dtype, axes = infer_reduce(OpType.MEAN, self.shape, self.dtype, axis=axis, keepdims=keepdims)
        n = self.graph._graph.add_node(
            op_type=OpType.MEAN,
            name=f"mean_{self.node.id}",
            inputs=[self.node.id],
            shape=out_shape,
            dtype=out_dtype,
            attrs={"axis": axes, "keepdims": bool(keepdims)},
        )
        return SymbolicTensor(graph=self.graph, node=n)

    def mark_as_output(self) -> "SymbolicTensor":
        self.graph._graph.mark_output(self.node.id)
        return self


class StaticCompiledGraph:
    """Compiled static graph executable."""

    def __init__(self, module: JITModule, input_names: List[str]) -> None:
        self._module = module
        self._input_names = list(input_names)

    @property
    def input_names(self) -> List[str]:
        return list(self._input_names)

    def run(self, *inputs: np.ndarray, **named_inputs: np.ndarray) -> np.ndarray:
        if named_inputs:
            if inputs:
                raise ValueError("Use either positional inputs or named_inputs, not both.")
            ordered = []
            for n in self._input_names:
                if n not in named_inputs:
                    raise ValueError(f"Missing input '{n}'. Expected inputs: {self._input_names}")
                ordered.append(named_inputs[n])
            return self._module.run(*ordered)
        return self._module.run(*inputs)


class StaticGraph:
    """Strict static graph session.

    API usage:
    1) Create symbolic inputs via `input(...)`.
    2) Build graph with SymbolicTensor ops.
    3) Mark output.
    4) Compile and run.
    """

    def __init__(self) -> None:
        self._graph = Graph()
        self._input_nodes: List[Node] = []
        self._input_name_to_id: Dict[str, int] = {}
        self._compiled: Optional[StaticCompiledGraph] = None

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def input_names(self) -> List[str]:
        return [n.name for n in self._input_nodes]

    def input_id(self, name: str) -> int:
        if name not in self._input_name_to_id:
            raise KeyError(f"Unknown input name: {name}")
        return self._input_name_to_id[name]

    def input(self, name: str, shape: tuple[int, ...], dtype: str = "float32") -> SymbolicTensor:
        if dtype != "float32":
            raise ValueError("StaticGraph currently supports only float32.")
        if name in self._input_name_to_id:
            raise ValueError(f"Duplicate input name: {name}")
        n = self._graph.add_node(op_type=OpType.INPUT, name=name, shape=tuple(shape), dtype=dtype)
        self._input_nodes.append(n)
        self._input_name_to_id[name] = n.id
        return SymbolicTensor(graph=self, node=n)

    def const(self, value: float, dtype: str = "float32") -> SymbolicTensor:
        if dtype != "float32":
            raise ValueError("StaticGraph currently supports only float32.")
        n = self._graph.add_node(
            op_type=OpType.CONST,
            name="const",
            shape=(),
            dtype=dtype,
            attrs={"value": float(value)},
        )
        return SymbolicTensor(graph=self, node=n)

    def compile(self, *, use_hpc_template: bool = False, enable_memory_planner: bool = True) -> StaticCompiledGraph:
        if not self._graph.output_ids:
            raise ValueError("StaticGraph has no output. Call `mark_as_output()` on a SymbolicTensor.")
        module = JITEngine(
            use_hpc_template=use_hpc_template,
            enable_memory_planner=enable_memory_planner,
        ).compile_graph(self._graph)
        compiled = StaticCompiledGraph(module=module, input_names=[n.name for n in self._input_nodes])
        self._compiled = compiled
        return compiled

    def run(self, *inputs: np.ndarray, **named_inputs: np.ndarray) -> np.ndarray:
        if self._compiled is None:
            raise RuntimeError("StaticGraph is not compiled. Call `compile()` first.")
        return self._compiled.run(*inputs, **named_inputs)

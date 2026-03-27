"""Stable C++ code generation orchestration for traced graphs."""

from __future__ import annotations

from typing import Dict, List

from ..ir.graph import Graph, Node
from ..ir.ops import OpType
from .common import GeneratedKernel
from .cpp_emitter import CppEmitterMixin
from .memory_planner import MemoryPlannerMixin, MemoryPlannerState
from .op_lowering import OpLoweringMixin
from .shape_solver import ShapeSolverMixin


class CppCodegen(ShapeSolverMixin, MemoryPlannerMixin, OpLoweringMixin, CppEmitterMixin):
    """Generate C++ kernels from the stable Graph IR."""

    def __init__(
        self, graph: Graph, use_hpc_template: bool = False, enable_memory_planner: bool = True
    ) -> None:
        self.graph = graph
        self.use_hpc_template = use_hpc_template
        self.enable_memory_planner = enable_memory_planner
        self._sym: Dict[int, tuple[str, ...]] = {}
        self._sym_str: Dict[int, tuple[str, ...]] = {}
        self._compute_dtype: str | None = None
        self._requires_exact_input_shapes = False

    def generate(self) -> GeneratedKernel:
        ordered = self.graph.topological_sort()
        inputs = [node for node in ordered if node.op_type == OpType.INPUT]
        output_node = self._resolve_output(ordered)

        self._compute_dtype = output_node.dtype or "float32"
        self._build_symbolic_shapes(ordered, inputs)

        declarations: List[str] = []
        body: List[str] = []
        names: Dict[int, str] = {}
        ctype = "double" if self._compute_dtype == "float64" else "float"

        for idx, node in enumerate(inputs):
            names[node.id] = f"arg{idx}"

        use_count = self._compute_use_count(ordered)
        memory = MemoryPlannerState(names=names)

        for node in ordered:
            if node.op_type == OpType.INPUT:
                continue

            if node.op_type == OpType.CONST:
                const_name = f"const_{node.id}"
                names[node.id] = const_name
                value = float(node.attrs.get("value", 0.0))
                declarations.append(f"{ctype} {const_name}_val = {value};")
                declarations.append(f"{ctype}* {const_name} = &{const_name}_val;")
                continue

            self._assign_node_storage(node=node, output_node=output_node, state=memory)

            emit = self._emit_node(node=node, names=names)
            if emit:
                body.extend(emit)

            self._release_consumed_storage(node=node, use_count=use_count, state=memory)

        kernel = self._render_cpp(
            ordered=ordered,
            inputs=inputs,
            output_node=output_node,
            declarations=declarations,
            body=body,
            workspace_slots=memory.workspace_slots,
        )
        output_sym = self._sym.get(output_node.id, tuple(str(d) for d in output_node.shape))
        input_ranks = tuple(node.rank for node in inputs)
        return GeneratedKernel(
            source=kernel,
            entry="run_kernel",
            output_sym_shape=output_sym,
            input_ranks=input_ranks,
            workspace_slots=tuple(memory.workspace_sym),
            exact_input_shapes=(
                tuple(tuple(node.shape) for node in inputs)
                if self._requires_exact_input_shapes
                else ()
            ),
        )

    def _resolve_output(self, ordered: List[Node]) -> Node:
        if len(self.graph.output_ids) > 1:
            raise ValueError("Only single-output graphs are supported on the stable codegen path.")
        if self.graph.output_ids:
            return self.graph.get_node(self.graph.output_ids[-1])
        if not ordered:
            raise ValueError("Graph is empty.")
        return ordered[-1]


__all__ = ["CppCodegen", "GeneratedKernel"]

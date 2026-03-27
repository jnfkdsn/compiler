"""Operation dispatch for stable Graph IR lowering to emitted C++ statements."""

from __future__ import annotations

from ..ir.graph import Node
from ..ir.ops import OpType
from .common import CppStmt, render_cpp_stmts
from .elementwise_lowering import ElementwiseLoweringMixin
from .matmul_lowering import MatmulLoweringMixin
from .reduce_lowering import ReduceLoweringMixin


class OpLoweringMixin(ElementwiseLoweringMixin, ReduceLoweringMixin, MatmulLoweringMixin):
    @staticmethod
    def _cpp_type(node: Node) -> str:
        return "double" if node.dtype == "float64" else "float"

    @staticmethod
    def _cpp_zero(node: Node) -> str:
        return "0.0" if node.dtype == "float64" else "0.0f"

    @staticmethod
    def _cpp_neg_inf(node: Node) -> str:
        return "-DBL_MAX" if node.dtype == "float64" else "-FLT_MAX"

    @staticmethod
    def _cpp_vec_type(node: Node) -> str:
        return "double" if node.dtype == "float64" else "float"

    def _emit_structured(self, stmts: list[CppStmt]) -> list[str]:
        return render_cpp_stmts(stmts)

    def _emit_node(self, node: Node, names: dict[int, str]) -> list[str]:
        if node.op_type == OpType.ADD:
            return self._emit_add(node, names)
        if node.op_type == OpType.SUB:
            return self._emit_sub(node, names)
        if node.op_type == OpType.MUL:
            return self._emit_mul(node, names)
        if node.op_type == OpType.DIV:
            return self._emit_div(node, names)
        if node.op_type == OpType.EQ:
            return self._emit_eq(node, names)
        if node.op_type == OpType.RELU:
            return self._emit_relu(node, names)
        if node.op_type == OpType.RELU_GRAD:
            return self._emit_relu_grad(node, names)
        if node.op_type == OpType.EXP:
            return self._emit_exp(node, names)
        if node.op_type == OpType.LOG:
            return self._emit_log(node, names)
        if node.op_type == OpType.SIGMOID:
            return self._emit_sigmoid(node, names)
        if node.op_type == OpType.TRANSPOSE:
            return self._emit_transpose(node, names)
        if node.op_type == OpType.BROADCAST_TO:
            return self._emit_broadcast_to(node, names)
        if node.op_type == OpType.PACK:
            return self._emit_pack(node, names)
        if node.op_type == OpType.SUM:
            return self._emit_sum(node, names)
        if node.op_type == OpType.MEAN:
            return self._emit_mean(node, names)
        if node.op_type == OpType.MAX:
            return self._emit_max(node, names)
        if node.op_type == OpType.MATMUL:
            return self._emit_matmul(node, names)
        if node.op_type in (OpType.FUSED_MATMUL_BIAS, OpType.FUSED_MATMUL_BIAS_RELU):
            return self._emit_fused_matmul(node, names)
        raise NotImplementedError(f"Unsupported op in codegen: {node.op_type}")

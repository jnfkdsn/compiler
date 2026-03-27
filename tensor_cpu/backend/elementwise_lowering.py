"""Elementwise and broadcast-oriented lowering for the stable C++ backend."""

from __future__ import annotations

from typing import Dict, List

from ..ir.graph import Node
from .common import CppFor, CppLine, CppStmt


class ElementwiseLoweringMixin:
    def _emit_sub(self, node: Node, names: Dict[int, str]) -> List[str]:
        return self._emit_binary_elementwise(node=node, names=names, op="-")

    def _emit_add(self, node: Node, names: Dict[int, str]) -> List[str]:
        return self._emit_binary_elementwise(node=node, names=names, op="+")

    def _emit_binary_elementwise(self, *, node: Node, names: Dict[int, str], op: str) -> List[str]:
        lhs_node = self.graph.get_node(node.inputs[0])
        rhs_node = self.graph.get_node(node.inputs[1])
        lhs = names[node.inputs[0]]
        rhs = names[node.inputs[1]]
        dst = names[node.id]
        total = self._numel_expr(node)
        li = self._broadcast_index_expr(dst_node=node, src_node=lhs_node, linear_index_var="i")
        ri = self._broadcast_index_expr(dst_node=node, src_node=rhs_node, linear_index_var="i")
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[i] = {lhs}[{li}] {op} {rhs}[{ri}];")],
                )
            ]
        )

    def _emit_mul(self, node: Node, names: Dict[int, str]) -> List[str]:
        return self._emit_binary_elementwise(node=node, names=names, op="*")

    def _emit_div(self, node: Node, names: Dict[int, str]) -> List[str]:
        return self._emit_binary_elementwise(node=node, names=names, op="/")

    def _emit_eq(self, node: Node, names: Dict[int, str]) -> List[str]:
        lhs_node = self.graph.get_node(node.inputs[0])
        rhs_node = self.graph.get_node(node.inputs[1])
        lhs = names[node.inputs[0]]
        rhs = names[node.inputs[1]]
        dst = names[node.id]
        total = self._numel_expr(node)
        li = self._broadcast_index_expr(dst_node=node, src_node=lhs_node, linear_index_var="i")
        ri = self._broadcast_index_expr(dst_node=node, src_node=rhs_node, linear_index_var="i")
        one = "1.0" if node.dtype == "float64" else "1.0f"
        zero = self._cpp_zero(node)
        eps = "1e-12" if node.dtype == "float64" else "1e-6f"
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[
                        CppLine(
                            f"{dst}[i] = (std::fabs({lhs}[{li}] - {rhs}[{ri}]) <= {eps}) ? {one} : {zero};"
                        )
                    ],
                )
            ]
        )

    def _emit_relu(self, node: Node, names: Dict[int, str]) -> List[str]:
        src = names[node.inputs[0]]
        dst = names[node.id]
        total = self._numel_expr(node)
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[
                        CppLine(f"const {ctype} v = {src}[i];"),
                        CppLine(f"{dst}[i] = v > {zero} ? v : {zero};"),
                    ],
                )
            ]
        )

    def _emit_relu_grad(self, node: Node, names: Dict[int, str]) -> List[str]:
        src_node = self.graph.get_node(node.inputs[0])
        grad_node = self.graph.get_node(node.inputs[1])
        src = names[src_node.id]
        gout = names[grad_node.id]
        dst = names[node.id]
        total = self._numel_expr(node)
        si = self._broadcast_index_expr(dst_node=node, src_node=src_node, linear_index_var="i")
        gi = self._broadcast_index_expr(dst_node=node, src_node=grad_node, linear_index_var="i")
        zero = self._cpp_zero(node)
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[i] = {src}[{si}] > {zero} ? {gout}[{gi}] : {zero};")],
                )
            ]
        )

    def _emit_broadcast_to(self, node: Node, names: Dict[int, str]) -> List[str]:
        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        total = self._numel_expr(node)
        si = self._broadcast_index_expr(dst_node=node, src_node=src_node, linear_index_var="i")
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[i] = {src}[{si}];")],
                )
            ]
        )

    def _emit_pack(self, node: Node, names: Dict[int, str]) -> List[str]:
        dst = names[node.id]
        lines: List[str] = ["long long pack_offset = 0;"]
        for input_id in node.inputs:
            src_node = self.graph.get_node(input_id)
            src = names[input_id]
            numel = self._numel_expr(src_node)
            lines.extend(
                self._emit_structured(
                    [
                        CppFor(
                            init="long long i = 0",
                            cond=f"i < {numel}",
                            inc="++i",
                            body=[CppLine(f"{dst}[pack_offset + i] = {src}[i];")],
                        )
                    ]
                )
            )
            lines.append(f"pack_offset += {numel};")
        return lines

    def _emit_exp(self, node: Node, names: Dict[int, str]) -> List[str]:
        src = names[node.inputs[0]]
        dst = names[node.id]
        total = self._numel_expr(node)
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[i] = std::exp({src}[i]);")],
                )
            ]
        )

    def _emit_log(self, node: Node, names: Dict[int, str]) -> List[str]:
        src = names[node.inputs[0]]
        dst = names[node.id]
        total = self._numel_expr(node)
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[i] = std::log({src}[i]);")],
                )
            ]
        )

    def _emit_sigmoid(self, node: Node, names: Dict[int, str]) -> List[str]:
        src = names[node.inputs[0]]
        dst = names[node.id]
        total = self._numel_expr(node)
        ctype = self._cpp_type(node)
        one = "1.0" if node.dtype == "float64" else "1.0f"
        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[
                        CppLine(f"const {ctype} v = {src}[i];"),
                        CppLine(f"{dst}[i] = {one} / ({one} + std::exp(-v));"),
                    ],
                )
            ]
        )

    def _emit_transpose(self, node: Node, names: Dict[int, str]) -> List[str]:
        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        rank = len(self._sym.get(src_node.id, ()))
        total = self._numel_expr(node)

        if rank <= 1:
            return self._emit_structured(
                [
                    CppFor(
                        init="long long i = 0",
                        cond=f"i < {total}",
                        inc="++i",
                        body=[CppLine(f"{dst}[i] = {src}[i];")],
                    )
                ]
            )

        dst_strides = self._sym_str.get(node.id, ())
        src_strides = self._sym_str.get(src_node.id, ())

        body: List[CppStmt] = [CppLine("long long src_idx = 0;"), CppLine("long long rem = i;")]
        for d in range(rank):
            src_dim_idx = rank - 1 - d
            body.append(
                CppLine(f"long long c{d} = rem / ({dst_strides[d]}); rem %= ({dst_strides[d]});")
            )
            body.append(CppLine(f"src_idx += c{d} * ({src_strides[src_dim_idx]});"))
        body.append(CppLine(f"{dst}[i] = {src}[src_idx];"))

        return self._emit_structured(
            [CppFor(init="long long i = 0", cond=f"i < {total}", inc="++i", body=body)]
        )

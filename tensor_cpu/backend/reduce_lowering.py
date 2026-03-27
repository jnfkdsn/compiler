"""Reduction lowering for the stable C++ backend."""

from __future__ import annotations

from typing import Dict, List

from ..ir.graph import Node
from .common import CppFor, CppLine


class ReduceLoweringMixin:
    def _emit_sum(self, node: Node, names: Dict[int, str]) -> List[str]:
        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        total = self._numel_expr(src_node)
        out_total = self._numel_expr(node)
        axes = tuple(int(a) for a in node.attrs.get("axis", ()))
        dst_idx = self._reduce_dst_index_expr(
            src_node=src_node, dst_node=node, axes=axes, idx_var="i"
        )
        zero = self._cpp_zero(node)
        return self._emit_structured(
            [
                CppFor(
                    init="long long o = 0",
                    cond=f"o < {out_total}",
                    inc="++o",
                    body=[CppLine(f"{dst}[o] = {zero};")],
                ),
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[{dst_idx}] += {src}[i];")],
                ),
            ]
        )

    def _emit_mean(self, node: Node, names: Dict[int, str]) -> List[str]:
        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        total = self._numel_expr(src_node)
        out_total = self._numel_expr(node)
        axes = tuple(int(a) for a in node.attrs.get("axis", ()))
        dst_idx = self._reduce_dst_index_expr(
            src_node=src_node, dst_node=node, axes=axes, idx_var="i"
        )
        src_sym = self._sym.get(src_node.id, ())
        reduce_parts = [src_sym[a] for a in axes if a < len(src_sym)]
        if not reduce_parts:
            reduce_size_expr = "1"
        elif len(reduce_parts) == 1:
            reduce_size_expr = reduce_parts[0]
        else:
            reduce_size_expr = " * ".join(f"({d})" for d in reduce_parts)
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)
        div_expr = f"({ctype})({reduce_size_expr})"
        return self._emit_structured(
            [
                CppFor(
                    init="long long o = 0",
                    cond=f"o < {out_total}",
                    inc="++o",
                    body=[CppLine(f"{dst}[o] = {zero};")],
                ),
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[{dst_idx}] += {src}[i];")],
                ),
                CppFor(
                    init="long long o = 0",
                    cond=f"o < {out_total}",
                    inc="++o",
                    body=[CppLine(f"{dst}[o] /= {div_expr};")],
                ),
            ]
        )

    def _emit_max(self, node: Node, names: Dict[int, str]) -> List[str]:
        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        total = self._numel_expr(src_node)
        out_total = self._numel_expr(node)
        axes = tuple(int(a) for a in node.attrs.get("axis", ()))
        dst_idx = self._reduce_dst_index_expr(
            src_node=src_node, dst_node=node, axes=axes, idx_var="i"
        )
        return self._emit_structured(
            [
                CppFor(
                    init="long long o = 0",
                    cond=f"o < {out_total}",
                    inc="++o",
                    body=[CppLine(f"{dst}[o] = {self._cpp_neg_inf(node)};")],
                ),
                CppFor(
                    init="long long i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[
                        CppLine(f"const long long di = {dst_idx};"),
                        CppLine(f"if ({src}[i] > {dst}[di]) {dst}[di] = {src}[i];"),
                    ],
                ),
            ]
        )

"""Symbolic shape propagation helpers for the stable C++ backend."""

from __future__ import annotations

from ..ir.graph import Node
from ..ir.ops import OpType


class ShapeSolverMixin:
    _sym: dict[int, tuple[str, ...]]
    _sym_str: dict[int, tuple[str, ...]]

    @staticmethod
    def _sym_strides_from(dims: tuple[str, ...]) -> tuple[str, ...]:
        if not dims:
            return ()
        strides = ["1"] * len(dims)
        for i in range(len(dims) - 2, -1, -1):
            if strides[i + 1] == "1":
                strides[i] = f"{dims[i + 1]}"
            else:
                strides[i] = f"({strides[i + 1]} * {dims[i + 1]})"
        return tuple(strides)

    @staticmethod
    def _sym_broadcast(lhs: tuple[str, ...], rhs: tuple[str, ...]) -> tuple[str, ...]:
        rank = max(len(lhs), len(rhs))
        lp = ("1",) * (rank - len(lhs)) + lhs
        rp = ("1",) * (rank - len(rhs)) + rhs
        return tuple(r if l == "1" else l for l, r in zip(lp, rp))

    def _build_symbolic_shapes(self, ordered: list[Node], inputs: list[Node]) -> None:
        for idx, node in enumerate(inputs):
            dims = tuple(f"in{idx}_d{d}" for d in range(node.rank))
            self._sym[node.id] = dims
            self._sym_str[node.id] = self._sym_strides_from(dims)

        for node in ordered:
            if node.id in self._sym:
                continue
            self._propagate_symbolic(node)

    def _propagate_symbolic(self, node: Node) -> None:
        if node.op_type == OpType.CONST:
            dims = tuple(str(d) for d in node.shape)
            self._sym[node.id] = dims
            self._sym_str[node.id] = self._sym_strides_from(dims)
            return

        if node.op_type in (
            OpType.ADD,
            OpType.SUB,
            OpType.MUL,
            OpType.DIV,
            OpType.RELU_GRAD,
            OpType.EQ,
        ):
            lhs = self._sym.get(node.inputs[0], ())
            rhs = self._sym.get(node.inputs[1], ())
            dims = self._sym_broadcast(lhs, rhs)
            self._sym[node.id] = dims
            self._sym_str[node.id] = self._sym_strides_from(dims)
            return

        if node.op_type in (OpType.RELU, OpType.EXP, OpType.LOG, OpType.SIGMOID):
            self._sym[node.id] = self._sym.get(node.inputs[0], ())
            self._sym_str[node.id] = self._sym_str.get(node.inputs[0], ())
            return

        if node.op_type == OpType.TRANSPOSE:
            src = self._sym.get(node.inputs[0], ())
            dims = tuple(reversed(src))
            self._sym[node.id] = dims
            self._sym_str[node.id] = self._sym_strides_from(dims)
            return

        if node.op_type == OpType.MATMUL:
            a = self._sym.get(node.inputs[0], ())
            b = self._sym.get(node.inputs[1], ())
            ba, bb = a[:-2], b[:-2]
            batch = self._sym_broadcast(ba, bb) if (ba or bb) else ()
            dims = batch + (a[-2], b[-1])
            self._sym[node.id] = dims
            self._sym_str[node.id] = self._sym_strides_from(dims)
            return

        if node.op_type in (OpType.FUSED_MATMUL_BIAS, OpType.FUSED_MATMUL_BIAS_RELU):
            a = self._sym.get(node.inputs[0], ())
            b = self._sym.get(node.inputs[1], ())
            if len(a) >= 2 and len(b) >= 2:
                ba, bb = a[:-2], b[:-2]
                batch = self._sym_broadcast(ba, bb) if (ba or bb) else ()
                dims = batch + (a[-2], b[-1])
            else:
                dims = tuple(str(d) for d in node.shape)
            self._sym[node.id] = dims
            self._sym_str[node.id] = self._sym_strides_from(dims)
            return

        if node.op_type in (OpType.SUM, OpType.MEAN, OpType.MAX):
            src = self._sym.get(node.inputs[0], ())
            axes_set = {int(a) for a in node.attrs.get("axis", ())}
            keepdims = bool(node.attrs.get("keepdims", False))
            dims: list[str] = []
            for i, d in enumerate(src):
                if i in axes_set:
                    if keepdims:
                        dims.append("1")
                else:
                    dims.append(d)
            self._sym[node.id] = tuple(dims)
            self._sym_str[node.id] = self._sym_strides_from(tuple(dims))
            return

        if node.op_type == OpType.BROADCAST_TO:
            src_node = self.graph.get_node(node.inputs[0])
            src_sym = self._sym.get(node.inputs[0], ())
            target = tuple(int(d) for d in node.attrs.get("target_shape", node.shape))

            if src_sym and tuple(d for d in target if d != 1) == tuple(src_node.shape):
                dims_exp: list[str] = []
                si = 0
                for td in target:
                    if td == 1:
                        dims_exp.append("1")
                    else:
                        dims_exp.append(src_sym[si])
                        si += 1
                sym = tuple(dims_exp)
                self._sym[node.id] = sym
                self._sym_str[node.id] = self._sym_strides_from(sym)
                return

            for nid, sym in self._sym.items():
                other = self.graph.get_node(nid)
                if other.shape == target:
                    self._sym[node.id] = sym
                    self._sym_str[node.id] = self._sym_strides_from(sym)
                    return

            dims_lit = tuple(str(d) for d in target)
            self._sym[node.id] = dims_lit
            self._sym_str[node.id] = self._sym_strides_from(dims_lit)
            return

        dims_lit = tuple(str(d) for d in node.shape)
        self._sym[node.id] = dims_lit
        self._sym_str[node.id] = self._sym_strides_from(dims_lit)

    def _numel_expr(self, node: Node) -> str:
        dims = self._sym.get(node.id)
        if dims is None:
            return str(node.numel)
        if not dims:
            return "1"
        if len(dims) == 1:
            return f"((long long)({dims[0]}))"
        return "(" + " * ".join(f"(long long)({d})" for d in dims) + ")"

    def _broadcast_index_expr(
        self, *, dst_node: Node, src_node: Node, linear_index_var: str
    ) -> str:
        src_sym = self._sym.get(src_node.id, ())
        dst_sym = self._sym.get(dst_node.id, ())

        if src_sym == dst_sym:
            return linear_index_var

        if not src_sym or all(d == "1" for d in src_sym):
            return "0"

        dst_non1 = tuple(d for d in dst_sym if d != "1")
        if len(dst_sym) >= len(src_sym) and dst_non1 == src_sym:
            dst_strides = self._sym_strides_from(dst_sym)
            src_strides = self._sym_strides_from(src_sym)
            dst_axes = [i for i, d in enumerate(dst_sym) if d != "1"]
            terms: list[str] = []
            for src_axis, dst_axis in enumerate(dst_axes):
                coord_expr = (
                    f"(({linear_index_var} / ({dst_strides[dst_axis]})) % ({dst_sym[dst_axis]}))"
                )
                src_stride = src_strides[src_axis]
                if src_stride == "1":
                    terms.append(coord_expr)
                else:
                    terms.append(f"({coord_expr} * ({src_stride}))")
            if terms:
                return " + ".join(terms)
            return "0"

        if len(src_sym) == 1 and len(dst_sym) >= 1 and src_sym[0] == dst_sym[-1]:
            return f"({linear_index_var} % ({dst_sym[-1]}))"

        aligned_src = ("1",) * (len(dst_sym) - len(src_sym)) + src_sym

        def _suffix_product_sym(dims: tuple[str, ...], start: int) -> str:
            parts = dims[start:]
            if not parts:
                return "1"
            if len(parts) == 1:
                return parts[0]
            return " * ".join(f"({d})" for d in parts)

        terms: list[str] = []
        for axis, src_d in enumerate(aligned_src):
            if src_d == "1":
                continue
            dst_stride = _suffix_product_sym(dst_sym, axis + 1)
            src_stride = _suffix_product_sym(aligned_src, axis + 1)
            coord_expr = f"(({linear_index_var} / ({dst_stride})) % ({dst_sym[axis]}))"
            if src_stride == "1":
                terms.append(coord_expr)
            else:
                terms.append(f"({coord_expr} * ({src_stride}))")

        if not terms:
            return "0"
        return " + ".join(terms)

    def _reduce_dst_index_expr(
        self, *, src_node: Node, dst_node: Node, axes: tuple[int, ...], idx_var: str
    ) -> str:
        dst_sym = self._sym.get(dst_node.id, ())
        if not dst_sym:
            return "0"

        axes_set = set(axes)
        src_sym = self._sym.get(src_node.id, ())
        src_strides_sym = self._sym_str.get(src_node.id, ())
        dst_strides_sym = self._sym_str.get(dst_node.id, ())

        keepdims = bool(dst_node.attrs.get("keepdims", False))
        terms: list[str] = []
        dst_axis = 0
        for src_axis, src_d in enumerate(src_sym):
            if src_axis in axes_set:
                if keepdims:
                    dst_axis += 1
                continue
            src_s = src_strides_sym[src_axis] if src_axis < len(src_strides_sym) else "1"
            coord_expr = f"(({idx_var} / ({src_s})) % ({src_d}))"
            dst_s = dst_strides_sym[dst_axis] if dst_axis < len(dst_strides_sym) else "1"
            if dst_s == "1":
                terms.append(coord_expr)
            else:
                terms.append(f"({coord_expr} * ({dst_s}))")
            dst_axis += 1

        if not terms:
            return "0"
        return " + ".join(terms)

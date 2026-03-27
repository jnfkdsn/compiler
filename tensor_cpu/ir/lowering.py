"""Lowering from Graph IR to Tensor IR (TIR).

This module converts high-level Graph IR operations into TIR loop nests.
Each operation is lowered into explicit loop structures with buffer accesses.

The lowering process:
1. Create TIR buffers for all tensors
2. Convert each Graph Node into TIR statements
3. Build loop nests for operations like MatMul, element-wise ops, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import Graph, Node
from .ops import OpType
from .tir import (
    Binary,
    Block,
    Buffer,
    BufferLoad,
    BufferStore,
    CallExpr,
    Const,
    For,
    IfStmt,
    IRModule,
    LetStmt,
    PrimFunc,
    Ternary,
    TIRExpr,
    TIRStmt,
    Unary,
    Var,
)


@dataclass(slots=True)
class LoweringContext:
    """Context for the lowering process."""

    graph: Graph
    sym_shapes: dict[int, tuple[str, ...]] = field(default_factory=dict)
    sym_strides: dict[int, tuple[str, ...]] = field(default_factory=dict)
    buffers: dict[int, Buffer] = field(default_factory=dict)
    var_names: dict[int, str] = field(default_factory=dict)
    compute_dtype: str = "float32"


class GraphLowering:
    """Lower Graph IR to TIR."""

    def __init__(
        self,
        graph: Graph,
        sym_shapes: dict[int, tuple[str, ...]] = None,
        sym_strides: dict[int, tuple[str, ...]] = None,
    ) -> None:
        self.ctx = LoweringContext(
            graph=graph,
            sym_shapes=sym_shapes or {},
            sym_strides=sym_strides or {},
        )

    def lower(self) -> IRModule:
        """Lower the entire graph to an IRModule."""
        ordered = self.ctx.graph.topological_sort()
        inputs = [n for n in ordered if n.op_type == OpType.INPUT]
        output_node = self._resolve_output(ordered)

        self.ctx.compute_dtype = output_node.dtype or "float32"

        for idx, node in enumerate(inputs):
            self.ctx.var_names[node.id] = f"arg{idx}"
            self._create_buffer(node)

        for node in ordered:
            if node.id in self.ctx.buffers:
                continue
            self._create_buffer(node)

        stmts: list[TIRStmt] = []
        for node in ordered:
            if node.op_type == OpType.INPUT:
                continue
            if node.op_type == OpType.CONST:
                stmts.extend(self._lower_const(node))
            else:
                stmts.extend(self._lower_node(node))

        params = [Var(f"arg{i}", self.ctx.compute_dtype) for i in range(len(inputs))]
        params.append(Var("out_ptr", self.ctx.compute_dtype))

        func = PrimFunc(
            name="run_kernel",
            params=params,
            body=Block(stmts=stmts),
            buffers=self.ctx.buffers,
        )

        return IRModule(functions={"run_kernel": func})

    def _resolve_output(self, ordered: list[Node]) -> Node:
        if self.ctx.graph.output_ids:
            return self.ctx.graph.get_node(self.ctx.graph.output_ids[-1])
        if not ordered:
            raise ValueError("Graph is empty.")
        return ordered[-1]

    def _create_buffer(self, node: Node) -> None:
        sym = self.ctx.sym_shapes.get(node.id, tuple(str(d) for d in node.shape))
        strides = self.ctx.sym_strides.get(node.id, self._compute_strides(sym))

        shape_exprs = [Const(int(d)) if d.isdigit() else Var(d) for d in sym]
        stride_exprs = [Const(int(s)) if s.isdigit() else Var(s) for s in strides]

        buffer = Buffer(
            name=self.ctx.var_names.get(node.id, f"buf_{node.id}"),
            shape=shape_exprs,
            dtype=node.dtype or self.ctx.compute_dtype,
            strides=stride_exprs,
        )
        self.ctx.buffers[node.id] = buffer

    def _compute_strides(self, dims: tuple[str, ...]) -> tuple[str, ...]:
        if not dims:
            return ()
        strides = ["1"] * len(dims)
        for i in range(len(dims) - 2, -1, -1):
            if strides[i + 1] == "1":
                strides[i] = dims[i + 1]
            else:
                strides[i] = f"({strides[i + 1]} * {dims[i + 1]})"
        return tuple(strides)

    def _lower_const(self, node: Node) -> list[TIRStmt]:
        return []

    def _lower_node(self, node: Node) -> list[TIRStmt]:
        op = node.op_type
        if op in (OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV):
            return self._lower_binary_elementwise(node)
        if op == OpType.EQ:
            return self._lower_eq(node)
        if op == OpType.RELU:
            return self._lower_relu(node)
        if op == OpType.RELU_GRAD:
            return self._lower_relu_grad(node)
        if op in (OpType.EXP, OpType.LOG, OpType.SIGMOID):
            return self._lower_unary_math(node)
        if op == OpType.TRANSPOSE:
            return self._lower_transpose(node)
        if op == OpType.BROADCAST_TO:
            return self._lower_broadcast_to(node)
        if op in (OpType.SUM, OpType.MEAN, OpType.MAX):
            return self._lower_reduce(node)
        if op == OpType.MATMUL:
            return self._lower_matmul(node)
        if op in (OpType.FUSED_MATMUL_BIAS, OpType.FUSED_MATMUL_BIAS_RELU):
            return self._lower_fused_matmul(node)
        raise NotImplementedError(f"Unsupported op in lowering: {op}")

    def _lower_binary_elementwise(self, node: Node) -> list[TIRStmt]:
        lhs_node = self.ctx.graph.get_node(node.inputs[0])
        rhs_node = self.ctx.graph.get_node(node.inputs[1])
        op_map = {OpType.ADD: "+", OpType.SUB: "-", OpType.MUL: "*", OpType.DIV: "/"}

        i_var = Var("i", "int64")
        total = self._numel_expr(node)

        lhs_load = self._create_load_with_broadcast(lhs_node, node, i_var)
        rhs_load = self._create_load_with_broadcast(rhs_node, node, i_var)
        op_str = op_map[node.op_type]

        store = self._create_store(node, [i_var], Binary(lhs_load, op_str, rhs_load))

        loop = For(
            loop_var=i_var,
            start=Const(0),
            stop=total,
            body=Block(stmts=[store]),
        )
        return [loop]

    def _lower_eq(self, node: Node) -> list[TIRStmt]:
        lhs_node = self.ctx.graph.get_node(node.inputs[0])
        rhs_node = self.ctx.graph.get_node(node.inputs[1])

        i_var = Var("i", "int64")
        total = self._numel_expr(node)

        lhs_load = self._create_load_with_broadcast(lhs_node, node, i_var)
        rhs_load = self._create_load_with_broadcast(rhs_node, node, i_var)

        eps = Const(1e-6 if node.dtype == "float32" else 1e-12)
        diff = Binary(lhs_load, "-", rhs_load)
        abs_diff = CallExpr("std::fabs", [diff])
        cond = Binary(abs_diff, "<=", eps)

        one = Const(1.0)
        zero = Const(0.0)
        value = Ternary(cond, one, zero)

        store = self._create_store(node, [i_var], value)

        loop = For(
            loop_var=i_var,
            start=Const(0),
            stop=total,
            body=Block(stmts=[store]),
        )
        return [loop]

    def _lower_relu(self, node: Node) -> list[TIRStmt]:
        src_node = self.ctx.graph.get_node(node.inputs[0])
        i_var = Var("i", "int64")
        total = self._numel_expr(node)

        src_load = self._create_load_with_broadcast(src_node, node, i_var)
        zero = Const(0.0)
        value = Ternary(Binary(src_load, ">", zero), src_load, zero)

        store = self._create_store(node, [i_var], value)

        loop = For(
            loop_var=i_var,
            start=Const(0),
            stop=total,
            body=Block(stmts=[store]),
        )
        return [loop]

    def _lower_relu_grad(self, node: Node) -> list[TIRStmt]:
        src_node = self.ctx.graph.get_node(node.inputs[0])
        grad_node = self.ctx.graph.get_node(node.inputs[1])

        i_var = Var("i", "int64")
        total = self._numel_expr(node)

        src_load = self._create_load_with_broadcast(src_node, node, i_var)
        grad_load = self._create_load_with_broadcast(grad_node, node, i_var)
        zero = Const(0.0)

        value = Ternary(
            Binary(src_load, ">", zero),
            grad_load,
            zero,
        )

        store = self._create_store(node, [i_var], value)

        loop = For(
            loop_var=i_var,
            start=Const(0),
            stop=total,
            body=Block(stmts=[store]),
        )
        return [loop]

    def _lower_unary_math(self, node: Node) -> list[TIRStmt]:
        src_node = self.ctx.graph.get_node(node.inputs[0])
        i_var = Var("i", "int64")
        total = self._numel_expr(node)

        src_load = self._create_load_with_broadcast(src_node, node, i_var)

        func_map = {
            OpType.EXP: "std::exp",
            OpType.LOG: "std::log",
        }

        if node.op_type == OpType.SIGMOID:
            one = Const(1.0)
            neg_v = Unary("-", src_load)
            exp_neg = CallExpr("std::exp", [neg_v])
            value = Binary(one, "/", Binary(one, "+", exp_neg))
        else:
            value = CallExpr(func_map[node.op_type], [src_load])

        store = self._create_store(node, [i_var], value)

        loop = For(
            loop_var=i_var,
            start=Const(0),
            stop=total,
            body=Block(stmts=[store]),
        )
        return [loop]

    def _lower_transpose(self, node: Node) -> list[TIRStmt]:
        src_node = self.ctx.graph.get_node(node.inputs[0])
        src_sym = self.ctx.sym_shapes.get(src_node.id, ())
        rank = len(src_sym)

        if rank <= 1:
            i_var = Var("i", "int64")
            total = self._numel_expr(node)
            src_load = BufferLoad(self.ctx.buffers[src_node.id], [i_var])
            store = self._create_store(node, [i_var], src_load)
            return [For(i_var, Const(0), total, Block([store]))]

        i_var = Var("i", "int64")
        total = self._numel_expr(node)

        dst_strides = self.ctx.sym_strides.get(node.id, ())
        src_strides = self.ctx.sym_strides.get(src_node.id, ())

        stmts: list[TIRStmt] = [
            LetStmt(Var("src_idx", "int64"), Const(0), Block([])),
            LetStmt(Var("rem", "int64"), i_var, Block([])),
        ]

        src_idx_var = Var("src_idx")
        rem_var = Var("rem")

        inner_stmts: list[TIRStmt] = []
        for d in range(rank):
            dst_stride = dst_strides[d]
            c_var = Var(f"c{d}", "int64")
            dst_stride_expr = Const(int(dst_stride)) if dst_stride.isdigit() else Var(dst_stride)
            self._get_dim_expr(node, d)

            coord_expr = Binary(rem_var, "/", dst_stride_expr)
            inner_stmts.append(LetStmt(c_var, coord_expr, Block([])))
            inner_stmts.append(
                LetStmt(
                    rem_var,
                    Binary(rem_var, "%", dst_stride_expr),
                    Block([]),
                )
            )

            src_dim_idx = rank - 1 - d
            src_stride = src_strides[src_dim_idx]
            src_stride_expr = Const(int(src_stride)) if src_stride.isdigit() else Var(src_stride)
            inner_stmts.append(
                LetStmt(
                    src_idx_var,
                    Binary(src_idx_var, "+", Binary(c_var, "*", src_stride_expr)),
                    Block([]),
                )
            )

        src_load = BufferLoad(self.ctx.buffers[src_node.id], [src_idx_var])
        store = self._create_store(node, [i_var], src_load)
        inner_stmts.append(store)

        body = Block(inner_stmts)
        for stmt in reversed(stmts):
            if isinstance(stmt, LetStmt):
                body = LetStmt(stmt.var, stmt.value, body)

        return [For(i_var, Const(0), total, body)]

    def _lower_broadcast_to(self, node: Node) -> list[TIRStmt]:
        src_node = self.ctx.graph.get_node(node.inputs[0])
        i_var = Var("i", "int64")
        total = self._numel_expr(node)

        src_load = self._create_load_with_broadcast(src_node, node, i_var)
        store = self._create_store(node, [i_var], src_load)

        loop = For(
            loop_var=i_var,
            start=Const(0),
            stop=total,
            body=Block(stmts=[store]),
        )
        return [loop]

    def _lower_reduce(self, node: Node) -> list[TIRStmt]:
        src_node = self.ctx.graph.get_node(node.inputs[0])
        axes = tuple(int(a) for a in node.attrs.get("axis", ()))

        total = self._numel_expr(src_node)
        out_total = self._numel_expr(node)

        i_var = Var("i", "int64")
        o_var = Var("o", "int64")

        init_val = Const(0.0)
        if node.op_type == OpType.MAX:
            init_val = Const(float("-inf"))

        init_store = self._create_store(node, [o_var], init_val)
        init_loop = For(o_var, Const(0), out_total, Block([init_store]))

        dst_idx = self._reduce_dst_index_expr(src_node, node, axes, i_var)
        src_load = BufferLoad(self.ctx.buffers[src_node.id], [i_var])

        if node.op_type == OpType.MAX:
            update = IfStmt(
                Binary(src_load, ">", BufferLoad(self.ctx.buffers[node.id], [dst_idx])),
                self._create_store(node, [dst_idx], src_load),
            )
        else:
            current = BufferLoad(self.ctx.buffers[node.id], [dst_idx])
            update = self._create_store(node, [dst_idx], Binary(current, "+", src_load))

        reduce_loop = For(i_var, Const(0), total, Block([update]))

        stmts = [init_loop, reduce_loop]

        if node.op_type == OpType.MEAN:
            src_sym = self.ctx.sym_shapes.get(src_node.id, ())
            reduce_parts = [src_sym[a] for a in axes if a < len(src_sym)]
            if reduce_parts:
                if len(reduce_parts) == 1:
                    reduce_size = (
                        Var(reduce_parts[0])
                        if not reduce_parts[0].isdigit()
                        else Const(int(reduce_parts[0]))
                    )
                else:
                    reduce_size = Const(1)
                    for p in reduce_parts:
                        part = Var(p) if not p.isdigit() else Const(int(p))
                        reduce_size = Binary(reduce_size, "*", part)

                final_load = BufferLoad(self.ctx.buffers[node.id], [o_var])
                div_store = self._create_store(node, [o_var], Binary(final_load, "/", reduce_size))
                final_loop = For(o_var, Const(0), out_total, Block([div_store]))
                stmts.append(final_loop)

        return stmts

    def _lower_matmul(self, node: Node) -> list[TIRStmt]:
        a_node = self.ctx.graph.get_node(node.inputs[0])
        b_node = self.ctx.graph.get_node(node.inputs[1])

        a_sym = self.ctx.sym_shapes.get(a_node.id, ())
        b_sym = self.ctx.sym_shapes.get(b_node.id, ())

        M = a_sym[-2] if len(a_sym) >= 2 else "1"
        K = a_sym[-1] if len(a_sym) >= 1 else "1"
        N = b_sym[-1] if len(b_sym) >= 1 else "1"

        M_expr = Var(M) if not M.isdigit() else Const(int(M))
        K_expr = Var(K) if not K.isdigit() else Const(int(K))
        N_expr = Var(N) if not N.isdigit() else Const(int(N))

        i_var = Var("i", "int64")
        j_var = Var("j", "int64")
        k_var = Var("kk", "int64")
        acc_var = Var("acc", self.ctx.compute_dtype)

        a_load = BufferLoad(
            self.ctx.buffers[a_node.id], [Binary(Binary(i_var, "*", K_expr), "+", k_var)]
        )
        b_load = BufferLoad(
            self.ctx.buffers[b_node.id], [Binary(Binary(k_var, "*", N_expr), "+", j_var)]
        )
        c_idx = Binary(Binary(i_var, "*", N_expr), "+", j_var)

        inner_body = Block(
            [
                LetStmt(acc_var, Binary(acc_var, "+", Binary(a_load, "*", b_load)), Block([])),
            ]
        )

        k_loop = For(k_var, Const(0), K_expr, inner_body)

        store = self._create_store(node, [c_idx], acc_var)

        j_body = Block(
            [
                LetStmt(acc_var, Const(0.0), Block([k_loop, store])),
            ]
        )

        j_loop = For(j_var, Const(0), N_expr, j_body)
        i_loop = For(i_var, Const(0), M_expr, Block([j_loop]))

        return [i_loop]

    def _lower_fused_matmul(self, node: Node) -> list[TIRStmt]:
        a_node = self.ctx.graph.get_node(node.inputs[0])
        b_node = self.ctx.graph.get_node(node.inputs[1])
        bias_node = self.ctx.graph.get_node(node.inputs[2])

        a_sym = self.ctx.sym_shapes.get(a_node.id, ())
        b_sym = self.ctx.sym_shapes.get(b_node.id, ())

        M = a_sym[-2] if len(a_sym) >= 2 else "1"
        K = a_sym[-1] if len(a_sym) >= 1 else "1"
        N = b_sym[-1] if len(b_sym) >= 1 else "1"

        M_expr = Var(M) if not M.isdigit() else Const(int(M))
        K_expr = Var(K) if not K.isdigit() else Const(int(K))
        N_expr = Var(N) if not N.isdigit() else Const(int(N))

        i_var = Var("i", "int64")
        j_var = Var("j", "int64")
        k_var = Var("kk", "int64")
        acc_var = Var("acc", self.ctx.compute_dtype)
        v_var = Var("v", self.ctx.compute_dtype)

        a_load = BufferLoad(
            self.ctx.buffers[a_node.id], [Binary(Binary(i_var, "*", K_expr), "+", k_var)]
        )
        b_load = BufferLoad(
            self.ctx.buffers[b_node.id], [Binary(Binary(k_var, "*", N_expr), "+", j_var)]
        )
        bias_load = BufferLoad(self.ctx.buffers[bias_node.id], [j_var])
        c_idx = Binary(Binary(i_var, "*", N_expr), "+", j_var)

        k_body = Block(
            [
                LetStmt(acc_var, Binary(acc_var, "+", Binary(a_load, "*", b_load)), Block([])),
            ]
        )
        k_loop = For(k_var, Const(0), K_expr, k_body)

        with_relu = node.op_type == OpType.FUSED_MATMUL_BIAS_RELU
        v_expr = Binary(acc_var, "+", bias_load)

        if with_relu:
            zero = Const(0.0)
            v_expr = Ternary(Binary(v_expr, ">", zero), v_expr, zero)

        store = self._create_store(node, [c_idx], v_expr)

        j_body = Block(
            [
                LetStmt(
                    acc_var, Const(0.0), Block([k_loop, LetStmt(v_var, v_expr, Block([store]))])
                ),
            ]
        )

        j_loop = For(j_var, Const(0), N_expr, j_body)
        i_loop = For(i_var, Const(0), M_expr, Block([j_loop]))

        return [i_loop]

    def _numel_expr(self, node: Node) -> TIRExpr:
        dims = self.ctx.sym_shapes.get(node.id, tuple(str(d) for d in node.shape))
        if not dims:
            return Const(1)
        if len(dims) == 1:
            d = dims[0]
            return Const(int(d)) if d.isdigit() else Var(d)
        result: TIRExpr = Const(1)
        for d in dims:
            part = Const(int(d)) if d.isdigit() else Var(d)
            result = Binary(result, "*", part)
        return result

    def _create_load_with_broadcast(
        self, src_node: Node, dst_node: Node, linear_idx: Var
    ) -> TIRExpr:
        src_sym = self.ctx.sym_shapes.get(src_node.id, ())
        dst_sym = self.ctx.sym_shapes.get(dst_node.id, ())

        if src_sym == dst_sym:
            return BufferLoad(self.ctx.buffers[src_node.id], [linear_idx])

        if not src_sym or all(d == "1" for d in src_sym):
            return BufferLoad(self.ctx.buffers[src_node.id], [Const(0)])

        if len(src_sym) == 1 and len(dst_sym) >= 1 and src_sym[0] == dst_sym[-1]:
            idx = Binary(
                linear_idx,
                "%",
                Var(dst_sym[-1]) if not dst_sym[-1].isdigit() else Const(int(dst_sym[-1])),
            )
            return BufferLoad(self.ctx.buffers[src_node.id], [idx])

        src_idx = self._broadcast_index_expr(src_node, dst_node, linear_idx)
        return BufferLoad(self.ctx.buffers[src_node.id], [src_idx])

    def _broadcast_index_expr(self, src_node: Node, dst_node: Node, linear_idx: Var) -> TIRExpr:
        src_sym = self.ctx.sym_shapes.get(src_node.id, ())
        dst_sym = self.ctx.sym_shapes.get(dst_node.id, ())
        dst_strides = self.ctx.sym_strides.get(dst_node.id, ())

        aligned_src = ("1",) * (len(dst_sym) - len(src_sym)) + src_sym

        terms: list[TIRExpr] = []
        for axis, src_d in enumerate(aligned_src):
            if src_d == "1":
                continue
            dst_stride = dst_strides[axis] if axis < len(dst_strides) else "1"
            dst_dim = dst_sym[axis] if axis < len(dst_sym) else "1"

            stride_expr = Const(int(dst_stride)) if dst_stride.isdigit() else Var(dst_stride)
            dim_expr = Const(int(dst_dim)) if dst_dim.isdigit() else Var(dst_dim)

            coord_expr = Binary(Binary(linear_idx, "/", stride_expr), "%", dim_expr)
            terms.append(coord_expr)

        if not terms:
            return Const(0)

        result = terms[0]
        for t in terms[1:]:
            result = Binary(result, "+", t)
        return result

    def _reduce_dst_index_expr(
        self, src_node: Node, dst_node: Node, axes: tuple[int, ...], idx_var: Var
    ) -> TIRExpr:
        dst_sym = self.ctx.sym_shapes.get(dst_node.id, ())
        if not dst_sym:
            return Const(0)

        src_sym = self.ctx.sym_shapes.get(src_node.id, ())
        src_strides = self.ctx.sym_strides.get(src_node.id, ())
        dst_strides = self.ctx.sym_strides.get(dst_node.id, ())

        axes_set = set(axes)
        terms: list[TIRExpr] = []
        dst_axis = 0

        for src_axis, src_d in enumerate(src_sym):
            if src_axis in axes_set:
                continue
            src_s = src_strides[src_axis] if src_axis < len(src_strides) else "1"
            src_dim = src_d

            stride_expr = Const(int(src_s)) if src_s.isdigit() else Var(src_s)
            dim_expr = Const(int(src_dim)) if src_dim.isdigit() else Var(src_dim)

            coord_expr = Binary(Binary(idx_var, "/", stride_expr), "%", dim_expr)

            dst_s = dst_strides[dst_axis] if dst_axis < len(dst_strides) else "1"
            dst_stride_expr = Const(int(dst_s)) if dst_s.isdigit() else Var(dst_s)

            if dst_s == "1":
                terms.append(coord_expr)
            else:
                terms.append(Binary(coord_expr, "*", dst_stride_expr))
            dst_axis += 1

        if not terms:
            return Const(0)

        result = terms[0]
        for t in terms[1:]:
            result = Binary(result, "+", t)
        return result

    def _create_store(self, node: Node, indices: list[TIRExpr], value: TIRExpr) -> BufferStore:
        return BufferStore(
            buffer=self.ctx.buffers[node.id],
            value=value,
            indices=indices,
        )

    def _get_dim_expr(self, node: Node, axis: int) -> TIRExpr:
        sym = self.ctx.sym_shapes.get(node.id, ())
        if axis < len(sym):
            d = sym[axis]
            return Const(int(d)) if d.isdigit() else Var(d)
        return Const(1)

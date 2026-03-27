"""C++ code generation using layered IR architecture.

This module provides the new code generation pipeline:
1. Graph IR -> TIR (Tensor IR) via lowering
2. TIR -> C++ AST via conversion
3. C++ AST -> C++ source code via Visitor pattern

This design enables:
- Instruction-level optimizations (loop unrolling, vectorization)
- Better separation of concerns
- Easier maintenance and extensibility
"""

from __future__ import annotations

from dataclasses import dataclass

from tensor_cpu.abi import AbiStatus
from tensor_cpu.ir.graph import Graph, Node
from tensor_cpu.ir.ops import OpType

from .ir import (
    Assign,
    BinaryOp,
    Block,
    Define,
    FunctionDecl,
    GraphLowering,
    Identifier,
    Include,
    Index,
    Literal,
    Program,
    RawCode,
    Return,
    StructDecl,
    TIRToCppConverter,
    VarDecl,
    generate_cpp,
)


@dataclass(slots=True)
class GeneratedKernel:
    source: str
    entry: str
    output_sym_shape: tuple[str, ...] = ()
    input_ranks: tuple[int, ...] = ()
    workspace_slots: tuple[tuple[str, tuple[str, ...]], ...] = ()


class LayeredCodegen:
    """Generate C++ kernels using the layered IR architecture."""

    def __init__(
        self, graph: Graph, use_hpc_template: bool = False, enable_memory_planner: bool = True
    ) -> None:
        self.graph = graph
        self.use_hpc_template = use_hpc_template
        self.enable_memory_planner = enable_memory_planner
        self._sym: dict[int, tuple[str, ...]] = {}
        self._sym_str: dict[int, tuple[str, ...]] = {}
        self._compute_dtype = None

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
    def _sym_strides_from(dims: tuple[str, ...]) -> tuple[str, ...]:
        if not dims:
            return ()
        strides = ["1"] * len(dims)
        for i in range(len(dims) - 2, -1, -1):
            if strides[i + 1] == "1":
                strides[i] = dims[i + 1]
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

    def generate(self) -> GeneratedKernel:
        ordered = self.graph.topological_sort()
        inputs = [node for node in ordered if node.op_type == OpType.INPUT]
        output_node = self._resolve_output(ordered)

        self._compute_dtype = output_node.dtype or "float32"
        self._build_symbolic_shapes(ordered, inputs)

        lowering = GraphLowering(
            graph=self.graph,
            sym_shapes=self._sym,
            sym_strides=self._sym_str,
        )
        ir_module = lowering.lower()

        converter = TIRToCppConverter()
        for _func_name, func in ir_module.functions.items():
            func.accept(converter)

        program = self._build_full_program(ordered, inputs, output_node)
        source = generate_cpp(program)

        output_sym = self._sym.get(output_node.id, tuple(str(d) for d in output_node.shape))
        input_ranks = tuple(node.rank for node in inputs)

        workspace_sym = self._compute_workspace_slots(ordered, inputs)

        return GeneratedKernel(
            source=source,
            entry="run_kernel",
            output_sym_shape=output_sym,
            input_ranks=input_ranks,
            workspace_slots=workspace_sym,
        )

    def _build_full_program(
        self, ordered: list[Node], inputs: list[Node], output_node: Node
    ) -> Program:
        max_rank = 8

        includes = [
            Include("algorithm"),
            Include("cfloat"),
            Include("cstddef"),
            Include("cmath"),
            Include("cstdint"),
            Include("immintrin.h"),
        ]

        defines = [
            Define("_WIN32", None),
            RawCode("#define EXPORT __declspec(dllexport)\n#else\n#define EXPORT\n#endif"),
        ]

        struct_decl = StructDecl(
            name="TensorDesc",
            fields=[
                ("void*", "data"),
                ("long long", "numel"),
                ("long long", "rank"),
                ("long long", f"shape[{max_rank}]"),
                ("long long", f"strides[{max_rank}]"),
            ],
        )

        body_stmts = self._build_kernel_body(ordered, inputs, output_node)

        func = FunctionDecl(
            return_type="int",
            name="run_kernel",
            params=[
                ("const TensorDesc*", "inputs"),
                ("long long", "num_inputs"),
                ("TensorDesc*", "out_desc"),
                ("void*", "workspace"),
            ],
            body=Block(stmts=body_stmts),
            export=True,
        )

        return Program(
            includes=includes,
            defines=defines,
            decls=[struct_decl, func],
        )

    def _build_kernel_body(
        self, ordered: list[Node], inputs: list[Node], output_node: Node
    ) -> list:

        ctype = "double" if self._compute_dtype == "float64" else "float"
        stmts = []

        input_count_check = (
            f"if (num_inputs != {len(inputs)}LL) return {int(AbiStatus.INPUT_COUNT_MISMATCH)};"
        )
        stmts.append(RawCode(input_count_check))

        for i, node in enumerate(inputs):
            guard = f"if (inputs[{i}].data == nullptr) return {int(AbiStatus.INPUT_DATA_NULL_BASE) + i};"
            stmts.append(RawCode(guard))
            guard = f"if (inputs[{i}].rank != {node.rank}LL) return {int(AbiStatus.INPUT_RANK_MISMATCH_BASE) + i};"
            stmts.append(RawCode(guard))

        for i, _node in enumerate(inputs):
            stmts.append(
                VarDecl(
                    var_type=f"{ctype}*",
                    name=f"arg{i}",
                    init=RawCode(f"static_cast<{ctype}*>(inputs[{i}].data)"),
                )
            )

        for idx, node in enumerate(inputs):
            for d in range(node.rank):
                stmts.append(RawCode(f"const long long in{idx}_d{d} = inputs[{idx}].shape[{d}];"))

        output_guards = [
            f"if (out_desc == nullptr) return {int(AbiStatus.OUT_DESC_NULL)};",
            f"if (out_desc->data == nullptr) return {int(AbiStatus.OUT_DATA_NULL)};",
            f"if (out_desc->rank != {output_node.rank}LL) return {int(AbiStatus.OUT_RANK_MISMATCH)};",
        ]
        for guard in output_guards:
            stmts.append(RawCode(guard))

        stmts.append(
            VarDecl(
                var_type=f"{ctype}*",
                name="out_ptr",
                init=RawCode(f"static_cast<{ctype}*>(out_desc->data)"),
            )
        )

        names: dict[int, str] = {}
        for idx, node in enumerate(inputs):
            names[node.id] = f"arg{idx}"

        use_count = self._compute_use_count(ordered)
        temp_owner: dict[int, str] = {}
        free_slots: dict[tuple[str, ...], list[str]] = {}
        declared_slots: set = set()
        workspace_slots: list[tuple[str, str]] = []
        slot_counter = 0

        for node in ordered:
            if node.op_type == OpType.INPUT:
                continue

            if node.op_type == OpType.CONST:
                const_name = f"const_{node.id}"
                names[node.id] = const_name
                value = float(node.attrs.get("value", 0.0))
                stmts.append(VarDecl(var_type=ctype, name=f"{const_name}_val", init=Literal(value)))
                stmts.append(RawCode(f"{ctype}* {const_name} = &{const_name}_val;"))
                continue

            if node.id == output_node.id:
                names[node.id] = "out_ptr"
            else:
                if self.enable_memory_planner:
                    sym_key = self._sym.get(node.id, ())
                    numel = self._numel_expr(node)
                    bucket = free_slots.setdefault(sym_key, [])
                    if bucket:
                        slot_name = bucket.pop()
                    else:
                        slot_name = f"buf_s{slot_counter}"
                        slot_counter += 1
                        if slot_name not in declared_slots:
                            workspace_slots.append((slot_name, numel))
                            declared_slots.add(slot_name)
                    names[node.id] = slot_name
                    temp_owner[node.id] = slot_name
                else:
                    slot_name = f"buf_{node.id}"
                    names[node.id] = slot_name
                    numel = self._numel_expr(node)
                    workspace_slots.append((slot_name, numel))

            node_stmts = self._emit_node_stmts(node, names)
            stmts.extend(node_stmts)

            if self.enable_memory_planner:
                for in_id in node.inputs:
                    if in_id in use_count:
                        use_count[in_id] -= 1
                        if use_count[in_id] == 0 and in_id in temp_owner:
                            slot_name = temp_owner[in_id]
                            free_slots.setdefault(self._sym.get(in_id, ()), []).append(slot_name)

        output_numel = self._numel_expr(output_node)
        output_name = names.get(output_node.id, "out_ptr")
        stmts.append(
            RawCode(
                f"if ({output_name} != out_ptr) {{\n"
                f"    for (long long i = 0; i < {output_numel}; ++i) out_ptr[i] = {output_name}[i];\n"
                f"}}"
            )
        )
        stmts.append(Return(Literal(0)))

        return stmts

    def _emit_node_stmts(self, node: Node, names: dict[int, str]) -> list:
        from .ir import Call, ForLoop, TernaryOp

        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)
        stmts = []

        if node.op_type in (OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV):
            op_map = {OpType.ADD: "+", OpType.SUB: "-", OpType.MUL: "*", OpType.DIV: "/"}
            op = op_map[node.op_type]
            total = self._numel_expr(node)
            lhs = names[node.inputs[0]]
            rhs = names[node.inputs[1]]
            dst = names[node.id]

            lhs_node = self.graph.get_node(node.inputs[0])
            rhs_node = self.graph.get_node(node.inputs[1])
            li = self._broadcast_index_expr(dst_node=node, src_node=lhs_node, linear_index_var="i")
            ri = self._broadcast_index_expr(dst_node=node, src_node=rhs_node, linear_index_var="i")

            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                BinaryOp(
                                    Index(
                                        Identifier(lhs),
                                        Literal(li) if li.isdigit() else Identifier(li),
                                    ),
                                    op,
                                    Index(
                                        Identifier(rhs),
                                        Literal(ri) if ri.isdigit() else Identifier(ri),
                                    ),
                                ),
                            )
                        ]
                    ),
                )
            )

        elif node.op_type == OpType.EQ:
            total = self._numel_expr(node)
            lhs = names[node.inputs[0]]
            rhs = names[node.inputs[1]]
            dst = names[node.id]
            one = "1.0" if node.dtype == "float64" else "1.0f"
            eps = "1e-12" if node.dtype == "float64" else "1e-6f"

            lhs_node = self.graph.get_node(node.inputs[0])
            rhs_node = self.graph.get_node(node.inputs[1])
            li = self._broadcast_index_expr(dst_node=node, src_node=lhs_node, linear_index_var="i")
            ri = self._broadcast_index_expr(dst_node=node, src_node=rhs_node, linear_index_var="i")

            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                TernaryOp(
                                    BinaryOp(
                                        Call(
                                            "std::fabs",
                                            [
                                                BinaryOp(
                                                    Index(
                                                        Identifier(lhs),
                                                        (
                                                            Literal(li)
                                                            if li.isdigit()
                                                            else Identifier(li)
                                                        ),
                                                    ),
                                                    "-",
                                                    Index(
                                                        Identifier(rhs),
                                                        (
                                                            Literal(ri)
                                                            if ri.isdigit()
                                                            else Identifier(ri)
                                                        ),
                                                    ),
                                                )
                                            ],
                                        ),
                                        "<=",
                                        Literal(eps),
                                    ),
                                    Literal(one),
                                    Literal(zero),
                                ),
                            )
                        ]
                    ),
                )
            )

        elif node.op_type == OpType.RELU:
            total = self._numel_expr(node)
            src = names[node.inputs[0]]
            dst = names[node.id]

            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            VarDecl(
                                var_type=ctype,
                                name="v",
                                init=Index(Identifier(src), Identifier("i")),
                            ),
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                TernaryOp(
                                    BinaryOp(Identifier("v"), ">", Literal(zero)),
                                    Identifier("v"),
                                    Literal(zero),
                                ),
                            ),
                        ]
                    ),
                )
            )

        elif node.op_type == OpType.EXP:
            total = self._numel_expr(node)
            src = names[node.inputs[0]]
            dst = names[node.id]

            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                Call("std::exp", [Index(Identifier(src), Identifier("i"))]),
                            )
                        ]
                    ),
                )
            )

        elif node.op_type == OpType.LOG:
            total = self._numel_expr(node)
            src = names[node.inputs[0]]
            dst = names[node.id]

            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                Call("std::log", [Index(Identifier(src), Identifier("i"))]),
                            )
                        ]
                    ),
                )
            )

        elif node.op_type == OpType.SIGMOID:
            total = self._numel_expr(node)
            src = names[node.inputs[0]]
            dst = names[node.id]
            one = "1.0" if node.dtype == "float64" else "1.0f"

            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            VarDecl(
                                var_type=ctype,
                                name="v",
                                init=Index(Identifier(src), Identifier("i")),
                            ),
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                BinaryOp(
                                    Literal(one),
                                    "/",
                                    BinaryOp(
                                        Literal(one),
                                        "+",
                                        Call("std::exp", [UnaryOp("-", Identifier("v"))]),
                                    ),
                                ),
                            ),
                        ]
                    ),
                )
            )

        elif node.op_type == OpType.MATMUL:
            stmts.extend(self._emit_matmul_stmts(node, names))

        elif node.op_type in (OpType.FUSED_MATMUL_BIAS, OpType.FUSED_MATMUL_BIAS_RELU):
            stmts.extend(self._emit_fused_matmul_stmts(node, names))

        elif node.op_type == OpType.TRANSPOSE:
            stmts.extend(self._emit_transpose_stmts(node, names))

        elif node.op_type in (OpType.SUM, OpType.MEAN, OpType.MAX):
            stmts.extend(self._emit_reduce_stmts(node, names))

        elif node.op_type == OpType.BROADCAST_TO:
            total = self._numel_expr(node)
            src = names[node.inputs[0]]
            dst = names[node.id]
            src_node = self.graph.get_node(node.inputs[0])
            si = self._broadcast_index_expr(dst_node=node, src_node=src_node, linear_index_var="i")

            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                Index(
                                    Identifier(src), Literal(si) if si.isdigit() else Identifier(si)
                                ),
                            )
                        ]
                    ),
                )
            )

        return stmts

    def _emit_matmul_stmts(self, node: Node, names: dict[int, str]) -> list:
        from .ir import ForLoop

        a_node = self.graph.get_node(node.inputs[0])
        b_node = self.graph.get_node(node.inputs[1])
        a = names[a_node.id]
        b = names[b_node.id]
        c = names[node.id]

        a_sym = self._sym.get(a_node.id, ())
        b_sym = self._sym.get(b_node.id, ())

        M = a_sym[-2] if len(a_sym) >= 2 else "1"
        K = a_sym[-1] if len(a_sym) >= 1 else "1"
        N = b_sym[-1] if len(b_sym) >= 1 else "1"

        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)

        inner_body = Block(
            stmts=[
                VarDecl(var_type=ctype, name="acc", init=Literal(zero)),
                ForLoop(
                    init=VarDecl(var_type="long long", name="kk", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("kk"), "<", Literal(K) if K.isdigit() else Identifier(K)
                    ),
                    update=Assign(Identifier("kk"), BinaryOp(Identifier("kk"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Identifier("acc"),
                                BinaryOp(
                                    Identifier("acc"),
                                    "+",
                                    BinaryOp(
                                        Index(
                                            Identifier(a),
                                            BinaryOp(
                                                BinaryOp(
                                                    Identifier("i"),
                                                    "*",
                                                    Literal(K) if K.isdigit() else Identifier(K),
                                                ),
                                                "+",
                                                Identifier("kk"),
                                            ),
                                        ),
                                        "*",
                                        Index(
                                            Identifier(b),
                                            BinaryOp(
                                                BinaryOp(
                                                    Identifier("kk"),
                                                    "*",
                                                    Literal(N) if N.isdigit() else Identifier(N),
                                                ),
                                                "+",
                                                Identifier("j"),
                                            ),
                                        ),
                                    ),
                                ),
                            )
                        ]
                    ),
                ),
                Assign(
                    Index(
                        Identifier(c),
                        BinaryOp(
                            BinaryOp(
                                Identifier("i"), "*", Literal(N) if N.isdigit() else Identifier(N)
                            ),
                            "+",
                            Identifier("j"),
                        ),
                    ),
                    Identifier("acc"),
                ),
            ]
        )

        j_loop = ForLoop(
            init=VarDecl(var_type="long long", name="j", init=Literal(0)),
            cond=BinaryOp(Identifier("j"), "<", Literal(N) if N.isdigit() else Identifier(N)),
            update=Assign(Identifier("j"), BinaryOp(Identifier("j"), "+", Literal(1))),
            body=inner_body,
        )

        i_loop = ForLoop(
            init=VarDecl(var_type="long long", name="i", init=Literal(0)),
            cond=BinaryOp(Identifier("i"), "<", Literal(M) if M.isdigit() else Identifier(M)),
            update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
            body=Block(stmts=[j_loop]),
        )

        return [i_loop]

    def _emit_fused_matmul_stmts(self, node: Node, names: dict[int, str]) -> list:
        from .ir import ForLoop

        a_node = self.graph.get_node(node.inputs[0])
        b_node = self.graph.get_node(node.inputs[1])
        bias_node = self.graph.get_node(node.inputs[2])

        a = names[a_node.id]
        b = names[b_node.id]
        bias = names[bias_node.id]
        out = names[node.id]

        a_sym = self._sym.get(a_node.id, ())
        b_sym = self._sym.get(b_node.id, ())

        m = a_sym[-2] if len(a_sym) >= 2 else "1"
        k = a_sym[-1] if len(a_sym) >= 1 else "1"
        n = b_sym[-1] if len(b_sym) >= 1 else "1"

        with_relu = node.op_type == OpType.FUSED_MATMUL_BIAS_RELU
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)

        inner_stmts = [
            VarDecl(var_type=ctype, name="acc", init=Literal(zero)),
            ForLoop(
                init=VarDecl(var_type="long long", name="kk", init=Literal(0)),
                cond=BinaryOp(Identifier("kk"), "<", Literal(k) if k.isdigit() else Identifier(k)),
                update=Assign(Identifier("kk"), BinaryOp(Identifier("kk"), "+", Literal(1))),
                body=Block(
                    stmts=[
                        Assign(
                            Identifier("acc"),
                            BinaryOp(
                                Identifier("acc"),
                                "+",
                                BinaryOp(
                                    Index(
                                        Identifier(a),
                                        BinaryOp(
                                            BinaryOp(
                                                Identifier("i"),
                                                "*",
                                                Literal(k) if k.isdigit() else Identifier(k),
                                            ),
                                            "+",
                                            Identifier("kk"),
                                        ),
                                    ),
                                    "*",
                                    Index(
                                        Identifier(b),
                                        BinaryOp(
                                            BinaryOp(
                                                Identifier("kk"),
                                                "*",
                                                Literal(n) if n.isdigit() else Identifier(n),
                                            ),
                                            "+",
                                            Identifier("j"),
                                        ),
                                    ),
                                ),
                            ),
                        )
                    ]
                ),
            ),
            VarDecl(
                var_type=ctype,
                name="v",
                init=BinaryOp(Identifier("acc"), "+", Index(Identifier(bias), Identifier("j"))),
            ),
        ]

        if with_relu:
            inner_stmts.append(
                Assign(
                    Identifier("v"),
                    TernaryOp(
                        BinaryOp(Identifier("v"), ">", Literal(zero)),
                        Identifier("v"),
                        Literal(zero),
                    ),
                )
            )

        inner_stmts.append(
            Assign(
                Index(
                    Identifier(out),
                    BinaryOp(
                        BinaryOp(
                            Identifier("i"), "*", Literal(n) if n.isdigit() else Identifier(n)
                        ),
                        "+",
                        Identifier("j"),
                    ),
                ),
                Identifier("v"),
            )
        )

        j_loop = ForLoop(
            init=VarDecl(var_type="long long", name="j", init=Literal(0)),
            cond=BinaryOp(Identifier("j"), "<", Literal(n) if n.isdigit() else Identifier(n)),
            update=Assign(Identifier("j"), BinaryOp(Identifier("j"), "+", Literal(1))),
            body=Block(stmts=inner_stmts),
        )

        i_loop = ForLoop(
            init=VarDecl(var_type="long long", name="i", init=Literal(0)),
            cond=BinaryOp(Identifier("i"), "<", Literal(m) if m.isdigit() else Identifier(m)),
            update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
            body=Block(stmts=[j_loop]),
        )

        return [i_loop]

    def _emit_transpose_stmts(self, node: Node, names: dict[int, str]) -> list:
        from .ir import ForLoop

        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        rank = len(self._sym.get(src_node.id, ()))
        total = self._numel_expr(node)

        if rank <= 1:
            return [
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Index(Identifier(dst), Identifier("i")),
                                Index(Identifier(src), Identifier("i")),
                            )
                        ]
                    ),
                )
            ]

        dst_strides = self._sym_str.get(node.id, ())
        src_strides = self._sym_str.get(src_node.id, ())

        body_stmts = [
            VarDecl(var_type="long long", name="src_idx", init=Literal(0)),
            VarDecl(var_type="long long", name="rem", init=Identifier("i")),
        ]

        for d in range(rank):
            src_dim_idx = rank - 1 - d
            dst_stride = dst_strides[d]
            src_stride = src_strides[src_dim_idx]

            body_stmts.append(
                RawCode(f"long long c{d} = rem / ({dst_stride}); rem %= ({dst_stride});")
            )
            body_stmts.append(RawCode(f"src_idx += c{d} * ({src_stride});"))

        body_stmts.append(
            Assign(
                Index(Identifier(dst), Identifier("i")),
                Index(Identifier(src), Identifier("src_idx")),
            )
        )

        return [
            ForLoop(
                init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                cond=BinaryOp(
                    Identifier("i"), "<", Literal(total) if total.isdigit() else Identifier(total)
                ),
                update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                body=Block(stmts=body_stmts),
            )
        ]

    def _emit_reduce_stmts(self, node: Node, names: dict[int, str]) -> list:
        from .ir import ForLoop

        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        total = self._numel_expr(src_node)
        out_total = self._numel_expr(node)
        axes = tuple(int(a) for a in node.attrs.get("axis", ()))
        dst_idx = self._reduce_dst_index_expr(
            src_node=src_node, dst_node=node, axes=axes, idx_var="i"
        )

        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)

        init_val = zero
        if node.op_type == OpType.MAX:
            init_val = self._cpp_neg_inf(node)

        stmts = [
            ForLoop(
                init=VarDecl(var_type="long long", name="o", init=Literal(0)),
                cond=BinaryOp(
                    Identifier("o"),
                    "<",
                    Literal(out_total) if out_total.isdigit() else Identifier(out_total),
                ),
                update=Assign(Identifier("o"), BinaryOp(Identifier("o"), "+", Literal(1))),
                body=Block(
                    stmts=[Assign(Index(Identifier(dst), Identifier("o")), Literal(init_val))]
                ),
            ),
        ]

        if node.op_type == OpType.MAX:
            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            VarDecl(
                                var_type="int",
                                name="di",
                                init=Literal(dst_idx) if dst_idx.isdigit() else Identifier(dst_idx),
                            ),
                            RawCode(f"if ({src}[i] > {dst}[di]) {dst}[di] = {src}[i];"),
                        ]
                    ),
                )
            )
        else:
            stmts.append(
                ForLoop(
                    init=VarDecl(var_type="long long", name="i", init=Literal(0)),
                    cond=BinaryOp(
                        Identifier("i"),
                        "<",
                        Literal(total) if total.isdigit() else Identifier(total),
                    ),
                    update=Assign(Identifier("i"), BinaryOp(Identifier("i"), "+", Literal(1))),
                    body=Block(
                        stmts=[
                            Assign(
                                Index(
                                    Identifier(dst),
                                    Literal(dst_idx) if dst_idx.isdigit() else Identifier(dst_idx),
                                ),
                                BinaryOp(
                                    Index(
                                        Identifier(dst),
                                        (
                                            Literal(dst_idx)
                                            if dst_idx.isdigit()
                                            else Identifier(dst_idx)
                                        ),
                                    ),
                                    "+",
                                    Index(Identifier(src), Identifier("i")),
                                ),
                            )
                        ]
                    ),
                )
            )

        if node.op_type == OpType.MEAN:
            src_sym = self._sym.get(src_node.id, ())
            reduce_parts = [src_sym[a] for a in axes if a < len(src_sym)]
            if reduce_parts:
                if len(reduce_parts) == 1:
                    reduce_size_expr = reduce_parts[0]
                else:
                    reduce_size_expr = " * ".join(f"({d})" for d in reduce_parts)

                stmts.append(
                    ForLoop(
                        init=VarDecl(var_type="long long", name="o", init=Literal(0)),
                        cond=BinaryOp(
                            Identifier("o"),
                            "<",
                            Literal(out_total) if out_total.isdigit() else Identifier(out_total),
                        ),
                        update=Assign(Identifier("o"), BinaryOp(Identifier("o"), "+", Literal(1))),
                        body=Block(
                            stmts=[
                                Assign(
                                    Index(Identifier(dst), Identifier("o")),
                                    BinaryOp(
                                        Index(Identifier(dst), Identifier("o")),
                                        "/",
                                        Cast(
                                            ctype,
                                            (
                                                Literal(reduce_size_expr)
                                                if reduce_size_expr.isdigit()
                                                else Identifier(reduce_size_expr)
                                            ),
                                        ),
                                    ),
                                )
                            ]
                        ),
                    )
                )

        return stmts

    def _compute_use_count(self, ordered: list[Node]) -> dict[int, int]:
        use_count: dict[int, int] = {n.id: 0 for n in ordered}
        for node in ordered:
            for in_id in node.inputs:
                if in_id in use_count:
                    use_count[in_id] += 1
        return use_count

    def _resolve_output(self, ordered: list[Node]) -> Node:
        if self.graph.output_ids:
            return self.graph.get_node(self.graph.output_ids[-1])
        if not ordered:
            raise ValueError("Graph is empty.")
        return ordered[-1]

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

        terms: list[str] = []
        dst_axis = 0
        for src_axis, src_d in enumerate(src_sym):
            if src_axis in axes_set:
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

    def _compute_workspace_slots(
        self, ordered: list[Node], inputs: list[Node]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        names: dict[int, str] = {}
        for idx, node in enumerate(inputs):
            names[node.id] = f"arg{idx}"

        use_count = self._compute_use_count(ordered)
        temp_owner: dict[int, str] = {}
        free_slots: dict[tuple[str, ...], list[str]] = {}
        declared_slots: set = set()
        workspace_sym: list[tuple[str, tuple[str, ...]]] = []
        slot_counter = 0

        for node in ordered:
            if node.op_type == OpType.INPUT:
                continue

            if node.op_type == OpType.CONST:
                names[node.id] = f"const_{node.id}"
                continue

            if self.enable_memory_planner:
                sym_key = self._sym.get(node.id, ())
                bucket = free_slots.setdefault(sym_key, [])
                if bucket:
                    slot_name = bucket.pop()
                else:
                    slot_name = f"buf_s{slot_counter}"
                    slot_counter += 1
                    if slot_name not in declared_slots:
                        workspace_sym.append((slot_name, sym_key))
                        declared_slots.add(slot_name)
                names[node.id] = slot_name
                temp_owner[node.id] = slot_name
            else:
                slot_name = f"buf_{node.id}"
                names[node.id] = slot_name
                sym_key = self._sym.get(node.id, ())
                workspace_sym.append((slot_name, sym_key))

            if self.enable_memory_planner:
                for in_id in node.inputs:
                    if in_id in use_count:
                        use_count[in_id] -= 1
                        if use_count[in_id] == 0 and in_id in temp_owner:
                            slot_name = temp_owner[in_id]
                            free_slots.setdefault(self._sym.get(in_id, ()), []).append(slot_name)

        return tuple(workspace_sym)

"""C++ code generation for traced graph."""
from __future__ import annotations
"""只编译一次，能处理任意输入的shape，shape使用符号"""
"""目前只支持单输出节点"""

from dataclasses import dataclass
from typing import Dict, List, Union

from .abi import AbiStatus
from .graph import Graph, Node
from .ops import OpType


@dataclass(slots=True)
class GeneratedKernel:
    source: str #c++ kernel源码
    entry: str #kernel入口函数名
    output_sym_shape: tuple[str, ...] = ()  # symbolic output dim exprs 输出节点的符号shape表达式
    input_ranks: tuple[int, ...] = ()       # required rank for each input 输入节点的rank维度
    workspace_slots: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (name, sym_dims) for arena，数组实现


@dataclass(slots=True)
class CppLine:
    text: str


@dataclass(slots=True)
class CppFor:
    init: str
    cond: str
    inc: str
    body: List["CppStmt"]


CppStmt = Union[CppLine, CppFor]


def render_cpp_stmts(stmts: List[CppStmt], indent: int = 0) -> List[str]:
    lines: List[str] = []
    pad = " " * indent
    for stmt in stmts:
        if isinstance(stmt, CppLine):
            lines.append(f"{pad}{stmt.text}")
            continue
        if isinstance(stmt, CppFor):
            lines.append(f"{pad}for ({stmt.init}; {stmt.cond}; {stmt.inc}) {{")
            lines.extend(render_cpp_stmts(stmt.body, indent=indent + 4))
            lines.append(f"{pad}}}")
            continue
        raise TypeError(f"Unsupported CppStmt: {type(stmt)!r}")
    return lines


class CppCodegen:
    """Generate C++ kernels from Graph IR."""

    def __init__(self, graph: Graph, use_hpc_template: bool = False, enable_memory_planner: bool = True) -> None:
        self.graph = graph
        self.use_hpc_template = use_hpc_template
        self.enable_memory_planner = enable_memory_planner
        self._sym: Dict[int, tuple[str, ...]] = {}      # node_id -> symbolic dim exprs
        self._sym_str: Dict[int, tuple[str, ...]] = {}   # node_id -> symbolic strides
        self._compute_dtype = None
    # ---- dtype helpers ----

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

    # ---- symbolic shape system ----

    @staticmethod
    def _sym_strides_from(dims: tuple[str, ...]) -> tuple[str, ...]:
        """Compute contiguous strides from symbolic dim expressions. element stride"""
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

    def _build_symbolic_shapes(self, ordered: List[Node], inputs: List[Node]) -> None:
        """Build symbolic dimension expressions for all nodes."""
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

        if node.op_type in (OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.RELU_GRAD):
            lhs = self._sym.get(node.inputs[0], ())
            rhs = self._sym.get(node.inputs[1], ())
            dims = self._sym_broadcast(lhs, rhs)
            self._sym[node.id] = dims
            self._sym_str[node.id] = self._sym_strides_from(dims)
            return

        if node.op_type == OpType.EQ:
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
            axes_set = set(int(a) for a in node.attrs.get("axis", ()))
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
            #  Try to match target shape against known symbolic dims
            src_node = self.graph.get_node(node.inputs[0])
            src_sym = self._sym.get(node.inputs[0], ())
            target = tuple(int(d) for d in node.attrs.get("target_shape", node.shape))

            # Expand-dims style reshape: source dims are embedded in target
            # with singleton axes inserted (e.g. (B,) -> (B, 1)).
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

            # Look for a node whose literal shape matches target to reuse its symbolic shape
            for nid, sym in self._sym.items():
                other = self.graph.get_node(nid)
                if other.shape == target:
                    self._sym[node.id] = sym
                    self._sym_str[node.id] = self._sym_strides_from(sym)
                    return
            # Fallback: use literal dims
            dims_lit = tuple(str(d) for d in target)
            self._sym[node.id] = dims_lit
            self._sym_str[node.id] = self._sym_strides_from(dims_lit)
            return

        # Fallback (fused ops, output, unknown)
        dims_lit = tuple(str(d) for d in node.shape)
        self._sym[node.id] = dims_lit
        self._sym_str[node.id] = self._sym_strides_from(dims_lit)

    def generate(self) -> GeneratedKernel:
        ordered = self.graph.topological_sort() #node拓扑排序
        inputs = [node for node in ordered if node.op_type == OpType.INPUT]
        output_node = self._resolve_output(ordered)

        # Determine the compute dtype for this kernel
        self._compute_dtype = output_node.dtype or "float32"

        # Build symbolic shape expressions 构建所有节点的符号shape和stride表达式。
        self._build_symbolic_shapes(ordered, inputs)

        #print(self._sym,self._sym_str)

        declarations: List[str] = [] #声明语句列表
        body: List[str] = []
        names: Dict[int, str] = {}   #node_id -> C++ variable name
        declared_slots: set[str] = set() #记录已声明的内存缓冲区名，防止重复，存放中间节点的计算结果
        workspace_slots: List[tuple[str, str]] = []        # (slot_name, numel_expr)
        workspace_sym: List[tuple[str, tuple[str, ...]]] = []  # (slot_name, sym_dims)
        ctype = "double" if self._compute_dtype == "float64" else "float"

        for idx, node in enumerate(inputs):
            names[node.id] = f"arg{idx}"#输入节点直接命名为arg0, arg1, ...

        use_count = self._compute_use_count(ordered) #统计每个节点被引用次数
        temp_owner: Dict[int, str] = {}#记录每个中间节点占用的缓冲区名
        free_slots: Dict[int, List[str]] = {} #记录可复用的缓冲区（按符号shape分组）。

        slot_counter = 0

        for node in ordered: #遍历所有节点，分配变量、内存、生成代码
            if node.op_type == OpType.INPUT:
                continue

            if node.op_type == OpType.CONST:
                const_name = f"const_{node.id}"
                names[node.id] = const_name
                value = float(node.attrs.get("value", 0.0))
                declarations.append(f"{ctype} {const_name}_val = {value};")
                declarations.append(f"{ctype}* {const_name} = &{const_name}_val;")
                continue
            if node.id == output_node.id:
                names[node.id] = "out_ptr"
            else:
                if self.enable_memory_planner:#启用内存规划时，按符号shape分组复用缓冲区，否则每个节点独立分配
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
                            workspace_sym.append((slot_name, sym_key))
                            declared_slots.add(slot_name)
                    names[node.id] = slot_name
                    temp_owner[node.id] = slot_name
                else:
                    slot_name = f"buf_{node.id}"
                    names[node.id] = slot_name
                    numel = self._numel_expr(node)
                    sym_key = self._sym.get(node.id, ())
                    workspace_slots.append((slot_name, numel))
                    workspace_sym.append((slot_name, sym_key))

            # Emit node code
            emit = self._emit_node(node=node, names=names) #根据节点类型生成计算代码，返回字符串列表
            if emit:
                body.extend(emit)

            if self.enable_memory_planner: #某些输入节点引用计数归零时，释放其占用的缓冲区。
                for in_id in node.inputs:
                    if in_id in use_count:
                        use_count[in_id] -= 1
                        if use_count[in_id] == 0 and in_id in temp_owner:
                            slot_name = temp_owner[in_id]
                            src_node = self.graph.get_node(in_id)
                            free_slots.setdefault(self._sym.get(in_id, ()), []).append(slot_name)

        kernel = self._render_cpp(
            ordered=ordered,
            inputs=inputs,
            output_node=output_node,
            declarations=declarations,
            body=body,
            workspace_slots=workspace_slots,
        )#拼接生成kernel源码
        output_sym = self._sym.get(output_node.id, tuple(str(d) for d in output_node.shape))
        input_ranks = tuple(node.rank for node in inputs)
        return GeneratedKernel(
            source=kernel,
            entry="run_kernel",
            output_sym_shape=output_sym,
            input_ranks=input_ranks,
            workspace_slots=tuple(workspace_sym),
        )

    def _compute_use_count(self, ordered: List[Node]) -> Dict[int, int]:
        use_count: Dict[int, int] = {n.id: 0 for n in ordered}
        for node in ordered:
            for in_id in node.inputs:
                if in_id in use_count:
                    use_count[in_id] += 1
        return use_count

    def _resolve_output(self, ordered: List[Node]) -> Node: #获得输出节点
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

    def _emit_structured(self, stmts: List[CppStmt]) -> List[str]:
        return render_cpp_stmts(stmts)

    def _emit_node(self, node: Node, names: Dict[int, str]) -> List[str]:
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

    def _emit_sub(self, node: Node, names: Dict[int, str]) -> List[str]: #生成减法代码，其他二元元素级操作类似
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
                    init="int i = 0",
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
                    body=[CppLine(f"{dst}[i] = (std::fabs({lhs}[{li}] - {rhs}[{ri}]) <= {eps}) ? {one} : {zero};")],
                )
            ]
        )
    
    #TODO:
    def _broadcast_index_expr(self, *, dst_node: Node, src_node: Node, linear_index_var: str) -> str:
        """Generate source flat-index expression for NumPy-style broadcast (symbolic)."""
        #负责生成C++代码中“如何根据输出的线性下标i，计算输入张量的实际访问下标”，即广播下标映射公式
        src_sym = self._sym.get(src_node.id, ())
        dst_sym = self._sym.get(dst_node.id, ())

        if src_sym == dst_sym:
            return linear_index_var

        if not src_sym or all(d == "1" for d in src_sym):
            return "0"

        # Expand-dims/reshape-like case: source dims appear in order among
        # destination dims after dropping singleton axes, e.g. (B,) -> (B, 1).
        dst_non1 = tuple(d for d in dst_sym if d != "1")
        if len(dst_sym) >= len(src_sym) and dst_non1 == src_sym:
            dst_strides = self._sym_strides_from(dst_sym)
            src_strides = self._sym_strides_from(src_sym)
            dst_axes = [i for i, d in enumerate(dst_sym) if d != "1"]
            terms: List[str] = []
            for src_axis, dst_axis in enumerate(dst_axes):
                coord_expr = f"(({linear_index_var} / ({dst_strides[dst_axis]})) % ({dst_sym[dst_axis]}))"
                src_stride = src_strides[src_axis]
                if src_stride == "1":
                    terms.append(coord_expr)
                else:
                    terms.append(f"({coord_expr} * ({src_stride}))")
            if terms:
                return " + ".join(terms)
            return "0"

        # Fast path for bias: src=(N,), dst=(..., N) where N is identical expression
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

        terms: List[str] = []
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

    def _emit_relu(self, node: Node, names: Dict[int, str]) -> List[str]:
        src = names[node.inputs[0]]
        dst = names[node.id]
        total = self._numel_expr(node)
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)
        return self._emit_structured(
            [
                CppFor(
                    init="int i = 0",
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
        return self._emit_structured(
            [
                CppFor(
                    init="int i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[i] = {src}[{si}] > {self._cpp_zero(node)} ? {gout}[{gi}] : {self._cpp_zero(node)};")],
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
                    init="int i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[i] = {src}[{si}];")],
                )
            ]
        )

    def _emit_exp(self, node: Node, names: Dict[int, str]) -> List[str]:
        src = names[node.inputs[0]]
        dst = names[node.id]
        total = self._numel_expr(node)
        return self._emit_structured(
            [
                CppFor(
                    init="int i = 0",
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
                    init="int i = 0",
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
                    init="int i = 0",
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
            return self._emit_structured([
                CppFor(init="long long i = 0", cond=f"i < {total}", inc="++i",
                       body=[CppLine(f"{dst}[i] = {src}[i];")])
            ])

        dst_strides = self._sym_str.get(node.id, ())
        src_strides = self._sym_str.get(src_node.id, ())

        body: List[CppStmt] = [CppLine("long long src_idx = 0;"), CppLine("long long rem = i;")]
        for d in range(rank):
            src_dim_idx = rank - 1 - d  # dst dim d maps to src dim (rank-1-d)
            body.append(CppLine(f"long long c{d} = rem / ({dst_strides[d]}); rem %= ({dst_strides[d]});"))
            body.append(CppLine(f"src_idx += c{d} * ({src_strides[src_dim_idx]});"))
        body.append(CppLine(f"{dst}[i] = {src}[src_idx];"))

        return self._emit_structured([
            CppFor(init="long long i = 0", cond=f"i < {total}", inc="++i", body=body)
        ])

    def _emit_sum(self, node: Node, names: Dict[int, str]) -> List[str]:
        src_node = self.graph.get_node(node.inputs[0])
        src = names[src_node.id]
        dst = names[node.id]
        total = self._numel_expr(src_node)
        out_total = self._numel_expr(node)
        axes = tuple(int(a) for a in node.attrs.get("axis", ()))
        dst_idx = self._reduce_dst_index_expr(src_node=src_node, dst_node=node, axes=axes, idx_var="i")
        zero = self._cpp_zero(node)
        return self._emit_structured(
            [
                CppFor(
                    init="int o = 0",
                    cond=f"o < {out_total}",
                    inc="++o",
                    body=[CppLine(f"{dst}[o] = {zero};")],
                ),
                CppFor(
                    init="int i = 0",
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
        dst_idx = self._reduce_dst_index_expr(src_node=src_node, dst_node=node, axes=axes, idx_var="i")
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
                    init="int o = 0",
                    cond=f"o < {out_total}",
                    inc="++o",
                    body=[CppLine(f"{dst}[o] = {zero};")],
                ),
                CppFor(
                    init="int i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[CppLine(f"{dst}[{dst_idx}] += {src}[i];")],
                ),
                CppFor(
                    init="int o = 0",
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
        dst_idx = self._reduce_dst_index_expr(src_node=src_node, dst_node=node, axes=axes, idx_var="i")
        return self._emit_structured(
            [
                CppFor(
                    init="int o = 0",
                    cond=f"o < {out_total}",
                    inc="++o",
                    body=[CppLine(f"{dst}[o] = {self._cpp_neg_inf(node)};")],
                ),
                CppFor(
                    init="int i = 0",
                    cond=f"i < {total}",
                    inc="++i",
                    body=[
                        CppLine(f"const int di = {dst_idx};"),
                        CppLine(f"if ({src}[i] > {dst}[di]) {dst}[di] = {src}[i];"),
                    ],
                ),
            ]
        )

    def _reduce_dst_index_expr(self, *, src_node: Node, dst_node: Node, axes: tuple[int, ...], idx_var: str) -> str:
        dst_sym = self._sym.get(dst_node.id, ())
        if not dst_sym:
            return "0"

        axes_set = set(axes)
        src_sym = self._sym.get(src_node.id, ())
        src_strides_sym = self._sym_str.get(src_node.id, ())
        dst_strides_sym = self._sym_str.get(dst_node.id, ())

        keepdims = bool(dst_node.attrs.get("keepdims", False))
        terms: List[str] = []
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

    def _emit_matmul(self, node: Node, names: Dict[int, str]) -> List[str]:
        a_node = self.graph.get_node(node.inputs[0])
        b_node = self.graph.get_node(node.inputs[1])
        a = names[a_node.id]
        b = names[b_node.id]
        c = names[node.id]
        a_sym = self._sym.get(a_node.id, ())
        b_sym = self._sym.get(b_node.id, ())
        c_sym = self._sym.get(node.id, ())

        M = a_sym[-2]
        K = a_sym[-1]
        N = b_sym[-1]
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)

        # Pure 2D matmul
        if len(a_sym) == 2 and len(b_sym) == 2:
            if self.use_hpc_template:
                # HPC uses literal ints for AVX tiling
                return self._emit_hpc_matmul(a, b, c, int(a_node.shape[-2]), int(b_node.shape[-1]), int(a_node.shape[-1]))
            return self._emit_structured([
                CppFor(init="long long i = 0", cond=f"i < ({M})", inc="++i", body=[
                    CppFor(init="long long j = 0", cond=f"j < ({N})", inc="++j", body=[
                        CppLine(f"{ctype} acc = {zero};"),
                        CppFor(init="long long kk = 0", cond=f"kk < ({K})", inc="++kk",
                               body=[CppLine(f"acc += {a}[i * ({K}) + kk] * {b}[kk * ({N}) + j];")]),
                        CppLine(f"{c}[i * ({N}) + j] = acc;"),
                    ])
                ])
            ])

        # Batch matmul with symbolic shapes
        batch_c = c_sym[:-2]
        batch_a = a_sym[:-2]
        batch_b = b_sym[:-2]
        rank_bc = len(batch_c)

        batch_size = " * ".join(f"({d})" for d in batch_c) if batch_c else "1"

        padded_a = ("1",) * (rank_bc - len(batch_a)) + batch_a
        padded_b = ("1",) * (rank_bc - len(batch_b)) + batch_b

        c_bs = self._sym_strides_from(batch_c)
        a_bs = self._sym_strides_from(padded_a)
        b_bs = self._sym_strides_from(padded_b)
        mat_a = f"(({M}) * ({K}))"
        mat_b = f"(({K}) * ({N}))"
        mat_c = f"(({M}) * ({N}))"

        body: List[CppStmt] = [
            CppLine("long long a_off = 0;"),
            CppLine("long long b_off = 0;"),
            CppLine("long long rem = batch;"),
        ]
        for d in range(rank_bc):
            body.append(CppLine(f"long long bc{d} = rem / ({c_bs[d]}); rem %= ({c_bs[d]});"))
            if padded_a[d] != "1":
                body.append(CppLine(f"a_off += bc{d} * ({a_bs[d]}) * {mat_a};"))
            if padded_b[d] != "1":
                body.append(CppLine(f"b_off += bc{d} * ({b_bs[d]}) * {mat_b};"))
        body.append(CppLine(f"const long long c_off = batch * {mat_c};"))
        body.append(
            CppFor(init="long long i = 0", cond=f"i < ({M})", inc="++i", body=[
                CppFor(init="long long j = 0", cond=f"j < ({N})", inc="++j", body=[
                    CppLine(f"{ctype} acc = {zero};"),
                    CppFor(init="long long kk = 0", cond=f"kk < ({K})", inc="++kk",
                           body=[CppLine(f"acc += {a}[a_off + i * ({K}) + kk] * {b}[b_off + kk * ({N}) + j];")]),
                    CppLine(f"{c}[c_off + i * ({N}) + j] = acc;"),
                ])
            ])
        )
        return self._emit_structured([
            CppFor(init="long long batch = 0", cond=f"batch < ({batch_size})", inc="++batch", body=body)
        ])

    def _emit_fused_matmul(self, node: Node, names: Dict[int, str]) -> List[str]:
        a_node = self.graph.get_node(node.inputs[0])
        b_node = self.graph.get_node(node.inputs[1])
        bias_node = self.graph.get_node(node.inputs[2])

        a_sym = self._sym.get(a_node.id, ())
        b_sym = self._sym.get(b_node.id, ())
        if len(a_sym) != 2 or len(b_sym) != 2:
            raise ValueError(f"Fused matmul requires 2D symbolic inputs, got {a_sym} and {b_sym}")

        a = names[a_node.id]
        b = names[b_node.id]
        bias = names[bias_node.id]
        out = names[node.id]

        m = a_sym[-2]
        k = a_sym[-1]
        n = b_sym[-1]
        with_relu = node.op_type == OpType.FUSED_MATMUL_BIAS_RELU
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)

        inner_body: List[CppStmt] = [
            CppLine(f"{ctype} acc = {zero};"),
            CppFor(
                init="long long kk = 0",
                cond=f"kk < ({k})",
                inc="++kk",
                body=[CppLine(f"acc += {a}[i * ({k}) + kk] * {b}[kk * ({n}) + j];")],
            ),
            CppLine(f"{ctype} v = acc + {bias}[j];"),
        ]
        if with_relu:
            czero = self._cpp_zero(node)
            inner_body.append(CppLine(f"v = v > {czero} ? v : {czero};"))
        inner_body.append(CppLine(f"{out}[i * ({n}) + j] = v;"))

        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < ({m})",
                    inc="++i",
                    body=[
                        CppFor(
                            init="long long j = 0",
                            cond=f"j < ({n})",
                            inc="++j",
                            body=inner_body,
                        )
                    ],
                )
            ]
        )

    def _emit_hpc_matmul(self, a: str, b: str, c: str, m: int, n: int, k: int) -> List[str]:
        return [
            "{",
            "constexpr int BM = 64;",
            "constexpr int BN = 64;",
            "constexpr int BK = 64;",
            f"for (int i = 0; i < {m} * {n}; ++i) {c}[i] = 0.0f;",
            f"#pragma omp parallel for schedule(static)",
            f"for (int ii = 0; ii < {m}; ii += BM) {{",
            f"  for (int kk = 0; kk < {k}; kk += BK) {{",
            f"    for (int jj = 0; jj < {n}; jj += BN) {{",
            f"      const int i_max = (ii + BM < {m}) ? (ii + BM) : {m};",
            f"      const int k_max = (kk + BK < {k}) ? (kk + BK) : {k};",
            f"      const int j_max = (jj + BN < {n}) ? (jj + BN) : {n};",
            "      for (int i = ii; i < i_max; ++i) {",
            "        for (int kk2 = kk; kk2 < k_max; ++kk2) {",
            f"          const float a_val = {a}[i * {k} + kk2];",
            "          int j = jj;",
            "          #if defined(__AVX512F__)",
            "          {",
            "            const __m512 a_vec = _mm512_set1_ps(a_val);",
            "            for (; j + 15 < j_max; j += 16) {",
            f"              __m512 c_vec = _mm512_loadu_ps(&{c}[i * {n} + j]);",
            f"              const __m512 b_vec = _mm512_loadu_ps(&{b}[kk2 * {n} + j]);",
            "              c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);",
            f"              _mm512_storeu_ps(&{c}[i * {n} + j], c_vec);",
            "            }",
            "          }",
            "          #elif defined(__AVX2__)",
            "          {",
            "            const __m256 a_vec = _mm256_set1_ps(a_val);",
            "            for (; j + 7 < j_max; j += 8) {",
            f"              __m256 c_vec = _mm256_loadu_ps(&{c}[i * {n} + j]);",
            f"              const __m256 b_vec = _mm256_loadu_ps(&{b}[kk2 * {n} + j]);",
            "              c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);",
            f"              _mm256_storeu_ps(&{c}[i * {n} + j], c_vec);",
            "            }",
            "          }",
            "          #endif",
            "          #pragma omp simd",
            "          for (int j2 = j; j2 < j_max; ++j2) {",
            f"            {c}[i * {n} + j2] += a_val * {b}[kk2 * {n} + j2];",
            "          }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "}",
        ]
#todo
    def _render_cpp(
        self,
        *,
        ordered: List[Node],
        inputs: List[Node],
        output_node: Node,
        declarations: List[str],
        body: List[str],
        workspace_slots: List[tuple[str, str]] = (),
    ) -> str:
        max_rank = 8
        output_numel = self._numel_expr(output_node)
        output_name = "out_ptr"
        if output_node.op_type == OpType.INPUT:
            output_name = f"arg{inputs.index(output_node)}"
        signature = ", ".join(["const TensorDesc* inputs", "long long num_inputs", "TensorDesc* out_desc", "void* workspace"])

        ctype = "double" if self._compute_dtype == "float64" else "float"

        input_bindings = [f"{ctype}* arg{i} = static_cast<{ctype}*>(inputs[{i}].data);" for i in range(len(inputs))]

        # Symbolic dim extraction: const long long in0_d0 = inputs[0].shape[0]; ...
        dim_declarations: List[str] = []
        for idx, node in enumerate(inputs):
            for d in range(node.rank):
                dim_declarations.append(f"const long long in{idx}_d{d} = inputs[{idx}].shape[{d}];")

        # Workspace arena pointer arithmetic (zero runtime allocations)
        ws_declarations: List[str] = []
        if workspace_slots:
            ws_declarations.append(f"{ctype}* ws = static_cast<{ctype}*>(workspace);")
            offset_parts: List[str] = []
            for slot_name, numel_expr in workspace_slots:
                if not offset_parts:
                    ws_declarations.append(f"{ctype}* {slot_name} = ws;")
                else:
                    offset_expr = " + ".join(offset_parts)
                    ws_declarations.append(f"{ctype}* {slot_name} = ws + ({offset_expr});")
                offset_parts.append(numel_expr)

        # Rank-only guards (no shape/stride/numel checks for symbolic mode)
        input_guards: List[str] = []
        for i, node in enumerate(inputs):
            input_guards.append(
                f"if (inputs[{i}].data == nullptr) return {int(AbiStatus.INPUT_DATA_NULL_BASE) + i};"
            )
            input_guards.append(
                f"if (inputs[{i}].rank != {node.rank}LL) return {int(AbiStatus.INPUT_RANK_MISMATCH_BASE) + i};"
            )

        output_guards: List[str] = [
            f"if (out_desc == nullptr) return {int(AbiStatus.OUT_DESC_NULL)};",
            f"if (out_desc->data == nullptr) return {int(AbiStatus.OUT_DATA_NULL)};",
            f"if (out_desc->rank != {output_node.rank}LL) return {int(AbiStatus.OUT_RANK_MISMATCH)};",
        ]

        return "\n".join(
            [
                "#include <algorithm>",
                "#include <cfloat>",
                "#include <cstddef>",
                "#include <cmath>",
                "#include <cstdint>",
                "#include <immintrin.h>",
                "#ifdef _OPENMP",
                "#include <omp.h>",
                "#endif",
                "",
                "#ifdef _WIN32",
                "#define EXPORT __declspec(dllexport)",
                "#else",
                "#define EXPORT",
                "#endif",
                "",
                f"constexpr int kMaxRank = {max_rank};",
                "struct TensorDesc {",
                "    void* data;",
                "    long long numel;",
                "    long long rank;",
                "    long long shape[kMaxRank];",
                "    long long strides[kMaxRank];",
                "};",
                "",
                "extern \"C\" EXPORT int run_kernel(" + signature + ") {",
                f"    if (num_inputs != {len(inputs)}LL) return {int(AbiStatus.INPUT_COUNT_MISMATCH)};",
                *[f"    {line}" for line in input_guards],
                *[f"    {line}" for line in input_bindings],
                *[f"    {line}" for line in dim_declarations],
                *[f"    {line}" for line in output_guards],
                f"    {ctype}* out_ptr = static_cast<{ctype}*>(out_desc->data);",
                *[f"    {line}" for line in ws_declarations],
                *[f"    {line}" for line in declarations],
                *[f"    {line}" for line in body],
                f"    if ({output_name} != out_ptr) {{",
                f"        for (long long i = 0; i < {output_numel}; ++i) out_ptr[i] = {output_name}[i];",
                "    }",
                "    return 0;",
                "}",
            ]
        )

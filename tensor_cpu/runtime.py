"""JIT runtime: compile generated C++ and execute it via ctypes."""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from .abi import decode_abi_status
from .backend.codegen import CppCodegen, GeneratedKernel
from .ir.graph import Graph, Node

_MAX_RANK = 8  # ABI 支持的最大秩（8）


class TensorDesc(ctypes.Structure):  # C++ kernel 以此结构读取张量信息，运行时 ABI 描述符
    _fields_ = [
        ("data", ctypes.c_void_p),  # 指向张量数据的指针
        ("numel", ctypes.c_longlong),  # 张量元素总数
        ("rank", ctypes.c_longlong),  # 张量秩（维度数量）
        (
            "shape",
            ctypes.c_longlong * _MAX_RANK,
        ),  # 张量每个维度的大小，长度为_MAX_RANK，不足部分填1
        (
            "strides",
            ctypes.c_longlong * _MAX_RANK,
        ),  # 张量每个维度的步长，长度为_MAX_RANK，不足部分填0
    ]


class JITCompileError(RuntimeError):
    """Raised when local C++ compilation fails."""


def _find_compiler() -> str:
    for candidate in ("clang++", "g++", "cl"):
        path = shutil.which(candidate)
        if path:
            return candidate
    raise JITCompileError("No C++ compiler found (clang++, g++, or cl).")


def _build_command(compiler: str, cpp_path: Path, lib_path: Path, openmp: bool = True) -> List[str]:
    if compiler == "cl":
        cmd = [
            "cl",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/LD",
            str(cpp_path),
        ]
        if openmp:
            cmd.append("/openmp")
        cmd.extend(["/link", f"/OUT:{lib_path}"])
        return cmd

    cmd = [
        compiler,
        "-std=c++17",
        "-O3",
        "-march=native",
        "-shared",
        "-fPIC",
        str(cpp_path),
        "-o",
        str(lib_path),
    ]
    if openmp:
        cmd.append("-fopenmp")
    return cmd


class JITModule:
    """Compiled runtime module loaded by ctypes."""

    _SYM_DIM_RE = re.compile(r"^in(\d+)_d(\d+)$")

    def __init__(
        self,
        lib_path: Path,
        input_nodes: list[Node],
        output_node: Node,
        output_sym_shape: tuple[str, ...] = (),
        input_ranks: tuple[int, ...] = (),
        workspace_slots: tuple[tuple[str, tuple[str, ...]], ...] = (),
        exact_input_shapes: tuple[tuple[int, ...], ...] = (),
        tmp_dir: Path | None = None,
    ) -> None:
        self._lib_path = lib_path
        self._tmp_dir = tmp_dir
        self.input_shapes = [node.shape for node in input_nodes]
        self.input_ranks = input_ranks or tuple(node.rank for node in input_nodes)
        self.exact_input_shapes = exact_input_shapes
        self.output_shape = output_node.shape
        self.output_rank = int(output_node.rank)
        self.output_sym_shape = output_sym_shape
        self.output_dtype = output_node.dtype or "float32"
        self.input_specs = [_spec_from_node(node) for node in input_nodes]
        self.output_spec = _spec_from_node(output_node)
        self.workspace_slots = workspace_slots
        self._np_dtype = np.float64 if self.output_dtype == "float64" else np.float32
        self._workspace_buf: np.ndarray | None = None
        self._lib = ctypes.CDLL(str(lib_path))
        self._fn = self._lib.run_kernel
        self._fn.argtypes = [
            ctypes.POINTER(TensorDesc),
            ctypes.c_longlong,
            ctypes.POINTER(TensorDesc),
            ctypes.c_void_p,  # workspace
        ]
        self._fn.restype = ctypes.c_int

    def __del__(self) -> None:
        """Clean up the temporary compilation directory."""
        try:
            # Release the shared library handle before removing files.
            if hasattr(self, "_lib") and self._lib is not None:
                if hasattr(self._lib, "_handle") and _is_posix():
                    try:
                        ctypes.cdll.LoadLibrary("").close()  # no-op, just to avoid errors
                    except Exception:
                        pass
                self._lib = None
            if hasattr(self, "_tmp_dir") and self._tmp_dir is not None and self._tmp_dir.exists():
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass

    def _eval_sym_dim(
        self, expr: str, input_shapes: list[tuple[int, ...]]
    ) -> (
        int
    ):  # 将符号维度表达式解析为整数：若匹配 _SYM_DIM_RE，从 input_shapes 中取对应维度；否则尝试 int(expr)
        """Evaluate a symbolic dimension expression given actual input shapes."""
        m = self._SYM_DIM_RE.match(expr)
        if m:
            return input_shapes[int(m.group(1))][int(m.group(2))]
        try:
            return int(expr)
        except ValueError:
            raise ValueError(f"Cannot evaluate symbolic dim: {expr}")

    def _compute_output_shape(
        self, input_shapes: list[tuple[int, ...]]
    ) -> tuple[
        int, ...
    ]:  # 对符号表达式 output_sym_shape 中的每个维度表达式调用 _eval_sym_dim 计算实际维度大小，返回结果
        """Compute output shape from symbolic expressions and actual input shapes."""
        if not self.output_sym_shape:
            return self.output_shape
        return tuple(self._eval_sym_dim(d, input_shapes) for d in self.output_sym_shape)

    def _eval_workspace_size(
        self, input_shapes: list[tuple[int, ...]]
    ) -> (
        int
    ):  # 计算工作空间大小：对 workspace_slots 中的每个符号维度表达式调用 _eval_sym_dim 计算实际大小，并累加总和返回
        """Compute total workspace size in elements from symbolic slot dims."""
        total = 0
        for _, sym_dims in self.workspace_slots:
            size = 1
            for d in sym_dims:
                size *= self._eval_sym_dim(d, input_shapes)
            total += size
        return total

    def run(self, *inputs: np.ndarray, validate_inputs: bool = True) -> np.ndarray:
        if not inputs:
            raise ValueError("At least one input is required.")
        if validate_inputs and len(inputs) != len(self.input_ranks):
            raise ValueError(
                f"Expected {len(self.input_ranks)} inputs, got {len(inputs)}"
            )  # 输入校验

        in_arrays = [np.asarray(x, dtype=self._np_dtype, order="C") for x in inputs]
        if validate_inputs:
            for idx, (arr, expected_rank) in enumerate(zip(in_arrays, self.input_ranks)):
                if arr.ndim != expected_rank:
                    raise ValueError(
                        f"Input {idx} rank mismatch: expected {expected_rank}, got {arr.ndim}"
                    )
            if self.exact_input_shapes:
                for idx, (arr, expected_shape) in enumerate(
                    zip(in_arrays, self.exact_input_shapes)
                ):
                    if tuple(arr.shape) != tuple(expected_shape):
                        raise ValueError(
                            f"Input {idx} shape mismatch: expected exact traced shape {expected_shape}, got {tuple(arr.shape)}"
                        )

        actual_input_shapes = [tuple(arr.shape) for arr in in_arrays]
        try:
            out_shape = self._compute_output_shape(actual_input_shapes)
        except (IndexError, KeyError):
            # Fall back to letting the C++ kernel report the ABI error
            out_shape = self.output_shape
        out = np.empty(out_shape, dtype=self._np_dtype)  # 构建输入输出
        in_desc_array_t = TensorDesc * len(in_arrays)
        in_descs = in_desc_array_t(*[_desc_from_array(arr) for arr in in_arrays])
        out_desc = _desc_from_array(out)

        # Arena workspace: evaluate size, allocate/reuse buffer
        ws_size = self._eval_workspace_size(actual_input_shapes)
        if ws_size > 0:
            if self._workspace_buf is None or self._workspace_buf.size < ws_size:
                self._workspace_buf = np.empty(ws_size, dtype=self._np_dtype)
            ws_ptr = self._workspace_buf.ctypes.data
        else:
            ws_ptr = 0

        status = self._fn(
            in_descs,
            ctypes.c_longlong(len(in_arrays)),
            ctypes.byref(out_desc),
            ctypes.c_void_p(ws_ptr),
        )
        if status != 0:
            raise RuntimeError(
                f"Kernel execution failed with ABI status code {status} ({decode_abi_status(status)})"
            )
        return out


class JITEngine:
    """Compile Graph IR into native code and execute.

    Args:
        use_hpc_template: whether to enable HPC template specialization
        enable_memory_planner: whether to use memory planner
        passes: optional list of optimization pass tokens to run (e.g. ['pipeline']).
                If None, falls back to environment variable `TENSOR_PASSES`.
    """

    def __init__(
        self,
        use_hpc_template: bool = False,
        enable_memory_planner: bool = True,
        passes: list[str] | None = None,
    ) -> None:
        self.use_hpc_template = use_hpc_template
        self.enable_memory_planner = enable_memory_planner
        self.passes = passes

    def compile_graph(self, graph: Graph) -> JITModule:  # 编译Graph IR为c++代码，
        _ensure_single_output(graph)

        # Determine passes to run: constructor arg overrides env var.
        passes_to_run = self.passes
        if passes_to_run is None:
            env = os.environ.get("TENSOR_PASSES", "")
            passes_to_run = [p.strip() for p in env.split(",") if p.strip()] if env else []

        # Supported token: 'pipeline' will run the default pipeline (fusion + dce).
        if passes_to_run:
            from .passes import optimize_graph

            if "pipeline" in passes_to_run:
                stats = optimize_graph(graph)
                if os.environ.get("TENSOR_PASS_DEBUG"):
                    print("optimize_graph stats:", stats)

        codegen = CppCodegen(
            graph=graph,
            use_hpc_template=self.use_hpc_template,
            enable_memory_planner=self.enable_memory_planner,
        )
        kernel = codegen.generate()

        tmp_path = Path(tempfile.mkdtemp(prefix="tensor_cpu_"))
        cpp_path = tmp_path / "kernel.cpp"
        lib_name = "kernel.dll" if _is_windows() else "libkernel.so"
        lib_path = tmp_path / lib_name

        cpp_path.write_text(kernel.source, encoding="utf-8")  # 将生成的c++代码写入临时文件

        compiler = _find_compiler()
        cmd = _build_command(compiler, cpp_path, lib_path, openmp=True)  # 构建编译命令
        proc = subprocess.run(cmd, capture_output=True, text=True)  # 执行编译命令为so/dll文件
        if proc.returncode != 0:
            raise JITCompileError(
                f"Compile failed with {compiler}\ncmd: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        output_node = _resolve_output_node(graph)
        input_nodes = _resolve_input_nodes(graph)
        return JITModule(
            lib_path=lib_path,
            input_nodes=input_nodes,
            output_node=output_node,
            output_sym_shape=kernel.output_sym_shape,
            input_ranks=kernel.input_ranks,
            workspace_slots=kernel.workspace_slots,
            exact_input_shapes=kernel.exact_input_shapes,
            tmp_dir=tmp_path,
        )  # 编译成功后，返回一个JITModule实例，包含了编译好的库路径、输入输出节点信息等


def _num_input_nodes(graph: Graph) -> int:
    from .ir.ops import OpType

    return sum(1 for node in graph.nodes() if node.op_type == OpType.INPUT)


def _is_windows() -> bool:
    import platform

    return platform.system().lower().startswith("win")


def _is_posix() -> bool:
    return os.name == "posix"


def _resolve_output_shape(graph: Graph) -> tuple[int, ...]:
    return _resolve_output_node(graph).shape


def _resolve_output_node(graph: Graph) -> Node:
    _ensure_single_output(graph)
    if graph.output_ids:
        return graph.get_node(graph.output_ids[-1])
    ordered = graph.topological_sort()
    if not ordered:
        raise ValueError("Graph is empty.")
    return ordered[-1]


def _resolve_input_nodes(graph: Graph) -> list[Node]:
    from .ir.ops import OpType

    nodes: list[Node] = []
    for node in graph.topological_sort():
        if node.op_type == OpType.INPUT:
            nodes.append(node)
    return nodes


def _ensure_single_output(graph: Graph) -> None:
    if len(graph.output_ids) > 1:
        raise ValueError("Only single-output graphs are supported on the stable runtime path.")


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _spec_from_node(node: Node) -> dict:
    return {
        "shape": tuple(node.shape),
        "strides": tuple(node.strides),
        "rank": int(node.rank),
        "numel": int(node.numel),
    }


def _desc_from_array(arr: np.ndarray) -> TensorDesc:
    shape = tuple(int(v) for v in arr.shape)
    rank = len(shape)
    if rank > _MAX_RANK:
        raise ValueError(f"Rank {rank} exceeds ABI limit {_MAX_RANK}")

    strides_elems = tuple(int(v // arr.itemsize) for v in arr.strides)
    desc = TensorDesc()
    desc.data = arr.ctypes.data
    desc.rank = rank
    desc.numel = _numel(shape)
    for i in range(_MAX_RANK):
        desc.shape[i] = 1
        desc.strides[i] = 0
    for i, dim in enumerate(shape):
        desc.shape[i] = dim
    for i, stride in enumerate(strides_elems):
        desc.strides[i] = stride
    return desc

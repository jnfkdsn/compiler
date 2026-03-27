"""pybind11-based runtime bindings.

This module provides a modern C++/Python bridge using pybind11/nanobind,
replacing the fragile ctypes-based FFI with:
- Strong type safety
- Automatic memory management
- Exception translation
- DLPack zero-copy support
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tensor_cpu.abi import decode_abi_status
from tensor_cpu.backend.codegen import CppCodegen, GeneratedKernel
from tensor_cpu.ir.graph import Graph, Node

CPP_BINDINGS_HEADER = """
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>

namespace py = pybind11;

namespace tensor_cpu {

constexpr int MAX_RANK = 8;

struct TensorDesc {
    void* data;
    int64_t numel;
    int64_t rank;
    int64_t shape[MAX_RANK];
    int64_t strides[MAX_RANK];
};

class KernelError : public std::runtime_error {
public:
    explicit KernelError(const std::string& msg, int code)
        : std::runtime_error(msg), error_code(code) {}
    int error_code;
};

class Tensor {
public:
    Tensor() : data_(nullptr), numel_(0), dtype_(DataType::Float32) {}
    
    Tensor(py::array_t<float> arr) {
        init_from_array(arr);
    }
    
    Tensor(py::array_t<double> arr) {
        init_from_array(arr);
        dtype_ = DataType::Float64;
    }
    
    void* data() { return data_; }
    const void* data() const { return data_; }
    int64_t numel() const { return numel_; }
    int64_t rank() const { return static_cast<int64_t>(shape_.size()); }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    
    py::array_t<float> as_float32() const {
        return py::array_t<float>(shape_, strides_, static_cast<float*>(data_), py::cast(this));
    }
    
    py::array_t<double> as_float64() const {
        return py::array_t<double>(shape_, strides_, static_cast<double*>(data_), py::cast(this));
    }
    
    TensorDesc to_desc() const {
        TensorDesc desc;
        desc.data = data_;
        desc.numel = numel_;
        desc.rank = rank();
        for (int i = 0; i < MAX_RANK; ++i) {
            desc.shape[i] = (i < rank()) ? shape_[i] : 1;
            desc.strides[i] = (i < rank()) ? strides_[i] : 0;
        }
        return desc;
    }
    
    bool is_float64() const { return dtype_ == DataType::Float64; }
    
private:
    enum class DataType { Float32, Float64 };
    
    void init_from_array(py::buffer_info& info) {
        shape_.assign(info.shape.begin(), info.shape.end());
        strides_.assign(info.strides.begin(), info.strides.end());
        for (auto& s : strides_) {
            s /= info.itemsize;
        }
        data_ = info.ptr;
        numel_ = 1;
        for (auto d : shape_) numel_ *= d;
    }
    
    template<typename T>
    void init_from_array(py::array_t<T> arr) {
        py::buffer_info info = arr.request();
        init_from_array(info);
    }
    
    void* data_;
    int64_t numel_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    DataType dtype_ = DataType::Float32;
};

class CompiledKernel {
public:
    CompiledKernel(const std::string& lib_path, const std::string& entry_name)
        : lib_path_(lib_path), entry_name_(entry_name) {
        load_kernel();
    }
    
    py::array run(py::list inputs, py::dict kwargs);
    
    const std::vector<int64_t>& input_ranks() const { return input_ranks_; }
    const std::vector<int64_t>& output_sym_shape() const { return output_sym_shape_; }
    
private:
    void load_kernel();
    
    std::string lib_path_;
    std::string entry_name_;
    void* lib_handle_ = nullptr;
    int (*kernel_fn_)(TensorDesc*, int64_t, TensorDesc*, void*) = nullptr;
    std::vector<int64_t> input_ranks_;
    std::vector<int64_t> output_sym_shape_;
    std::vector<std::pair<std::string, std::vector<int64_t>>> workspace_slots_;
};

void register_exception_translations(py::module& m);

}  // namespace tensor_cpu
"""

CPP_BINDINGS_IMPL = """
#include "bindings.h"
#include <dlfcn.h>
#include <sstream>

namespace tensor_cpu {

void CompiledKernel::load_kernel() {
    lib_handle_ = dlopen(lib_path_.c_str(), RTLD_NOW);
    if (!lib_handle_) {
        throw KernelError("Failed to load kernel library: " + std::string(dlerror()), -1);
    }
    
    kernel_fn_ = reinterpret_cast<int(*)(TensorDesc*, int64_t, TensorDesc*, void*)>(
        dlsym(lib_handle_, entry_name_.c_str()));
    if (!kernel_fn_) {
        throw KernelError("Failed to find kernel entry: " + entry_name_, -1);
    }
}

py::array CompiledKernel::run(py::list inputs, py::dict kwargs) {
    if (inputs.size() != input_ranks_.size()) {
        std::ostringstream oss;
        oss << "Input count mismatch: expected " << input_ranks_.size() 
            << ", got " << inputs.size();
        throw std::invalid_argument(oss.str());
    }
    
    std::vector<Tensor> input_tensors;
    std::vector<TensorDesc> input_descs;
    
    for (auto item : inputs) {
        if (py::isinstance<py::array_t<float>>(item)) {
            input_tensors.emplace_back(py::cast<py::array_t<float>>(item));
        } else if (py::isinstance<py::array_t<double>>(item)) {
            input_tensors.emplace_back(py::cast<py::array_t<double>>(item));
        } else {
            throw std::invalid_argument("Input must be numpy array of float32 or float64");
        }
        input_descs.push_back(input_tensors.back().to_desc());
    }
    
    // Compute output shape
    std::vector<int64_t> out_shape;
    for (auto dim : output_sym_shape_) {
        out_shape.push_back(dim);
    }
    
    py::array_t<float> output(out_shape);
    Tensor out_tensor(output);
    TensorDesc out_desc = out_tensor.to_desc();
    
    // Allocate workspace
    std::vector<float> workspace;
    void* ws_ptr = nullptr;
    size_t ws_size = 0;
    for (const auto& [name, dims] : workspace_slots_) {
        size_t slot_size = 1;
        for (auto d : dims) slot_size *= d;
        ws_size += slot_size;
    }
    if (ws_size > 0) {
        workspace.resize(ws_size);
        ws_ptr = workspace.data();
    }
    
    int status = kernel_fn_(
        input_descs.data(),
        static_cast<int64_t>(inputs.size()),
        &out_desc,
        ws_ptr
    );
    
    if (status != 0) {
        throw KernelError("Kernel execution failed with status " + std::to_string(status), status);
    }
    
    return output;
}

void register_exception_translations(py::module& m) {
    static py::exception<KernelError> exc(m, "KernelError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const KernelError& e) {
            exc(e.what());
        }
    });
}

PYBIND11_MODULE(tensor_cpu_native, m) {
    m.doc() = "Tensor CPU native bindings";
    
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<py::array_t<float>>())
        .def(py::init<py::array_t<double>>())
        .def_property_readonly("data", [](Tensor& t) { 
            return reinterpret_cast<uintptr_t>(t.data()); 
        })
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("strides", &Tensor::strides)
        .def("as_float32", &Tensor::as_float32)
        .def("as_float64", &Tensor::as_float64);
    
    py::class_<CompiledKernel>(m, "CompiledKernel")
        .def(py::init<std::string, std::string>())
        .def("run", &CompiledKernel::run)
        .def_property_readonly("input_ranks", &CompiledKernel::input_ranks)
        .def_property_readonly("output_sym_shape", &CompiledKernel::output_sym_shape);
    
    register_exception_translations(m);
}

}  // namespace tensor_cpu
"""


class Pybind11Runtime:
    """Runtime using pybind11 for C++ bindings."""

    def __init__(self) -> None:
        self._native_module = None
        self._ensure_native_module()

    def _ensure_native_module(self) -> None:
        """Build and load the native module if not already loaded."""
        try:
            from . import tensor_cpu_native

            self._native_module = tensor_cpu_native
        except ImportError:
            self._build_native_module()

    def _build_native_module(self) -> None:
        """Build the native module from C++ source."""
        tmp_path = Path(tempfile.mkdtemp(prefix="tensor_cpu_native_"))

        header_path = tmp_path / "bindings.h"
        impl_path = tmp_path / "bindings.cpp"

        header_path.write_text(CPP_BINDINGS_HEADER)
        impl_path.write_text(CPP_BINDINGS_IMPL)

        compiler = self._find_compiler()
        cmd = self._build_compile_command(compiler, impl_path, tmp_path)

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
        if proc.returncode != 0:
            raise RuntimeError(f"Native module build failed:\n{proc.stderr}")

        import sys

        if str(tmp_path) not in sys.path:
            sys.path.insert(0, str(tmp_path))

        import tensor_cpu_native

        self._native_module = tensor_cpu_native

    def _find_compiler(self) -> str:
        for candidate in ("clang++", "g++"):
            if shutil.which(candidate):
                return candidate
        raise RuntimeError("No C++ compiler found")

    def _build_compile_command(self, compiler: str, source: Path, out_dir: Path) -> List[str]:
        import sysconfig

        python_include = sysconfig.get_path("include")
        pybind11_include = self._get_pybind11_include()

        return [
            compiler,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            f"-I{python_include}",
            f"-I{pybind11_include}",
            str(source),
            "-o",
            str(out_dir / "tensor_cpu_native.so"),
        ]

    def _get_pybind11_include(self) -> str:
        try:
            import pybind11

            return pybind11.get_include()
        except ImportError:
            raise RuntimeError("pybind11 not installed. Run: pip install pybind11")


class DLPackTensor:
    """DLPack-based zero-copy tensor wrapper."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array
        self._capsule = None

    def __dlpack__(self, stream: Optional[int] = None) -> Any:
        """Return a DLPack capsule for zero-copy sharing."""
        if self._capsule is None:
            self._capsule = self._create_dlpack_capsule()
        return self._capsule

    def __dlpack_device__(self) -> Tuple[int, int]:
        """Return (device_type, device_id) tuple."""
        return (1, 0)  # kDLCPU, device_id=0

    def _create_dlpack_capsule(self) -> Any:
        """Create a DLPack capsule from the numpy array."""
        try:
            return self._array.__dlpack__()
        except AttributeError:
            return self._create_dlpack_capsule_fallback()

    def _create_dlpack_capsule_fallback(self) -> Any:
        """Fallback DLPack capsule creation for older numpy versions."""
        import ctypes

        class DLDevice(ctypes.Structure):
            _fields_ = [
                ("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int),
            ]

        class DLTensor(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("device", DLDevice),
                ("ndim", ctypes.c_int),
                ("dtype_code", ctypes.c_uint8),
                ("dtype_bits", ctypes.c_uint8),
                ("dtype_lanes", ctypes.c_uint16),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64),
            ]

        arr = self._array
        ndim = arr.ndim

        shape = (ctypes.c_int64 * ndim)(*arr.shape)
        strides_arr = tuple(s // arr.itemsize for s in arr.strides)
        strides = (ctypes.c_int64 * ndim)(*strides_arr)

        dl_tensor = DLTensor()
        dl_tensor.data = arr.ctypes.data
        dl_tensor.device = DLDevice(1, 0)  # kDLCPU
        dl_tensor.ndim = ndim
        dl_tensor.dtype_code = 2 if arr.dtype == np.float32 else 3  # kDLFloat
        dl_tensor.dtype_bits = arr.itemsize * 8
        dl_tensor.dtype_lanes = 1
        dl_tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
        dl_tensor.strides = ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64))
        dl_tensor.byte_offset = 0

        class DLManagedTensor(ctypes.Structure):
            pass

        DLManagedTensor._fields_ = [
            ("dl_tensor", DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))),
        ]

        managed = DLManagedTensor()
        managed.dl_tensor = dl_tensor
        managed.manager_ctx = None

        capsule = ctypes.pythonapi.PyCapsule_New(ctypes.byref(managed), b"dltensor", None)

        return capsule

    @staticmethod
    def from_dlpack(capsule: Any) -> np.ndarray:
        """Create a numpy array from a DLPack capsule."""
        try:
            return np.from_dlpack(capsule)
        except AttributeError:
            return DLPackTensor._from_dlpack_fallback(capsule)

    @staticmethod
    def _from_dlpack_fallback(capsule: Any) -> np.ndarray:
        """Fallback DLPack to numpy conversion."""
        import ctypes

        ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")

        class DLDevice(ctypes.Structure):
            _fields_ = [
                ("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int),
            ]

        class DLTensor(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.c_void_p),
                ("device", DLDevice),
                ("ndim", ctypes.c_int),
                ("dtype_code", ctypes.c_uint8),
                ("dtype_bits", ctypes.c_uint8),
                ("dtype_lanes", ctypes.c_uint16),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64),
            ]

        dl_tensor = ctypes.cast(ptr, ctypes.POINTER(DLTensor)).contents

        shape = tuple(dl_tensor.shape[i] for i in range(dl_tensor.ndim))
        strides = tuple(
            dl_tensor.strides[i] * dl_tensor.dtype_bits // 8 for i in range(dl_tensor.ndim)
        )

        dtype = np.float32 if dl_tensor.dtype_bits == 32 else np.float64

        arr = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=(ctypes.c_char * (dl_tensor.numel * dl_tensor.dtype_bits // 8)).from_address(
                dl_tensor.data
            ),
            strides=strides,
        )

        ctypes.pythonapi.PyCapsule_SetName(capsule, b"used_dltensor")

        return arr


class PybindJITModule:
    """JIT module using pybind11 bindings."""

    def __init__(
        self,
        lib_path: Path,
        input_nodes: List[Node],
        output_node: Node,
        output_sym_shape: Tuple[str, ...] = (),
        input_ranks: Tuple[int, ...] = (),
        workspace_slots: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
    ) -> None:
        self._lib_path = lib_path
        self.input_ranks = input_ranks
        self.output_sym_shape = output_sym_shape
        self.workspace_slots = workspace_slots
        self.output_dtype = output_node.dtype or "float32"
        self._np_dtype = np.float64 if self.output_dtype == "float64" else np.float32

        self._kernel = self._load_kernel()

    def _load_kernel(self):
        """Load the compiled kernel using pybind11."""
        runtime = Pybind11Runtime()
        if runtime._native_module:
            return runtime._native_module.CompiledKernel(str(self._lib_path), "run_kernel")
        return None

    def run(self, *inputs: np.ndarray, validate_inputs: bool = True) -> np.ndarray:
        """Execute the kernel with zero-copy DLPack."""
        if not inputs:
            raise ValueError("At least one input is required.")

        if validate_inputs and len(inputs) != len(self.input_ranks):
            raise ValueError(f"Expected {len(self.input_ranks)} inputs, got {len(inputs)}")

        in_arrays = [np.asarray(x, dtype=self._np_dtype, order="C") for x in inputs]

        if validate_inputs:
            for idx, (arr, expected_rank) in enumerate(zip(in_arrays, self.input_ranks)):
                if arr.ndim != expected_rank:
                    raise ValueError(
                        f"Input {idx} rank mismatch: expected {expected_rank}, got {arr.ndim}"
                    )

        if self._kernel:
            return self._run_with_pybind(in_arrays)
        else:
            return self._run_with_ctypes_fallback(in_arrays)

    def _run_with_pybind(self, in_arrays: List[np.ndarray]) -> np.ndarray:
        """Run using pybind11 bindings."""
        import pybind11

        dlpack_inputs = [DLPackTensor(arr) for arr in in_arrays]
        input_list = pybind11.list([arr.__dlpack__() for arr in dlpack_inputs])

        result = self._kernel.run(input_list, {})
        return np.asarray(result)

    def _run_with_ctypes_fallback(self, in_arrays: List[np.ndarray]) -> np.ndarray:
        """Fallback to ctypes if pybind11 is not available."""
        import ctypes

        from .runtime import TensorDesc, _desc_from_array, _numel

        actual_input_shapes = [tuple(arr.shape) for arr in in_arrays]
        out_shape = self._compute_output_shape(actual_input_shapes)
        out = np.empty(out_shape, dtype=self._np_dtype)

        in_desc_array_t = TensorDesc * len(in_arrays)
        in_descs = in_desc_array_t(*[_desc_from_array(arr) for arr in in_arrays])
        out_desc = _desc_from_array(out)

        ws_size = self._eval_workspace_size(actual_input_shapes)
        ws_ptr = 0
        if ws_size > 0:
            ws_buf = np.empty(ws_size, dtype=self._np_dtype)
            ws_ptr = ws_buf.ctypes.data

        lib = ctypes.CDLL(str(self._lib_path))
        fn = lib.run_kernel
        fn.argtypes = [
            ctypes.POINTER(TensorDesc),
            ctypes.c_longlong,
            ctypes.POINTER(TensorDesc),
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int

        status = fn(in_descs, len(in_arrays), ctypes.byref(out_desc), ctypes.c_void_p(ws_ptr))
        if status != 0:
            raise RuntimeError(
                f"Kernel execution failed with ABI status code {status} ({decode_abi_status(status)})"
            )

        return out

    def _compute_output_shape(self, input_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        """Compute output shape from symbolic expressions."""
        if not self.output_sym_shape:
            return ()

        result = []
        for expr in self.output_sym_shape:
            result.append(self._eval_sym_dim(expr, input_shapes))
        return tuple(result)

    def _eval_sym_dim(self, expr: str, input_shapes: List[Tuple[int, ...]]) -> int:
        """Evaluate a symbolic dimension expression."""
        import re

        m = re.match(r"^in(\d+)_d(\d+)$", expr)
        if m:
            return input_shapes[int(m.group(1))][int(m.group(2))]
        try:
            return int(expr)
        except ValueError:
            raise ValueError(f"Cannot evaluate symbolic dim: {expr}")

    def _eval_workspace_size(self, input_shapes: List[Tuple[int, ...]]) -> int:
        """Compute total workspace size in elements."""
        total = 0
        for _, sym_dims in self.workspace_slots:
            size = 1
            for d in sym_dims:
                size *= self._eval_sym_dim(d, input_shapes)
            total += size
        return total


class PybindJITEngine:
    """JIT engine using pybind11 bindings."""

    def __init__(
        self,
        use_hpc_template: bool = False,
        enable_memory_planner: bool = True,
        passes: Optional[List[str]] = None,
    ) -> None:
        self.use_hpc_template = use_hpc_template
        self.enable_memory_planner = enable_memory_planner
        self.passes = passes

    def compile_graph(self, graph: Graph) -> PybindJITModule:
        """Compile a Graph IR into a native module."""
        import os

        passes_to_run = self.passes
        if passes_to_run is None:
            env = os.environ.get("TENSOR_PASSES", "")
            passes_to_run = [p.strip() for p in env.split(",") if p.strip()] if env else []

        if passes_to_run:
            from .passes import optimize_graph

            if "pipeline" in passes_to_run:
                optimize_graph(graph)

        codegen = CppCodegen(
            graph=graph,
            use_hpc_template=self.use_hpc_template,
            enable_memory_planner=self.enable_memory_planner,
        )
        kernel = codegen.generate()

        tmp_path = Path(tempfile.mkdtemp(prefix="tensor_cpu_"))
        cpp_path = tmp_path / "kernel.cpp"
        lib_name = "kernel.dll" if self._is_windows() else "libkernel.so"
        lib_path = tmp_path / lib_name

        cpp_path.write_text(kernel.source, encoding="utf-8")

        compiler = self._find_compiler()
        cmd = self._build_command(compiler, cpp_path, lib_path, openmp=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(f"Compile failed with {compiler}\nstderr:\n{proc.stderr}")

        output_node = self._resolve_output_node(graph)
        input_nodes = self._resolve_input_nodes(graph)

        return PybindJITModule(
            lib_path=lib_path,
            input_nodes=input_nodes,
            output_node=output_node,
            output_sym_shape=kernel.output_sym_shape,
            input_ranks=kernel.input_ranks,
            workspace_slots=kernel.workspace_slots,
        )

    def _find_compiler(self) -> str:
        for candidate in ("clang++", "g++", "cl"):
            if shutil.which(candidate):
                return candidate
        raise RuntimeError("No C++ compiler found")

    def _build_command(
        self, compiler: str, cpp_path: Path, lib_path: Path, openmp: bool = True
    ) -> List[str]:
        if compiler == "cl":
            cmd = ["cl", "/std:c++17", "/O2", "/EHsc", "/LD", str(cpp_path)]
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

    def _is_windows(self) -> bool:
        import platform

        return platform.system().lower().startswith("win")

    def _resolve_output_node(self, graph: Graph) -> Node:
        if graph.output_ids:
            return graph.get_node(graph.output_ids[-1])
        ordered = graph.topological_sort()
        if not ordered:
            raise ValueError("Graph is empty.")
        return ordered[-1]

    def _resolve_input_nodes(self, graph: Graph) -> List[Node]:
        from tensor_cpu.ir.ops import OpType

        nodes = []
        for node in graph.topological_sort():
            if node.op_type == OpType.INPUT:
                nodes.append(node)
        return nodes

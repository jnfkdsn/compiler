"""Public package entry for the stable tensor_cpu execution path."""

from .abi import AbiStatus, decode_abi_status
from .autodiff.train_jit import (
    compile_adam_update_kernels,
    compile_sgd_update_kernel,
    compile_training_step,
)
from .backend.codegen import CppCodegen, GeneratedKernel
from .frontend.tracer import TraceContext
from .passes import optimize_graph
from .runtime import JITEngine
from .static_graph import StaticGraph, SymbolicTensor
from .tensor import Tensor
from .jit_matmul import enable_jit_matmul, disable_jit_matmul

__all__ = [
    "Tensor",
    "enable_jit_matmul",
    "disable_jit_matmul",
    "TraceContext",
    "CppCodegen",
    "GeneratedKernel",
    "JITEngine",
    "StaticGraph",
    "SymbolicTensor",
    "compile_training_step",
    "compile_sgd_update_kernel",
    "compile_adam_update_kernels",
    "AbiStatus",
    "decode_abi_status",
    "optimize_graph",
]

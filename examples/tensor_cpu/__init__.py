"""tensor_cpu package entry."""

from .codegen import CppCodegen
from . import nn, optim, jit
from .abi import AbiStatus, decode_abi_status
from .lazy import LazyTensor, lazy_binary_cross_entropy, lazy_mse_loss
from .passes import optimize_graph
from .runtime import JITEngine
from .static_graph import StaticGraph, SymbolicTensor
from .train_jit import compile_adam_update_kernels, compile_sgd_update_kernel, compile_training_step
from .tensor import Tensor
from .jit_matmul import enable_jit_matmul, disable_jit_matmul
from .tracer import TraceContext

__all__ = [
	"Tensor",
	"enable_jit_matmul",
	"disable_jit_matmul",
	"TraceContext",
	"CppCodegen",
	"JITEngine",
	"StaticGraph",
	"SymbolicTensor",
	"LazyTensor",
	"lazy_mse_loss",
	"lazy_binary_cross_entropy",
	"compile_training_step",
	"compile_sgd_update_kernel",
	"compile_adam_update_kernels",
	"AbiStatus",
	"decode_abi_status",
	"optimize_graph",
	"nn",
	"optim",
	"jit",
]

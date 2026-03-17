"""Optimization passes."""

from .dce import dead_code_elimination
from .fusion import fuse_matmul_bias_relu
from .pipeline import optimize_graph
from .constfold import constant_fold
from .cse import cse

__all__ = [
	"dead_code_elimination",
	"fuse_matmul_bias_relu",
	"optimize_graph",
	"constant_fold",
	"cse",
]

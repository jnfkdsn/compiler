"""Supported operation types for graph tracing."""

from __future__ import annotations

from enum import Enum


class OpType(str, Enum):
    """Operation kinds that can appear in the graph."""

    INPUT = "input"
    CONST = "const"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    EQ = "eq"
    MATMUL = "matmul"
    TRANSPOSE = "transpose"
    BROADCAST_TO = "broadcast_to"
    RELU = "relu"
    RELU_GRAD = "relu_grad"
    EXP = "exp"
    LOG = "log"
    SIGMOID = "sigmoid"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    FUSED_MATMUL_BIAS = "fused_matmul_bias"
    FUSED_MATMUL_BIAS_RELU = "fused_matmul_bias_relu"
    OUTPUT = "output"


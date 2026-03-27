"""Optimization pipeline utilities."""

from __future__ import annotations

from .constfold import constant_fold
from .cse import cse
from .dce import dead_code_elimination
from .fusion import fuse_matmul_bias_relu


def optimize_graph(graph) -> dict:
    """Run canonical optimization passes and report stats.

    Pipeline: constant-fold -> cse -> fusion -> dce
    """
    stats = {}
    stats["const_folded"] = constant_fold(graph)
    stats["cse_removed"] = cse(graph)
    stats["fused_subgraphs"] = fuse_matmul_bias_relu(graph)
    stats["dce_removed"] = dead_code_elimination(graph)
    return stats

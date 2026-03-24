"""Core JIT compilation tests: naive codegen, fusion pass, HPC matmul, new ops, ABI guards."""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, ".")

from tensor_cpu import (
    AbiStatus,
    JITEngine,
    Tensor,
    TraceContext,
    decode_abi_status,
    jit,
    optimize_graph,
)
from tensor_cpu.runtime import JITCompileError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_relu_graph(x_np, w_np, b_np):
    """Trace: relu(x @ w + b)"""
    with TraceContext() as tc:
        x = Tensor.from_numpy(x_np, name="x")
        w = Tensor.from_numpy(w_np, name="w")
        b = Tensor.from_numpy(b_np, name="b")
        out = (x @ w) + b
        out = out.relu().mark_as_output()
        return tc.graph


def build_add_graph(x_np, y_np):
    """Trace: x + y"""
    with TraceContext() as tc:
        x = Tensor.from_numpy(x_np, name="x")
        y = Tensor.from_numpy(y_np, name="y")
        out = (x + y).mark_as_output()
        return tc.graph


# ---------------------------------------------------------------------------
# 1. Naive JIT correctness
# ---------------------------------------------------------------------------

def test_naive_jit():
    np.random.seed(0)
    x = np.random.randn(4, 8).astype(np.float32)
    w = np.random.randn(8, 6).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)

    graph = build_relu_graph(x, w, b)
    engine = JITEngine(use_hpc_template=False)
    module = engine.compile_graph(graph)
    jit_out = module.run(x, w, b)

    ref = np.maximum(x @ w + b, 0.0)
    np.testing.assert_allclose(jit_out, ref, rtol=1e-5, atol=1e-5)
    print("  PASS: naive JIT")


# ---------------------------------------------------------------------------
# 2. Fusion pass correctness
# ---------------------------------------------------------------------------

def test_fusion():
    np.random.seed(1)
    x = np.random.randn(32, 64).astype(np.float32)
    w = np.random.randn(64, 32).astype(np.float32)
    b = np.random.randn(32).astype(np.float32)

    # baseline (no fusion)
    graph_base = build_relu_graph(x, w, b)
    engine = JITEngine(use_hpc_template=False)
    mod_base = engine.compile_graph(graph_base)
    out_base = mod_base.run(x, w, b)

    # fused
    graph_fused = build_relu_graph(x, w, b)
    stats = optimize_graph(graph_fused)
    mod_fused = engine.compile_graph(graph_fused)
    out_fused = mod_fused.run(x, w, b)

    np.testing.assert_allclose(out_base, out_fused, rtol=1e-5, atol=1e-5)
    print(f"  PASS: fusion (fused={stats['fused_subgraphs']}, dce={stats['dce_removed']})")


# ---------------------------------------------------------------------------
# 3. HPC matmul correctness
# ---------------------------------------------------------------------------

def test_hpc_matmul():
    np.random.seed(2)
    x = np.random.randn(64, 128).astype(np.float32)
    w = np.random.randn(128, 64).astype(np.float32)
    b = np.random.randn(64).astype(np.float32)

    graph = build_relu_graph(x, w, b)
    engine_hpc = JITEngine(use_hpc_template=True)
    mod = engine_hpc.compile_graph(graph)
    out = mod.run(x, w, b)

    ref = np.maximum(x @ w + b, 0.0)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
    print("  PASS: HPC matmul")


def test_hpc_matmul_rejects_dynamic_shape_reuse():
    np.random.seed(200)
    x_trace = np.random.randn(4, 8).astype(np.float32)
    w = np.random.randn(8, 6).astype(np.float32)

    with TraceContext() as tc:
        xt = Tensor.from_numpy(x_trace, name="x")
        wt = Tensor.from_numpy(w, name="w")
        (xt @ wt).mark_as_output()
        graph = tc.graph

    mod = JITEngine(use_hpc_template=True).compile_graph(graph)

    x_other = np.random.randn(7, 8).astype(np.float32)
    try:
        mod.run(x_other, w)
        raise AssertionError("Expected exact-shape guard for HPC-specialized matmul")
    except ValueError as exc:
        assert "exact traced shape" in str(exc)


# ---------------------------------------------------------------------------
# 4. New ops: exp, log, sigmoid, sum, mean, max
# ---------------------------------------------------------------------------

def test_new_ops():
    np.random.seed(3)
    x_np = np.random.randn(4, 8).astype(np.float32)
    y_np = np.random.randn(4, 8).astype(np.float32)
    z_np = np.abs(np.random.randn(4, 8).astype(np.float32)) + 0.1

    with TraceContext() as tc:
        x = Tensor.from_numpy(x_np, name="x")
        y = Tensor.from_numpy(y_np, name="y")
        z = Tensor.from_numpy(z_np, name="z")
        diff = (x - y) / z
        pipe = diff.exp().log().sigmoid()
        reduced = pipe.sum(axis=1).sum() + pipe.mean(axis=0).mean()
        reduced.mark_as_output()
        graph = tc.graph

    engine = JITEngine(use_hpc_template=False)
    mod = engine.compile_graph(graph)
    jit_out = mod.run(x_np, y_np, z_np)

    diff_np = (x_np - y_np) / z_np
    pipe_np = 1.0 / (1.0 + np.exp(-np.log(np.exp(diff_np))))
    ref = pipe_np.sum(axis=1).sum() + pipe_np.mean(axis=0).mean()
    np.testing.assert_allclose(jit_out, ref, rtol=1e-5, atol=1e-5)
    print("  PASS: new ops")


# ---------------------------------------------------------------------------
# 5. ABI guard checks
# ---------------------------------------------------------------------------

def test_abi_guards():
    np.random.seed(4)
    x = np.random.randn(4, 8).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)

    graph = build_add_graph(x, y)
    engine = JITEngine(use_hpc_template=False)
    mod = engine.compile_graph(graph)

    # Normal path
    out = mod.run(x, y)
    np.testing.assert_allclose(out, x + y, rtol=1e-5, atol=1e-5)

    # Different shapes, same rank → should work (symbolic shapes)
    x2 = np.random.randn(6, 10).astype(np.float32)
    y2 = np.random.randn(6, 10).astype(np.float32)
    out2 = mod.run(x2, y2)
    np.testing.assert_allclose(out2, x2 + y2, rtol=1e-5, atol=1e-5)

    # Rank mismatch → should raise ValueError from Python side
    x3 = np.random.randn(3).astype(np.float32)
    try:
        mod.run(x3, y)
        raise AssertionError("Expected ValueError for rank mismatch")
    except ValueError:
        pass

    print("  PASS: ABI guards")


# ---------------------------------------------------------------------------
# 6. Symbolic shape reuse
# ---------------------------------------------------------------------------

def test_symbolic_shapes():
    np.random.seed(5)
    x = np.random.randn(4, 8).astype(np.float32)
    w = np.random.randn(8, 6).astype(np.float32)

    with TraceContext() as tc:
        xt = Tensor.from_numpy(x, name="x")
        wt = Tensor.from_numpy(w, name="w")
        out = (xt @ wt).mark_as_output()
        graph = tc.graph

    engine = JITEngine()
    mod = engine.compile_graph(graph)

    # Same shape
    r1 = mod.run(x, w)
    np.testing.assert_allclose(r1, x @ w, rtol=1e-4, atol=1e-4)

    # Different batch dim, same rank
    x2 = np.random.randn(7, 8).astype(np.float32)
    r2 = mod.run(x2, w)
    np.testing.assert_allclose(r2, x2 @ w, rtol=1e-4, atol=1e-4)
    assert r2.shape == (7, 6)

    print("  PASS: symbolic shapes")


def test_fused_symbolic_shapes_dynamic_batch():
    np.random.seed(6)
    w = np.random.randn(8, 6).astype(np.float32)
    b = np.random.randn(6).astype(np.float32)

    x_trace = np.random.randn(4, 8).astype(np.float32)
    g = build_relu_graph(x_trace, w, b)
    stats = optimize_graph(g)
    assert stats["fused_subgraphs"] >= 1, "Expected at least one fused subgraph"

    mod = JITEngine(use_hpc_template=False).compile_graph(g)

    x1 = np.random.randn(4, 8).astype(np.float32)
    y1 = mod.run(x1, w, b)
    ref1 = np.maximum(x1 @ w + b, 0.0)
    np.testing.assert_allclose(y1, ref1, rtol=1e-5, atol=1e-5)

    x2 = np.random.randn(7, 8).astype(np.float32)
    y2 = mod.run(x2, w, b)
    ref2 = np.maximum(x2 @ w + b, 0.0)
    np.testing.assert_allclose(y2, ref2, rtol=1e-5, atol=1e-5)
    assert y2.shape == (7, 6)

    print("  PASS: fused symbolic shapes with dynamic batch")


def test_jit_trace_rejects_multi_output():
    np.random.seed(201)

    def pairwise(a, b):
        return a + b, a * b

    x = Tensor.from_numpy(np.random.randn(2, 2).astype(np.float32), name="x")
    y = Tensor.from_numpy(np.random.randn(2, 2).astype(np.float32), name="y")

    try:
        jit.trace(pairwise, (x, y), use_hpc=False)
        raise AssertionError("Expected multi-output trace to be rejected on stable JIT path")
    except RuntimeError as exc:
        assert "Multiple outputs are not supported" in str(exc)


# ===========================================================================

def main():
    print("=== JIT Core Tests ===")
    test_naive_jit()
    test_fusion()
    test_hpc_matmul()
    test_new_ops()
    test_abi_guards()
    test_symbolic_shapes()
    test_fused_symbolic_shapes_dynamic_batch()
    print("All JIT core tests passed.")


if __name__ == "__main__":
    main()

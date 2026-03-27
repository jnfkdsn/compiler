"""Comprehensive test suite for Tensor CPU AI Compiler.

This module contains parameterized tests for all operators,
comparing outputs against NumPy reference implementations.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
import pytest

sys.path.insert(0, ".")

from tensor_cpu import JITEngine, Tensor, TraceContext, optimize_graph


def assert_allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    msg: str = "",
) -> None:
    """Assert two arrays are close with detailed error message."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        if msg:
            raise AssertionError(f"{msg}\n{e}") from e
        raise


class TestBinaryOps:
    """Tests for binary element-wise operations."""

    @pytest.mark.parametrize(
        "op_name,op_func,numpy_func",
        [
            ("add", lambda x, y: x + y, np.add),
            ("sub", lambda x, y: x - y, np.subtract),
            ("mul", lambda x, y: x * y, np.multiply),
            ("div", lambda x, y: x / y, np.divide),
        ],
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (4, 8),
            (16, 32),
            (2, 4, 8),
        ],
    )
    def test_binary_elementwise(
        self,
        op_name: str,
        op_func: Callable,
        numpy_func: Callable,
        shape: tuple[int, ...],
    ) -> None:
        """Test binary element-wise operations against NumPy."""
        np.random.seed(42)
        x = np.random.randn(*shape).astype(np.float32)
        y = np.random.randn(*shape).astype(np.float32)

        if op_name == "div":
            y = np.abs(y) + 0.1  # Avoid division by zero

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            yt = Tensor.from_numpy(y, name="y")
            op_func(xt, yt).mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)
        result = mod.run(x, y)
        expected = numpy_func(x, y)

        assert_allclose(result, expected, msg=f"Failed for {op_name} with shape {shape}")

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            ((4, 8), (8, 6)),
            ((16, 32), (32, 16)),
            ((2, 4, 8), (8, 6)),
        ],
    )
    def test_matmul(self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> None:
        """Test matrix multiplication."""
        np.random.seed(42)
        a = np.random.randn(*shape_a).astype(np.float32)
        b = np.random.randn(*shape_b).astype(np.float32)

        with TraceContext() as tc:
            at = Tensor.from_numpy(a, name="a")
            bt = Tensor.from_numpy(b, name="b")
            (at @ bt).mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)
        result = mod.run(a, b)
        expected = a @ b

        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestUnaryOps:
    """Tests for unary operations."""

    @pytest.mark.parametrize(
        "op_name,op_func,numpy_func",
        [
            ("relu", lambda x: x.relu(), lambda x: np.maximum(x, 0)),
            ("exp", lambda x: x.exp(), np.exp),
            ("log", lambda x: x.log(), np.log),
            ("sigmoid", lambda x: x.sigmoid(), lambda x: 1.0 / (1.0 + np.exp(-x))),
        ],
    )
    @pytest.mark.parametrize("shape", [(4, 8), (16, 32)])
    def test_unary_ops(
        self,
        op_name: str,
        op_func: Callable,
        numpy_func: Callable,
        shape: tuple[int, ...],
    ) -> None:
        """Test unary operations against NumPy."""
        np.random.seed(42)
        x = np.random.randn(*shape).astype(np.float32)

        if op_name == "log":
            x = np.abs(x) + 0.1  # Avoid log of non-positive

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            op_func(xt).mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)
        result = mod.run(x)
        expected = numpy_func(x)

        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestReduceOps:
    """Tests for reduction operations."""

    @pytest.mark.parametrize(
        "op_name,op_kwargs,numpy_func",
        [
            ("sum", {}, np.sum),
            ("mean", {}, np.mean),
            ("max", {}, np.max),
        ],
    )
    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    def test_reduce_ops(
        self,
        op_name: str,
        op_kwargs: dict,
        numpy_func: Callable,
        axis: int,
    ) -> None:
        """Test reduction operations."""
        np.random.seed(42)
        x = np.random.randn(4, 8).astype(np.float32)

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            if axis is None:
                getattr(xt, op_name)().mark_as_output()
            else:
                getattr(xt, op_name)(axis=axis).mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)
        result = mod.run(x)

        if axis is None:
            expected = numpy_func(x)
        else:
            expected = numpy_func(x, axis=axis)

        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


class TestFusion:
    """Tests for operator fusion."""

    def test_matmul_bias_relu_fusion(self) -> None:
        """Test fused MatMul + Bias + ReLU."""
        np.random.seed(42)
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8, 6).astype(np.float32)
        b = np.random.randn(6).astype(np.float32)

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            wt = Tensor.from_numpy(w, name="w")
            bt = Tensor.from_numpy(b, name="b")
            ((xt @ wt) + bt).relu().mark_as_output()
            graph = tc.graph

        stats = optimize_graph(graph)
        assert stats["fused_subgraphs"] >= 1, "Expected at least one fused subgraph"

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)
        result = mod.run(x, w, b)
        expected = np.maximum(x @ w + b, 0)

        assert_allclose(result, expected)


class TestSymbolicShapes:
    """Tests for symbolic shape handling."""

    def test_dynamic_batch_size(self) -> None:
        """Test that compiled kernel works with different batch sizes."""
        np.random.seed(42)
        w = np.random.randn(8, 6).astype(np.float32)
        b = np.random.randn(6).astype(np.float32)

        x_trace = np.random.randn(4, 8).astype(np.float32)
        with TraceContext() as tc:
            xt = Tensor.from_numpy(x_trace, name="x")
            wt = Tensor.from_numpy(w, name="w")
            bt = Tensor.from_numpy(b, name="b")
            ((xt @ wt) + bt).relu().mark_as_output()
            graph = tc.graph

        optimize_graph(graph)
        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)

        for batch_size in [4, 7, 16, 32]:
            x_test = np.random.randn(batch_size, 8).astype(np.float32)
            result = mod.run(x_test, w, b)
            expected = np.maximum(x_test @ w + b, 0)

            assert_allclose(result, expected, msg=f"Failed for batch_size={batch_size}")
            assert result.shape == (batch_size, 6)


class TestABIGuards:
    """Tests for ABI boundary checks."""

    def test_rank_mismatch(self) -> None:
        """Test that rank mismatch raises ValueError."""
        np.random.seed(42)
        x = np.random.randn(4, 8).astype(np.float32)
        y = np.random.randn(4, 8).astype(np.float32)

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            yt = Tensor.from_numpy(y, name="y")
            (xt + yt).mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)

        x_wrong_rank = np.random.randn(32).astype(np.float32)
        with pytest.raises(ValueError, match="rank mismatch"):
            mod.run(x_wrong_rank, y)

    def test_input_count_mismatch(self) -> None:
        """Test that wrong number of inputs raises ValueError."""
        np.random.seed(42)
        x = np.random.randn(4, 8).astype(np.float32)

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            xt.relu().mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)

        with pytest.raises(ValueError, match="Expected"):
            mod.run(x, x)


class TestGradient:
    """Tests for gradient computation."""

    def test_relu_gradient(self) -> None:
        """Test ReLU gradient computation."""
        np.random.seed(42)
        x = np.random.randn(4, 8).astype(np.float32)
        grad = np.random.randn(4, 8).astype(np.float32)

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            grad_t = Tensor.from_numpy(grad, name="grad")
            xt.relu_grad(grad_t).mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)
        result = mod.run(x, grad)
        expected = np.where(x > 0, grad, 0)

        assert_allclose(result, expected)


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_mlp_forward(self) -> None:
        """Test a simple MLP forward pass."""
        np.random.seed(42)

        x = np.random.randn(16, 32).astype(np.float32)
        w1 = np.random.randn(32, 64).astype(np.float32)
        b1 = np.random.randn(64).astype(np.float32)
        w2 = np.random.randn(64, 32).astype(np.float32)
        b2 = np.random.randn(32).astype(np.float32)

        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            w1t = Tensor.from_numpy(w1, name="w1")
            b1t = Tensor.from_numpy(b1, name="b1")
            w2t = Tensor.from_numpy(w2, name="w2")
            b2t = Tensor.from_numpy(b2, name="b2")

            h = (xt @ w1t + b1t).relu()
            (h @ w2t + b2t).mark_as_output()
            graph = tc.graph

        optimize_graph(graph)
        engine = JITEngine(use_hpc_template=False)
        mod = engine.compile_graph(graph)
        result = mod.run(x, w1, b1, w2, b2)

        h_expected = np.maximum(x @ w1 + b1, 0)
        expected = h_expected @ w2 + b2

        assert_allclose(result, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.parametrize(
        "m,n,k",
        [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ],
    )
    def test_matmul_performance(self, benchmark, m: int, n: int, k: int) -> None:
        """Benchmark matrix multiplication performance."""
        np.random.seed(42)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        with TraceContext() as tc:
            at = Tensor.from_numpy(a, name="a")
            bt = Tensor.from_numpy(b, name="b")
            (at @ bt).mark_as_output()
            graph = tc.graph

        engine = JITEngine(use_hpc_template=True)
        mod = engine.compile_graph(graph)

        result = benchmark(mod.run, a, b)
        expected = a @ b

        assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

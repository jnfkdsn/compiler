"""Advanced feature tests: BN/LN/Dropout, Attention, LazyTensor, StaticGraph, jit.trace."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from experimental.lazy import LazyTensor, lazy_mse_loss
from tensor_cpu import (
    StaticGraph,
    Tensor,
    jit,
)
from tensor_cpu.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    MLP,
    ReLU,
    SelfAttention,
    Sequential,
    mse_loss,
)


# ---------------------------------------------------------------------------
# 1. BatchNorm1d train / eval
# ---------------------------------------------------------------------------

def test_batchnorm():
    np.random.seed(20)
    bn = BatchNorm1d(8)
    x = Tensor.from_numpy(np.random.randn(16, 8).astype(np.float32), requires_grad=True, name="x")

    # Training forward
    bn.train()
    out = bn(x)
    assert out.data.shape == (16, 8)
    # Output should be roughly zero-mean, unit-var
    np.testing.assert_allclose(out.data.mean(axis=0), 0.0, atol=0.15)

    # Eval forward (uses running stats)
    bn.eval()
    out_eval = bn(x)
    assert out_eval.data.shape == (16, 8)

    # Backward should work
    bn.train()
    out2 = bn(Tensor.from_numpy(x.data, requires_grad=True, name="x2"))
    loss = mse_loss(out2, Tensor.from_numpy(np.zeros((16, 8), dtype=np.float32), name="t"))
    loss.backward()
    print("  PASS: BatchNorm1d")


# ---------------------------------------------------------------------------
# 2. LayerNorm
# ---------------------------------------------------------------------------

def test_layernorm():
    np.random.seed(21)
    ln = LayerNorm(8)
    x = Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32), requires_grad=True, name="x")
    out = ln(x)
    assert out.data.shape == (4, 8)
    # Each row should be ~zero mean
    row_means = out.data.mean(axis=-1)
    np.testing.assert_allclose(row_means, 0.0, atol=0.15)
    print("  PASS: LayerNorm")


# ---------------------------------------------------------------------------
# 3. Dropout train vs eval
# ---------------------------------------------------------------------------

def test_dropout():
    np.random.seed(22)
    drop = Dropout(p=0.5)
    x_np = np.ones((100, 100), dtype=np.float32)
    x = Tensor.from_numpy(x_np, name="x")

    # Training mode: some elements should be zeroed
    drop.train()
    out_train = drop(x)
    zeros_ratio = np.mean(out_train.data == 0.0)
    assert 0.3 < zeros_ratio < 0.7, f"Dropout zero ratio {zeros_ratio:.2f} outside expected range"

    # Eval mode: identity
    drop.eval()
    out_eval = drop(x)
    np.testing.assert_array_equal(out_eval.data, x_np)
    print("  PASS: Dropout")


# ---------------------------------------------------------------------------
# 4. SelfAttention forward + backward
# ---------------------------------------------------------------------------

def test_self_attention():
    np.random.seed(23)
    d_model = 16
    attn = SelfAttention(d_model=d_model, d_k=8)
    x = Tensor.from_numpy(np.random.randn(5, d_model).astype(np.float32), requires_grad=True, name="x")
    out = attn(x)
    assert out.data.shape == (5, d_model), f"attention output shape {out.data.shape}"

    target = Tensor.from_numpy(np.zeros((5, d_model), dtype=np.float32), name="target")
    loss = mse_loss(out, target)
    loss.backward()
    assert x.grad is not None, "attention backward produced no grad for x"
    print("  PASS: SelfAttention")


# ---------------------------------------------------------------------------
# 5. MLP builder
# ---------------------------------------------------------------------------

def test_mlp():
    np.random.seed(24)
    mlp = MLP(in_features=8, hidden_features=[16, 16], out_features=2, activation="relu")
    x = Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32), name="x")
    out = mlp(x)
    assert out.data.shape == (4, 2), f"MLP output shape {out.data.shape}"
    assert len(mlp.parameters()) > 0
    print("  PASS: MLP")


# ---------------------------------------------------------------------------
# 6. LazyTensor deferred execution
# ---------------------------------------------------------------------------

def test_lazy_tensor():
    np.random.seed(25)
    a_np = np.random.randn(4, 4).astype(np.float32)
    b_np = np.random.randn(4, 4).astype(np.float32)

    a = LazyTensor.from_numpy(a_np, name="a", requires_grad=True)
    b = LazyTensor.from_numpy(b_np, name="b", requires_grad=True)

    # Nothing computed yet
    assert a._cached is None

    c = (a + b) * a - b
    result = c.data  # triggers eval

    ref = (a_np + b_np) * a_np - b_np
    np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)

    # Lazy MSE loss + backward
    pred = LazyTensor.from_numpy(np.random.randn(4, 4).astype(np.float32), name="pred", requires_grad=True)
    tgt = LazyTensor.from_numpy(np.zeros((4, 4), dtype=np.float32), name="tgt")
    loss = lazy_mse_loss(pred, tgt)
    loss.backward()
    assert pred.grad is not None
    print("  PASS: LazyTensor")


# ---------------------------------------------------------------------------
# 7. StaticGraph compile and run
# ---------------------------------------------------------------------------

def test_static_graph():
    np.random.seed(26)
    sg = StaticGraph()
    x = sg.input("x", shape=(4, 8))
    w = sg.input("w", shape=(8, 3))
    b = sg.input("b", shape=(3,))
    out = ((x @ w) + b).relu()
    out.mark_as_output()

    compiled = sg.compile()
    x_np = np.random.randn(4, 8).astype(np.float32)
    w_np = np.random.randn(8, 3).astype(np.float32)
    b_np = np.random.randn(3).astype(np.float32)

    result = compiled.run(x_np, w_np, b_np)
    ref = np.maximum(x_np @ w_np + b_np, 0.0)
    np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-5)

    # Also test named inputs
    result2 = compiled.run(x=x_np, w=w_np, b=b_np)
    np.testing.assert_allclose(result2, ref, rtol=1e-5, atol=1e-5)
    print("  PASS: StaticGraph")


# ---------------------------------------------------------------------------
# 8. jit.trace on function
# ---------------------------------------------------------------------------

def test_jit_trace_function():
    np.random.seed(27)

    def my_func(a, b):
        return (a + b).relu()

    x = Tensor.from_numpy(np.random.randn(4, 4).astype(np.float32), name="x")
    y = Tensor.from_numpy(np.random.randn(4, 4).astype(np.float32), name="y")

    traced = jit.trace(my_func, (x, y), use_hpc=False)
    result = traced(x, y)
    ref = np.maximum(x.data + y.data, 0.0)
    np.testing.assert_allclose(result.data, ref, rtol=1e-5, atol=1e-5)
    print("  PASS: jit.trace function")


# ---------------------------------------------------------------------------
# 9. jit.trace on Module
# ---------------------------------------------------------------------------

def test_jit_trace_module():
    np.random.seed(28)
    model = Sequential(Linear(8, 4), ReLU(), Linear(4, 2))
    x = Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32), name="x")

    eager_out = model(x)
    traced = jit.trace(model, (x,), use_hpc=False)
    jit_out = traced(x)

    np.testing.assert_allclose(jit_out.data, eager_out.data, rtol=1e-4, atol=1e-4)
    print("  PASS: jit.trace module")


# ===========================================================================

def main():
    print("=== Advanced Tests ===")
    test_batchnorm()
    test_layernorm()
    test_dropout()
    test_self_attention()
    test_mlp()
    test_lazy_tensor()
    test_static_graph()
    test_jit_trace_function()
    test_jit_trace_module()
    print("All advanced tests passed.")


if __name__ == "__main__":
    main()

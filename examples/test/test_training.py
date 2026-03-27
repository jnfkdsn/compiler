"""Training tests: eager backward, JITTrainer (SGD/Adam), LazyJITTrainer, full-JIT training step."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from experimental.lazy import lazy_mse_loss
from tensor_cpu import Tensor, TraceContext, compile_sgd_update_kernel, compile_training_step
from tensor_cpu.nn import JITTrainer, LazyJITTrainer, Linear, ReLU, Sequential, mse_loss
from tensor_cpu.optim import SGD, Adam

# ---------------------------------------------------------------------------
# 1. Eager backward correctness (autograd)
# ---------------------------------------------------------------------------


def test_eager_backward():
    np.random.seed(10)
    x = Tensor.from_numpy(np.random.randn(4, 8).astype(np.float32), requires_grad=True, name="x")
    w = Tensor.from_numpy(np.random.randn(8, 3).astype(np.float32), requires_grad=True, name="w")
    b = Tensor.from_numpy(np.random.randn(3).astype(np.float32), requires_grad=True, name="b")

    pred = (x @ w) + b
    target = Tensor.from_numpy(np.zeros((4, 3), dtype=np.float32), name="target")
    loss = mse_loss(pred, target)
    loss.backward()

    assert x.grad is not None and x.grad.shape == (4, 8), "x grad shape mismatch"
    assert w.grad is not None and w.grad.shape == (8, 3), "w grad shape mismatch"
    assert b.grad is not None and b.grad.shape == (3,), "b grad shape mismatch"

    # Numerical check for w: perturb and compare
    eps = 1e-4
    for i in range(min(3, w.data.shape[0])):
        for j in range(min(3, w.data.shape[1])):
            w_plus = w.data.copy()
            w_plus[i, j] += eps
            w_minus = w.data.copy()
            w_minus[i, j] -= eps

            def _loss(w_np):
                p = x.data @ w_np + b.data
                return float(np.mean(p**2))

            num = (_loss(w_plus) - _loss(w_minus)) / (2 * eps)
            np.testing.assert_allclose(w.grad[i, j], num, rtol=5e-2, atol=1e-3)

    print("  PASS: eager backward")


# ---------------------------------------------------------------------------
# 2. Eager training with SGD optimizer
# ---------------------------------------------------------------------------


def test_eager_sgd():
    np.random.seed(11)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    opt = SGD(model.parameters(), lr=0.01)

    x = np.random.randn(16, 4).astype(np.float32)
    y = np.random.randn(16, 2).astype(np.float32)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        pred = model(Tensor.from_numpy(x, name="x"))
        target_t = Tensor.from_numpy(y, name="y")
        loss = mse_loss(pred, target_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert (
        losses[-1] < losses[0]
    ), f"SGD did not converge: first={losses[0]:.4f} last={losses[-1]:.4f}"
    print(f"  PASS: eager SGD (loss {losses[0]:.4f} -> {losses[-1]:.4f})")


# ---------------------------------------------------------------------------
# 3. Eager training with Adam optimizer
# ---------------------------------------------------------------------------


def test_eager_adam():
    np.random.seed(12)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    opt = Adam(model.parameters(), lr=0.01)

    x = np.random.randn(16, 4).astype(np.float32)
    y = np.random.randn(16, 2).astype(np.float32)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        pred = model(Tensor.from_numpy(x, name="x"))
        target_t = Tensor.from_numpy(y, name="y")
        loss = mse_loss(pred, target_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert (
        losses[-1] < losses[0]
    ), f"Adam did not converge: first={losses[0]:.4f} last={losses[-1]:.4f}"
    print(f"  PASS: eager Adam (loss {losses[0]:.4f} -> {losses[-1]:.4f})")


# ---------------------------------------------------------------------------
# 4. JITTrainer with SGD
# ---------------------------------------------------------------------------


def test_jit_trainer_sgd():
    np.random.seed(13)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    trainer = JITTrainer(model, mse_loss, optimizer="sgd", lr=0.01)

    x = np.random.randn(16, 4).astype(np.float32)
    y = np.random.randn(16, 2).astype(np.float32)

    losses = []
    for _ in range(20):
        l = trainer.step(x, y)
        losses.append(l)

    assert (
        losses[-1] < losses[0]
    ), f"JITTrainer SGD did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  PASS: JITTrainer SGD (loss {losses[0]:.4f} -> {losses[-1]:.4f})")


# ---------------------------------------------------------------------------
# 5. JITTrainer with Adam
# ---------------------------------------------------------------------------


def test_jit_trainer_adam():
    np.random.seed(14)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    trainer = JITTrainer(model, mse_loss, optimizer="adam", lr=0.01)

    x = np.random.randn(16, 4).astype(np.float32)
    y = np.random.randn(16, 2).astype(np.float32)

    losses = []
    for _ in range(20):
        l = trainer.step(x, y)
        losses.append(l)

    assert (
        losses[-1] < losses[0]
    ), f"JITTrainer Adam did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  PASS: JITTrainer Adam (loss {losses[0]:.4f} -> {losses[-1]:.4f})")


# ---------------------------------------------------------------------------
# 6. LazyJITTrainer
# ---------------------------------------------------------------------------


def test_lazy_jit_trainer():
    np.random.seed(15)
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    trainer = LazyJITTrainer(model, lazy_loss_fn=lazy_mse_loss, optimizer="sgd", lr=0.01)

    x = np.random.randn(16, 4).astype(np.float32)
    y = np.random.randn(16, 2).astype(np.float32)

    losses = []
    for _ in range(20):
        l = trainer.step(x, y)
        losses.append(l)

    assert (
        losses[-1] < losses[0]
    ), f"LazyJITTrainer did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  PASS: LazyJITTrainer (loss {losses[0]:.4f} -> {losses[-1]:.4f})")


# ---------------------------------------------------------------------------
# 7. Full-JIT compile_training_step
# ---------------------------------------------------------------------------


def test_full_jit_training_step():
    np.random.seed(16)
    in_dim, out_dim = 4, 2
    x_np = np.random.randn(8, in_dim).astype(np.float32)
    y_np = np.random.randn(8, out_dim).astype(np.float32)
    w_np = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.1
    b_np = np.zeros((out_dim,), dtype=np.float32)

    with TraceContext() as tc:
        x = Tensor.from_numpy(x_np, name="x")
        y = Tensor.from_numpy(y_np, name="y")
        w = Tensor.from_numpy(w_np, name="w", requires_grad=True)
        b = Tensor.from_numpy(b_np, name="b", requires_grad=True)
        pred = (x @ w) + b
        loss = mse_loss(pred, y)
        loss.mark_as_output()
        graph = tc.graph

    w_node_id = w.node.id
    b_node_id = b.node.id

    step = compile_training_step(graph, param_input_ids=[w_node_id, b_node_id])
    update_w = compile_sgd_update_kernel(w_np.shape, lr=0.1)
    update_b = compile_sgd_update_kernel(b_np.shape, lr=0.1)

    losses = []
    for _ in range(30):
        l, grads = step.run_loss_and_grads(x_np, y_np, w_np, b_np)
        w_np = update_w.run(w_np, grads[w_node_id])
        b_np = update_b.run(b_np, grads[b_node_id])
        losses.append(l)

    assert (
        losses[-1] < losses[0]
    ), f"Full JIT training did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  PASS: full-JIT training step (loss {losses[0]:.4f} -> {losses[-1]:.4f})")


def test_compile_training_step_packs_gradient_outputs():
    np.random.seed(160)
    x_np = np.random.randn(4, 3).astype(np.float32)
    y_np = np.random.randn(4, 2).astype(np.float32)
    w_np = np.random.randn(3, 2).astype(np.float32)
    b_np = np.random.randn(2).astype(np.float32)

    with TraceContext() as tc:
        x = Tensor.from_numpy(x_np, name="x")
        y = Tensor.from_numpy(y_np, name="y")
        w = Tensor.from_numpy(w_np, name="w", requires_grad=True)
        b = Tensor.from_numpy(b_np, name="b", requires_grad=True)
        loss = mse_loss((x @ w) + b, y).mark_as_output()
        _ = loss
        graph = tc.graph

    step = compile_training_step(graph, param_input_ids=[w.node.id, b.node.id])
    assert step.grad_module.output_shape == (w_np.size + b_np.size,)
    assert len(step.grad_layout) == 2

    _, grads = step.run_loss_and_grads(x_np, y_np, w_np, b_np)
    assert grads[w.node.id].shape == w_np.shape
    assert grads[b.node.id].shape == b_np.shape


# ---------------------------------------------------------------------------
# 8. Graph-mode reduce gradients (mean/max) vs numerical gradients
# ---------------------------------------------------------------------------


def _numeric_grad_scalar_fn(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float32)
    it = np.ndindex(x.shape)
    for idx in it:
        x_p = x.copy()
        x_m = x.copy()
        x_p[idx] += eps
        x_m[idx] -= eps
        grad[idx] = (fn(x_p) - fn(x_m)) / (2.0 * eps)
    return grad


def test_graph_reduce_gradients_numerical():
    np.random.seed(17)

    x_mean = np.random.randn(3, 4).astype(np.float32)
    with TraceContext() as tc_mean:
        x = Tensor.from_numpy(x_mean, name="x", requires_grad=True)
        loss = x.mean(axis=1).mean().mark_as_output()
        _ = loss
        g_mean = tc_mean.graph
    x_mean_id = next(n.id for n in g_mean.topological_sort() if n.name == "x")
    step_mean = compile_training_step(g_mean, param_input_ids=[x_mean_id])
    _, grads_mean = step_mean.run_loss_and_grads(x_mean)
    grad_mean = grads_mean[x_mean_id]

    def fn_mean(arr: np.ndarray) -> float:
        return float(arr.mean(axis=1).mean())

    num_mean = _numeric_grad_scalar_fn(fn_mean, x_mean)
    np.testing.assert_allclose(grad_mean, num_mean, rtol=1e-3, atol=2e-3)

    x_max = np.random.randn(3, 5).astype(np.float32)
    # Avoid ties to keep the subgradient well-defined for finite differences.
    x_max += np.arange(x_max.size, dtype=np.float32).reshape(x_max.shape) * 1e-5
    with TraceContext() as tc_max:
        x = Tensor.from_numpy(x_max, name="x", requires_grad=True)
        loss = x.max(axis=1).mean().mark_as_output()
        _ = loss
        g_max = tc_max.graph
    x_max_id = next(n.id for n in g_max.topological_sort() if n.name == "x")
    step_max = compile_training_step(g_max, param_input_ids=[x_max_id])
    _, grads_max = step_max.run_loss_and_grads(x_max)
    grad_max = grads_max[x_max_id]

    def fn_max(arr: np.ndarray) -> float:
        return float(arr.max(axis=1).mean())

    num_max = _numeric_grad_scalar_fn(fn_max, x_max)
    np.testing.assert_allclose(grad_max, num_max, rtol=2e-2, atol=2e-2)
    print("  PASS: graph reduce gradients (mean/max) vs numerical")


# ===========================================================================


def main():
    print("=== Training Tests ===")
    test_eager_backward()
    test_eager_sgd()
    test_eager_adam()
    test_jit_trainer_sgd()
    test_jit_trainer_adam()
    test_lazy_jit_trainer()
    test_full_jit_training_step()
    test_graph_reduce_gradients_numerical()
    print("All training tests passed.")


if __name__ == "__main__":
    main()

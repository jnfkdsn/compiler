"""Graph-level training JIT utilities: backward graph build, optimizer kernels, and step runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .graph import Graph, Node
from .ops import OpType
from .runtime import JITEngine, JITModule
from .vjp import add_const_scalar, add_binary, apply_vjp, broadcast_to


def build_backward_graph(forward_graph: Graph, *, wrt_input_ids: Iterable[int]) -> Dict[int, Graph]:
    """Build backward graphs for selected forward INPUT node ids.

    Returns a mapping: input_node_id -> gradient graph with same forward inputs and one grad output.
    Uses the unified VJP rules from ``vjp.py``.
    """
    ordered = forward_graph.topological_sort()
    if not ordered:
        raise ValueError("Forward graph is empty.")

    if forward_graph.output_ids:
        loss_id = forward_graph.output_ids[-1]
        loss_node = forward_graph.get_node(loss_id)
    else:
        loss_node = ordered[-1]
    if loss_node.shape != ():
        raise ValueError(f"Backward builder currently expects scalar loss output, got shape {loss_node.shape}")

    out: Dict[int, Graph] = {}
    wrt_set = set(int(v) for v in wrt_input_ids)

    for wrt_id in wrt_set:
        graph = Graph()
        old_to_new: Dict[int, Node] = {}
        grads: Dict[int, Node] = {}

        # Clone forward graph first.
        for n in ordered:
            cloned = graph.add_node(
                op_type=n.op_type,
                name=f"fwd_{n.name}",
                inputs=[old_to_new[i].id for i in n.inputs],
                shape=n.shape,
                dtype=n.dtype,
                attrs=dict(n.attrs),
            )
            old_to_new[n.id] = cloned

        loss_new = old_to_new[loss_node.id]
        seed = add_const_scalar(graph, 1.0, dtype=loss_new.dtype)
        grads[loss_new.id] = seed

        # Reverse-mode differentiation using unified VJP rules.
        for n in reversed(ordered):
            n_new = old_to_new[n.id]
            gout = grads.get(n_new.id)
            if gout is None:
                continue

            if n.op_type in (OpType.INPUT, OpType.CONST):
                continue

            input_nodes = [old_to_new[i] for i in n.inputs]
            input_grads = apply_vjp(graph, n.op_type, gout, input_nodes, n_new, n.attrs)

            for inp_old_id, g in zip(n.inputs, input_grads):
                if g is None:
                    continue
                tgt = old_to_new[inp_old_id]
                prev = grads.get(tgt.id)
                if prev is None:
                    grads[tgt.id] = g
                else:
                    grads[tgt.id] = add_binary(graph, OpType.ADD, prev, g, f"grad_acc_{tgt.id}")

        wrt_old = forward_graph.get_node(wrt_id)
        wrt_new = old_to_new[wrt_old.id]
        grad_out = grads.get(wrt_new.id)
        if grad_out is None:
            # Disconnected parameter/input => zero gradient.
            z = add_const_scalar(graph, 0.0, dtype=wrt_new.dtype)
            grad_out = broadcast_to(graph, z, wrt_new.shape, f"zero_grad_{wrt_id}")

        graph.mark_output(grad_out.id)
        out[wrt_id] = graph

    return out


@dataclass
class JITSGDUpdateKernel:
    module: JITModule

    def run(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return self.module.run(param, grad)


def compile_sgd_update_kernel(shape: tuple[int, ...], lr: float, weight_decay: float = 0.0) -> JITSGDUpdateKernel:
    from .tensor import Tensor
    from .tracer import TraceContext

    p0 = np.zeros(shape, dtype=np.float32)
    g0 = np.zeros(shape, dtype=np.float32)
    with TraceContext() as tc:
        p = Tensor.from_numpy(p0, name="param")
        g = Tensor.from_numpy(g0, name="grad")
        if weight_decay != 0.0:
            g = g + float(weight_decay) * p
        out = (p - float(lr) * g).mark_as_output()
        _ = out
        graph = tc.graph
    mod = JITEngine(use_hpc_template=False).compile_graph(graph)
    return JITSGDUpdateKernel(module=mod)


@dataclass
class JITAdamUpdateKernels:
    m_kernel: JITModule
    v_kernel: JITModule
    p_kernel: JITModule
    beta1: float
    beta2: float
    eps: float

    def run(self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, step: int, lr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m_new = self.m_kernel.run(m, grad)
        v_new = self.v_kernel.run(v, grad)
        b1t = self.beta1 ** float(step)
        b2t = self.beta2 ** float(step)
        lr_t = float(lr) * (np.sqrt(1.0 - b2t) / (1.0 - b1t))
        lr_arr = np.asarray(lr_t, dtype=np.float32)
        p_new = self.p_kernel.run(param, m_new, v_new, lr_arr)
        return p_new, m_new, v_new


def compile_adam_update_kernels(shape: tuple[int, ...], *, beta1: float, beta2: float, eps: float) -> JITAdamUpdateKernels:
    from .tensor import Tensor
    from .tracer import TraceContext

    z = np.zeros(shape, dtype=np.float32)

    with TraceContext() as tc_m:
        m = Tensor.from_numpy(z, name="m")
        g = Tensor.from_numpy(z, name="g")
        m_out = (float(beta1) * m + (1.0 - float(beta1)) * g).mark_as_output()
        _ = m_out
        g_m = tc_m.graph
    m_kernel = JITEngine(use_hpc_template=False).compile_graph(g_m)

    with TraceContext() as tc_v:
        v = Tensor.from_numpy(z, name="v")
        g = Tensor.from_numpy(z, name="g")
        v_out = (float(beta2) * v + (1.0 - float(beta2)) * (g * g)).mark_as_output()
        _ = v_out
        g_v = tc_v.graph
    v_kernel = JITEngine(use_hpc_template=False).compile_graph(g_v)

    with TraceContext() as tc_p:
        p = Tensor.from_numpy(z, name="p")
        m = Tensor.from_numpy(z, name="m")
        v = Tensor.from_numpy(z, name="v")
        lr_t = Tensor.from_numpy(np.asarray(0.0, dtype=np.float32), name="lr_t")
        denom = ((v + float(eps)).log() * 0.5).exp()
        p_out = (p - lr_t * (m / (denom + float(eps)))).mark_as_output()
        _ = p_out
        g_p = tc_p.graph
    p_kernel = JITEngine(use_hpc_template=False).compile_graph(g_p)

    return JITAdamUpdateKernels(
        m_kernel=m_kernel,
        v_kernel=v_kernel,
        p_kernel=p_kernel,
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(eps),
    )


@dataclass
class CompiledTrainingStep:
    loss_module: JITModule
    grad_modules: Dict[int, JITModule]
    param_input_ids: List[int]

    def run_loss_and_grads(self, *inputs: np.ndarray) -> tuple[float, Dict[int, np.ndarray]]:
        loss = float(self.loss_module.run(*inputs))
        grads = {pid: mod.run(*inputs) for pid, mod in self.grad_modules.items()}
        return loss, grads


def compile_training_step(forward_graph: Graph, *, param_input_ids: Iterable[int], use_hpc_template: bool = False) -> CompiledTrainingStep:
    engine = JITEngine(use_hpc_template=use_hpc_template)
    loss_module = engine.compile_graph(forward_graph)

    backward_graphs = build_backward_graph(forward_graph, wrt_input_ids=param_input_ids)
    grad_modules: Dict[int, JITModule] = {}
    for pid, g in backward_graphs.items():
        grad_modules[pid] = engine.compile_graph(g)

    return CompiledTrainingStep(
        loss_module=loss_module,
        grad_modules=grad_modules,
        param_input_ids=list(param_input_ids),
    )

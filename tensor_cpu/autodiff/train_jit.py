"""Graph-level training JIT utilities: backward graph build, optimizer kernels, and step runner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..autodiff.vjp import add_binary, add_const_scalar, apply_vjp, broadcast_to
from ..ir.graph import Graph, Node
from ..ir.ops import OpType
from ..runtime import JITEngine, JITModule


@dataclass(frozen=True)
class GradLayoutEntry:
    input_id: int
    shape: tuple[int, ...]
    start: int
    end: int


def _build_backward_state(
    forward_graph: Graph,
) -> tuple[list[Node], Node, Graph, dict[int, Node], dict[int, Node]]:
    ordered = forward_graph.topological_sort()
    if not ordered:
        raise ValueError("Forward graph is empty.")

    if forward_graph.output_ids:
        loss_id = forward_graph.output_ids[-1]
        loss_node = forward_graph.get_node(loss_id)
    else:
        loss_node = ordered[-1]
    if loss_node.shape != ():
        raise ValueError(
            f"Backward builder currently expects scalar loss output, got shape {loss_node.shape}"
        )

    graph = Graph()
    old_to_new: dict[int, Node] = {}
    grads: dict[int, Node] = {}

    for node in ordered:
        cloned = graph.add_node(
            op_type=node.op_type,
            name=f"fwd_{node.name}",
            inputs=[old_to_new[i].id for i in node.inputs],
            shape=node.shape,
            dtype=node.dtype,
            attrs=dict(node.attrs),
        )
        old_to_new[node.id] = cloned

    loss_new = old_to_new[loss_node.id]
    grads[loss_new.id] = add_const_scalar(graph, 1.0, dtype=loss_new.dtype)

    for node in reversed(ordered):
        node_new = old_to_new[node.id]
        gout = grads.get(node_new.id)
        if gout is None:
            continue
        if node.op_type in (OpType.INPUT, OpType.CONST):
            continue

        input_nodes = [old_to_new[i] for i in node.inputs]
        input_grads = apply_vjp(graph, node.op_type, gout, input_nodes, node_new, node.attrs)
        for inp_old_id, grad in zip(node.inputs, input_grads):
            if grad is None:
                continue
            target = old_to_new[inp_old_id]
            prev = grads.get(target.id)
            if prev is None:
                grads[target.id] = grad
            else:
                grads[target.id] = add_binary(
                    graph, OpType.ADD, prev, grad, f"grad_acc_{target.id}"
                )

    return ordered, loss_node, graph, old_to_new, grads


def build_backward_graph(forward_graph: Graph, *, wrt_input_ids: Iterable[int]) -> dict[int, Graph]:
    """Build per-target backward graphs for selected forward INPUT node ids."""

    _, _, shared_graph, old_to_new, grads = _build_backward_state(forward_graph)
    out: dict[int, Graph] = {}
    for wrt_id in {int(v) for v in wrt_input_ids}:
        graph = Graph()
        node_map: dict[int, Node] = {}
        for node in shared_graph.topological_sort():
            cloned = graph.add_node(
                op_type=node.op_type,
                name=node.name,
                inputs=[node_map[i].id for i in node.inputs],
                shape=node.shape,
                dtype=node.dtype,
                attrs=dict(node.attrs),
            )
            node_map[node.id] = cloned

        wrt_old = forward_graph.get_node(wrt_id)
        wrt_new = old_to_new[wrt_old.id]
        grad_out = grads.get(wrt_new.id)
        if grad_out is None:
            z = add_const_scalar(graph, 0.0, dtype=wrt_new.dtype)
            grad_out = broadcast_to(graph, z, wrt_new.shape, f"zero_grad_{wrt_id}")
        else:
            grad_out = node_map[grad_out.id]

        graph.mark_output(grad_out.id)
        out[wrt_id] = graph

    return out


def build_joint_backward_graph(
    forward_graph: Graph,
    *,
    wrt_input_ids: Iterable[int],
) -> tuple[Graph, list[GradLayoutEntry]]:
    """Build a shared backward graph and pack all requested gradients into one output tensor."""

    _, _, graph, old_to_new, grads = _build_backward_state(forward_graph)

    layout: list[GradLayoutEntry] = []
    packed_inputs: list[int] = []
    offset = 0
    target_ids = [int(v) for v in wrt_input_ids]
    packed_dtype = "float32"
    for wrt_id in target_ids:
        wrt_old = forward_graph.get_node(wrt_id)
        wrt_new = old_to_new[wrt_old.id]
        if wrt_new.dtype == "float64":
            packed_dtype = "float64"
        grad_out = grads.get(wrt_new.id)
        if grad_out is None:
            zero = add_const_scalar(graph, 0.0, dtype=wrt_new.dtype)
            grad_out = broadcast_to(graph, zero, wrt_new.shape, f"zero_grad_{wrt_id}")

        numel = int(wrt_new.numel)
        packed_inputs.append(grad_out.id)
        layout.append(
            GradLayoutEntry(
                input_id=wrt_id,
                shape=tuple(wrt_new.shape),
                start=offset,
                end=offset + numel,
            )
        )
        offset += numel

    packed = graph.add_node(
        op_type=OpType.PACK,
        name="packed_grads",
        inputs=packed_inputs,
        shape=(offset,),
        dtype=packed_dtype,
        attrs={},
    )
    graph.mark_output(packed.id)
    return graph, layout


@dataclass
class JITSGDUpdateKernel:
    module: JITModule

    def run(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return self.module.run(param, grad)


def compile_sgd_update_kernel(
    shape: tuple[int, ...], lr: float, weight_decay: float = 0.0
) -> JITSGDUpdateKernel:
    from ..frontend.tracer import TraceContext
    from ..tensor import Tensor

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

    def run(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        step: int,
        lr: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m_new = self.m_kernel.run(m, grad)
        v_new = self.v_kernel.run(v, grad)
        b1t = self.beta1 ** float(step)
        b2t = self.beta2 ** float(step)
        lr_t = float(lr) * (np.sqrt(1.0 - b2t) / (1.0 - b1t))
        lr_arr = np.asarray(lr_t, dtype=np.float32)
        p_new = self.p_kernel.run(param, m_new, v_new, lr_arr)
        return p_new, m_new, v_new


def compile_adam_update_kernels(
    shape: tuple[int, ...],
    *,
    beta1: float,
    beta2: float,
    eps: float,
) -> JITAdamUpdateKernels:
    from ..frontend.tracer import TraceContext
    from ..tensor import Tensor

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
    grad_module: JITModule
    grad_layout: list[GradLayoutEntry]
    param_input_ids: list[int]

    def run_loss_and_grads(self, *inputs: np.ndarray) -> tuple[float, dict[int, np.ndarray]]:
        loss = float(self.loss_module.run(*inputs))
        packed = np.asarray(self.grad_module.run(*inputs), dtype=np.float32).reshape(-1)
        grads: dict[int, np.ndarray] = {}
        for entry in self.grad_layout:
            view = packed[entry.start : entry.end]
            grads[entry.input_id] = view.reshape(entry.shape).copy()
        return loss, grads


def compile_training_step(
    forward_graph: Graph,
    *,
    param_input_ids: Iterable[int],
    use_hpc_template: bool = False,
) -> CompiledTrainingStep:
    engine = JITEngine(use_hpc_template=use_hpc_template)
    loss_module = engine.compile_graph(forward_graph)

    grad_graph, grad_layout = build_joint_backward_graph(
        forward_graph,
        wrt_input_ids=param_input_ids,
    )
    grad_module = engine.compile_graph(grad_graph)

    return CompiledTrainingStep(
        loss_module=loss_module,
        grad_module=grad_module,
        grad_layout=grad_layout,
        param_input_ids=[entry.input_id for entry in grad_layout],
    )

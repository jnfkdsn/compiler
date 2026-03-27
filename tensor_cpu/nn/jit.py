"""High-level nn auto-JIT trainer that compiles loss/grad/update on first step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, TYPE_CHECKING

import numpy as np

from ..autodiff.train_jit import compile_adam_update_kernels, compile_sgd_update_kernel, compile_training_step
from ..frontend.tracer import TraceContext
from ..tensor import Tensor
from .modules import Module

if TYPE_CHECKING:
    from experimental.lazy import LazyTensor


@dataclass
class _TensorSlot:
    owner: object
    key: object
    is_index: bool

    def get(self) -> Tensor:
        if self.is_index:
            return self.owner[self.key]  # type: ignore[index]
        return getattr(self.owner, self.key)  # type: ignore[arg-type]

    def set(self, value: Tensor) -> None:
        if self.is_index:
            self.owner[self.key] = value  # type: ignore[index]
        else:
            setattr(self.owner, self.key, value)  # type: ignore[arg-type]


def _collect_parameter_slots(module: Module) -> List[_TensorSlot]:
    slots: List[_TensorSlot] = []

    def walk(obj: object) -> None:
        if isinstance(obj, Module):
            for k, v in obj.__dict__.items():
                if isinstance(v, Tensor) and v.requires_grad:
                    slots.append(_TensorSlot(owner=obj, key=k, is_index=False))
                elif isinstance(v, Module):
                    walk(v)
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, Tensor) and item.requires_grad:
                            slots.append(_TensorSlot(owner=v, key=i, is_index=True))
                        elif isinstance(item, Module):
                            walk(item)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, Module):
                    walk(item)

    walk(module)
    return slots


class JITTrainer:
    """Auto-compiling trainer for nn.Module.

    First `step(...)` traces and compiles:
    1) forward loss graph
    2) backward grad graphs for parameters
    3) optimizer update kernels

    Later steps reuse compiled kernels.
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        *,
        optimizer: str = "sgd",
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        use_hpc_template: bool = False,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer.lower()
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.use_hpc_template = bool(use_hpc_template)

        self._compiled = None
        self._param_slots: List[_TensorSlot] = []
        self._param_input_ids: List[int] = []
        self._x_shape: tuple[int, ...] | None = None
        self._y_shape: tuple[int, ...] | None = None
        self._sgd_kernels = []
        self._adam_kernels = []
        self._adam_m: List[np.ndarray] = []
        self._adam_v: List[np.ndarray] = []
        self._adam_step = 0

    def _compile(self, x: np.ndarray, y: np.ndarray) -> None:
        self._x_shape = tuple(x.shape)
        self._y_shape = tuple(y.shape)

        slots = _collect_parameter_slots(self.model)
        self._param_slots = slots

        originals = [slot.get() for slot in slots]
        traced_params: List[Tensor] = []

        with TraceContext() as tc:
            tx = Tensor.from_numpy(x, name="input_x")
            ty = Tensor.from_numpy(y, name="input_y")

            for i, (slot, p) in enumerate(zip(slots, originals)):
                tp = Tensor.from_numpy(p.data, name=f"param_{i}", requires_grad=True)
                slot.set(tp)
                traced_params.append(tp)

            try:
                pred = self.model(tx)
                loss = self.loss_fn(pred, ty)
                loss = loss.mark_as_output()
                _ = loss
                graph = tc.graph
            finally:
                for slot, orig in zip(slots, originals):
                    slot.set(orig)

        param_ids = [tp.node.id for tp in traced_params if tp.node is not None]
        self._param_input_ids = list(param_ids)

        self._compiled = compile_training_step(
            graph,
            param_input_ids=self._param_input_ids,
            use_hpc_template=self.use_hpc_template,
        )

        if self.optimizer == "sgd":
            self._sgd_kernels = [
                compile_sgd_update_kernel(shape=slot.get().data.shape, lr=self.lr, weight_decay=self.weight_decay)
                for slot in self._param_slots
            ]
        elif self.optimizer == "adam":
            self._adam_kernels = [
                compile_adam_update_kernels(
                    shape=slot.get().data.shape,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                )
                for slot in self._param_slots
            ]
            self._adam_m = [np.zeros_like(slot.get().data, dtype=np.float32) for slot in self._param_slots]
            self._adam_v = [np.zeros_like(slot.get().data, dtype=np.float32) for slot in self._param_slots]
            self._adam_step = 0
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if self._compiled is None:
            self._compile(x_arr, y_arr)
        else:
            if tuple(x_arr.shape) != self._x_shape or tuple(y_arr.shape) != self._y_shape:
                raise ValueError(
                    f"Static-shape compiled trainer expects x{self._x_shape}, y{self._y_shape}; "
                    f"got x{tuple(x_arr.shape)}, y{tuple(y_arr.shape)}"
                )

        assert self._compiled is not None

        inputs = [x_arr, y_arr]
        for slot in self._param_slots:
            inputs.append(np.asarray(slot.get().data, dtype=np.float32, order="C"))

        loss, grads = self._compiled.run_loss_and_grads(*inputs)

        if self.optimizer == "sgd":
            for i, slot in enumerate(self._param_slots):
                p = slot.get()
                pid = self._param_input_ids[i]
                g = grads[pid]
                p.data = self._sgd_kernels[i].run(p.data, g)
        else:
            self._adam_step += 1
            for i, slot in enumerate(self._param_slots):
                p = slot.get()
                pid = self._param_input_ids[i]
                g = grads[pid]
                p_new, m_new, v_new = self._adam_kernels[i].run(
                    p.data,
                    g,
                    self._adam_m[i],
                    self._adam_v[i],
                    step=self._adam_step,
                    lr=self.lr,
                )
                p.data = p_new
                self._adam_m[i] = m_new
                self._adam_v[i] = v_new

        return float(loss)


@dataclass
class _CompiledCacheEntry:
    compiled: object
    param_input_ids: List[int]
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]


class LazyJITTrainer:
    """Unified lazy-graph + auto-compile + cached execution trainer.

    Behavior:
    1) Build loss graph lazily using LazyTensor wrappers.
    2) First step for a new (x_shape, y_shape) triggers trace+compile.
    3) Later steps with same shape reuse cached compiled kernels.
    """

    def __init__(
        self,
        model: Module,
        *,
        lazy_loss_fn: Callable,
        optimizer: str = "sgd",
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        use_hpc_template: bool = False,
    ) -> None:
        self.model = model
        self.lazy_loss_fn = lazy_loss_fn
        self.optimizer = optimizer.lower()
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.use_hpc_template = bool(use_hpc_template)

        self._param_slots: List[_TensorSlot] = _collect_parameter_slots(model)
        self._compile_cache: Dict[tuple[int, int], _CompiledCacheEntry] = {}

        if self.optimizer == "sgd":
            self._sgd_kernels = [
                compile_sgd_update_kernel(shape=slot.get().data.shape, lr=self.lr, weight_decay=self.weight_decay)
                for slot in self._param_slots
            ]
            self._adam_kernels = []
            self._adam_m = []
            self._adam_v = []
            self._adam_step = 0
        elif self.optimizer == "adam":
            self._sgd_kernels = []
            self._adam_kernels = [
                compile_adam_update_kernels(
                    shape=slot.get().data.shape,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.eps,
                )
                for slot in self._param_slots
            ]
            self._adam_m = [np.zeros_like(slot.get().data, dtype=np.float32) for slot in self._param_slots]
            self._adam_v = [np.zeros_like(slot.get().data, dtype=np.float32) for slot in self._param_slots]
            self._adam_step = 0
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def _compile_for_shape(self, x: np.ndarray, y: np.ndarray) -> _CompiledCacheEntry:
        from experimental.lazy import LazyTensor

        x_shape = tuple(x.shape)
        y_shape = tuple(y.shape)

        originals = [slot.get() for slot in self._param_slots]
        traced_params: List[Tensor] = []

        with TraceContext() as tc:
            tx = Tensor.from_numpy(x, name="input_x")
            ty = Tensor.from_numpy(y, name="input_y")

            for i, (slot, p) in enumerate(zip(self._param_slots, originals)):
                tp = Tensor.from_numpy(p.data, name=f"param_{i}", requires_grad=True)
                slot.set(tp)
                traced_params.append(tp)

            try:
                pred = self.model(tx)
                lazy_pred = LazyTensor(lambda: pred)
                lazy_target = LazyTensor(lambda: ty)
                loss = self.lazy_loss_fn(lazy_pred, lazy_target).materialize().mark_as_output()
                _ = loss
                graph = tc.graph
            finally:
                for slot, orig in zip(self._param_slots, originals):
                    slot.set(orig)

        param_ids = [tp.node.id for tp in traced_params if tp.node is not None]
        compiled = compile_training_step(
            graph,
            param_input_ids=param_ids,
            use_hpc_template=self.use_hpc_template,
        )
        return _CompiledCacheEntry(
            compiled=compiled,
            param_input_ids=list(param_ids),
            x_shape=x_shape,
            y_shape=y_shape,
        )

    def step(self, x: np.ndarray, y: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        key = (x_arr.ndim, y_arr.ndim)

        entry = self._compile_cache.get(key)
        if entry is None:
            entry = self._compile_for_shape(x_arr, y_arr)
            self._compile_cache[key] = entry

        inputs = [x_arr, y_arr]
        for slot in self._param_slots:
            inputs.append(np.asarray(slot.get().data, dtype=np.float32, order="C"))

        loss, grads = entry.compiled.run_loss_and_grads(*inputs)

        if self.optimizer == "sgd":
            for i, slot in enumerate(self._param_slots):
                p = slot.get()
                pid = entry.param_input_ids[i]
                p.data = self._sgd_kernels[i].run(p.data, grads[pid])
        else:
            self._adam_step += 1
            for i, slot in enumerate(self._param_slots):
                p = slot.get()
                pid = entry.param_input_ids[i]
                p_new, m_new, v_new = self._adam_kernels[i].run(
                    p.data,
                    grads[pid],
                    self._adam_m[i],
                    self._adam_v[i],
                    step=self._adam_step,
                    lr=self.lr,
                )
                p.data = p_new
                self._adam_m[i] = m_new
                self._adam_v[i] = v_new

        return float(loss)

    def cache_size(self) -> int:
        return len(self._compile_cache)

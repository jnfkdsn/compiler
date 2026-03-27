"""Lazy evaluation layer built on top of eager Tensor.

LazyTensor defers all computation until `eval()`/`data`/`backward` is called.
This keeps existing eager/JIT code unchanged while enabling full deferred execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from tensor_cpu.tensor import Tensor


@dataclass
class _Thunk:
    fn: Callable[[], Tensor]


class LazyTensor:
    """Deferred tensor expression.

    Each operation creates a new LazyTensor by composing thunks without running
    numerical kernels immediately. The underlying eager Tensor is materialized
    only when `eval()` or `data` is accessed.
    """

    def __init__(self, thunk: Callable[[], Tensor]) -> None:
        self._thunk = _Thunk(thunk)
        self._cached: Optional[Tensor] = None

    @staticmethod
    def from_numpy(
        array: np.ndarray, *, name: str | None = None, requires_grad: bool = False
    ) -> "LazyTensor":
        arr = np.asarray(array, dtype=np.float32)
        return LazyTensor(lambda: Tensor.from_numpy(arr, name=name, requires_grad=requires_grad))

    @staticmethod
    def scalar(value: float, *, requires_grad: bool = False) -> "LazyTensor":
        return LazyTensor(lambda: Tensor.scalar(float(value), requires_grad=requires_grad))

    @staticmethod
    def _ensure_lazy(value: "LazyTensor | Tensor | float") -> "LazyTensor":
        if isinstance(value, LazyTensor):
            return value
        if isinstance(value, Tensor):
            return LazyTensor(lambda: value)
        return LazyTensor.scalar(float(value), requires_grad=False)

    def eval(self) -> Tensor:
        if self._cached is None:
            self._cached = self._thunk.fn()
        return self._cached

    @property
    def data(self) -> np.ndarray:
        out = self.eval().data
        if out is None:
            raise RuntimeError("LazyTensor evaluation produced no data.")
        return out

    @property
    def grad(self) -> Optional[np.ndarray]:
        return self.eval().grad

    @property
    def shape(self) -> tuple[int, ...]:
        return self.eval().shape

    @property
    def dtype(self) -> str:
        return self.eval().dtype

    def backward(self) -> None:
        self.eval().backward()

    def zero_grad(self) -> None:
        self.eval().zero_grad()

    def materialize(self) -> Tensor:
        return self.eval()

    def __add__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        rhs = LazyTensor._ensure_lazy(other)
        return LazyTensor(lambda: self.eval() + rhs.eval())

    def __radd__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        return self.__add__(other)

    def __sub__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        rhs = LazyTensor._ensure_lazy(other)
        return LazyTensor(lambda: self.eval() - rhs.eval())

    def __rsub__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        lhs = LazyTensor._ensure_lazy(other)
        return LazyTensor(lambda: lhs.eval() - self.eval())

    def __mul__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        rhs = LazyTensor._ensure_lazy(other)
        return LazyTensor(lambda: self.eval() * rhs.eval())

    def __rmul__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        return self.__mul__(other)

    def __truediv__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        rhs = LazyTensor._ensure_lazy(other)
        return LazyTensor(lambda: self.eval() / rhs.eval())

    def __rtruediv__(self, other: "LazyTensor | Tensor | float") -> "LazyTensor":
        lhs = LazyTensor._ensure_lazy(other)
        return LazyTensor(lambda: lhs.eval() / self.eval())

    def __neg__(self) -> "LazyTensor":
        return LazyTensor(lambda: -self.eval())

    def __matmul__(self, other: "LazyTensor | Tensor") -> "LazyTensor":
        rhs = LazyTensor._ensure_lazy(other)
        return LazyTensor(lambda: self.eval() @ rhs.eval())

    def relu(self) -> "LazyTensor":
        return LazyTensor(lambda: self.eval().relu())

    def exp(self) -> "LazyTensor":
        return LazyTensor(lambda: self.eval().exp())

    def log(self) -> "LazyTensor":
        return LazyTensor(lambda: self.eval().log())

    def sigmoid(self) -> "LazyTensor":
        return LazyTensor(lambda: self.eval().sigmoid())

    def transpose(self) -> "LazyTensor":
        return LazyTensor(lambda: self.eval().transpose())

    @property
    def T(self) -> "LazyTensor":
        return self.transpose()

    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> "LazyTensor":
        return LazyTensor(lambda: self.eval().sum(axis=axis, keepdims=keepdims))

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> "LazyTensor":
        return LazyTensor(lambda: self.eval().mean(axis=axis, keepdims=keepdims))

    def softmax(self, axis: int = -1, eps: float = 1e-6) -> "LazyTensor":
        # Keep `eps` for API compatibility; eager Tensor.softmax currently
        # implements stable softmax without an explicit epsilon parameter.
        _ = eps
        return LazyTensor(lambda: self.eval().softmax(axis=axis))


def lazy_mse_loss(pred: LazyTensor, target: LazyTensor) -> LazyTensor:
    diff = pred - target
    sq = diff * diff
    return (1.0 / np.float32(np.prod(sq.shape))) * sq.sum()


def lazy_binary_cross_entropy(
    pred: LazyTensor, target: LazyTensor, eps: float = 1e-6
) -> LazyTensor:
    eps_t = LazyTensor.scalar(float(eps), requires_grad=False)
    pos = target * (pred + eps_t).log()
    neg = (1.0 - target) * ((1.0 - pred) + eps_t).log()
    return -(pos + neg).mean()

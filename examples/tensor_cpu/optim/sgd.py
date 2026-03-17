"""SGD optimizer."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..tensor import Tensor


class SGD:
    """Stochastic gradient descent optimizer."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        dampening: float = 0.0,
    ) -> None:
        self.params: List[Tensor] = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.dampening = float(dampening)
        self._velocity: List[np.ndarray | None] = [None] * len(self.params)

        if self.nesterov and self.momentum <= 0.0:
            raise ValueError("Nesterov momentum requires momentum > 0")

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = np.asarray(p.grad, dtype=np.float32)
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            if self.momentum != 0.0:
                if self._velocity[idx] is None:
                    self._velocity[idx] = np.zeros_like(grad, dtype=np.float32)
                self._velocity[idx] = self.momentum * self._velocity[idx] + (1.0 - self.dampening) * grad
                if self.nesterov:
                    update = grad + self.momentum * self._velocity[idx]
                else:
                    update = self._velocity[idx]
            else:
                update = grad

            p.data = p.data - self.lr * update

"""Adam optimizer."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ..tensor import Tensor


class Adam:
    """Adam optimizer with optional L2 weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params: List[Tensor] = list(params)
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        self._step = 0
        self._m = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
        self._v = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self._step += 1
        t = float(self._step)

        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = np.asarray(p.grad, dtype=np.float32)
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self._m[idx] = self.beta1 * self._m[idx] + (1.0 - self.beta1) * grad
            self._v[idx] = self.beta2 * self._v[idx] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self._m[idx] / (1.0 - self.beta1**t)
            v_hat = self._v[idx] / (1.0 - self.beta2**t)

            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

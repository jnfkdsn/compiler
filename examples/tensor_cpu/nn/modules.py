"""Neural network module abstractions and losses."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..tensor import Tensor


class Module:
    """Base class for neural network modules."""

    training: bool = True

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        self.training = bool(mode)
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def children(self) -> List["Module"]:
        out: List[Module] = []
        for value in self.__dict__.values():
            if isinstance(value, Module):
                out.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        out.append(item)
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    """Fully connected layer: y = xW + b."""

    def __init__(self, in_features: int, out_features: int) -> None:
        scale = np.sqrt(2.0 / max(in_features, 1))
        w = np.random.randn(in_features, out_features).astype(np.float32) * scale
        b = np.zeros((out_features,), dtype=np.float32)
        self.weight = Tensor.from_numpy(w, requires_grad=True, name="weight")
        self.bias = Tensor.from_numpy(b, requires_grad=True, name="bias")

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight) + self.bias


class ReLU(Module):
    """ReLU activation module."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """Sigmoid activation module."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Dropout(Module):
    """Dropout layer with inverted scaling."""

    def __init__(self, p: float = 0.5) -> None:
        if p < 0.0 or p >= 1.0:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.p = float(p)
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if x.data is None:
            raise RuntimeError("Dropout requires eager tensor data.")

        keep_scale = np.float32(1.0 / (1.0 - self.p))
        mask = (np.random.rand(*x.data.shape) >= self.p).astype(np.float32) * keep_scale
        mask_t = Tensor(data=mask, node=None, name="dropout_mask", requires_grad=False)
        return x * mask_t


class BatchNorm1d(Module):
    """Batch normalization over channel dimension for 2D input: (N, C)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be positive")

        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        self.training = True

        if self.affine:
            gamma = np.ones((1, self.num_features), dtype=np.float32)
            beta = np.zeros((1, self.num_features), dtype=np.float32)
            self.weight = Tensor.from_numpy(gamma, requires_grad=True, name="bn_weight")
            self.bias = Tensor.from_numpy(beta, requires_grad=True, name="bn_bias")

        self.running_mean = np.zeros((1, self.num_features), dtype=np.float32)
        self.running_var = np.ones((1, self.num_features), dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if x.data is None:
            raise RuntimeError("BatchNorm1d requires eager tensor data.")
        if x.data.ndim != 2 or x.data.shape[1] != self.num_features:
            raise ValueError(f"BatchNorm1d expects input shape (N, {self.num_features}), got {x.data.shape}")

        eps_t = Tensor.scalar(self.eps, requires_grad=False)

        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            centered = x - mean
            var = (centered * centered).mean(axis=0, keepdims=True)

            if self.track_running_stats:
                self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            mean = Tensor(data=self.running_mean.copy(), node=None, name="bn_running_mean", requires_grad=False)
            var = Tensor(data=self.running_var.copy(), node=None, name="bn_running_var", requires_grad=False)
            centered = x - mean

        std = ((var + eps_t).log() * 0.5).exp()
        normalized = centered / std

        if self.affine:
            return normalized * self.weight + self.bias
        return normalized


class LayerNorm(Module):
    """Layer normalization over the last dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, affine: bool = True) -> None:
        if normalized_shape <= 0:
            raise ValueError("normalized_shape must be positive")

        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.training = True

        if self.affine:
            gamma = np.ones((1, self.normalized_shape), dtype=np.float32)
            beta = np.zeros((1, self.normalized_shape), dtype=np.float32)
            self.weight = Tensor.from_numpy(gamma, requires_grad=True, name="ln_weight")
            self.bias = Tensor.from_numpy(beta, requires_grad=True, name="ln_bias")

    def forward(self, x: Tensor) -> Tensor:
        if x.data is None:
            raise RuntimeError("LayerNorm requires eager tensor data.")
        if x.data.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expects last dimension {self.normalized_shape}, got {x.data.shape[-1]}"
            )

        eps_t = Tensor.scalar(self.eps, requires_grad=False)
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        std = ((var + eps_t).log() * 0.5).exp()
        normalized = centered / std

        if self.affine:
            return normalized * self.weight + self.bias
        return normalized


class MLP(Module):
    """Simple multi-layer perceptron builder."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        activation: str = "relu",
        out_activation: str | None = None,
    ) -> None:
        self.layers: List[Module] = []
        dims = [in_features, *hidden_features, out_features]

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if is_last:
                if out_activation is not None:
                    self.layers.append(_build_activation(out_activation))
            else:
                self.layers.append(_build_activation(activation))

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Sequential(Module):
    """Compose modules linearly."""

    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class SelfAttention(Module):
    """Single-head self-attention for 2D input: (seq_len, d_model)."""

    def __init__(self, d_model: int, d_k: int | None = None) -> None:
        self.d_model = int(d_model)
        self.d_k = int(d_k) if d_k is not None else int(d_model)
        self.q_proj = Linear(self.d_model, self.d_k)
        self.k_proj = Linear(self.d_model, self.d_k)
        self.v_proj = Linear(self.d_model, self.d_k)
        self.out_proj = Linear(self.d_k, self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.data is None:
            raise RuntimeError("SelfAttention requires eager tensor data.")
        if x.data.ndim != 2 or x.data.shape[1] != self.d_model:
            raise ValueError(f"SelfAttention expects input shape (L, {self.d_model}), got {x.data.shape}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scale = np.float32(1.0 / np.sqrt(float(self.d_k)))
        scores = (q @ k.transpose()) * scale
        weights = scores.softmax(axis=-1)
        context = weights @ v
        return self.out_proj(context)


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred - target
    sq = diff * diff
    # Scalar loss used for backward in this tiny framework.
    return (1.0 / np.float32(sq.data.size)) * sq.sum()


def binary_cross_entropy(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean binary cross entropy on probability predictions."""
    eps_t = Tensor.scalar(float(eps), requires_grad=False)
    pos = target * (pred + eps_t).log()
    neg = (1.0 - target) * ((1.0 - pred) + eps_t).log()
    return -(pos + neg).mean()


def _build_activation(name: str) -> Module:
    lower = name.lower()
    if lower == "relu":
        return ReLU()
    if lower == "sigmoid":
        return Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")

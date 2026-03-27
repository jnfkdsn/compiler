"""Static shape and dtype inference for graph operations."""

from __future__ import annotations

from .ops import OpType

Shape = tuple[int, ...]
AxisLike = int | tuple[int, ...] | None
SUPPORTED_DTYPES = frozenset({"float32", "float64"})


class ShapeInferenceError(ValueError):
    """Raised when tensor shapes are incompatible for an operation."""


def infer_binary(
    op_type: OpType, lhs_shape: Shape, rhs_shape: Shape, lhs_dtype: str, rhs_dtype: str
) -> tuple[Shape, str]:
    out_dtype = _merge_dtype(lhs_dtype, rhs_dtype)

    if op_type in (OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.EQ):
        return _broadcast_shape(lhs_shape, rhs_shape), out_dtype

    if op_type == OpType.RELU_GRAD:
        return _broadcast_shape(lhs_shape, rhs_shape), out_dtype

    if op_type == OpType.MATMUL:
        if len(lhs_shape) < 2 or len(rhs_shape) < 2:
            raise ShapeInferenceError(
                f"MatMul requires at least 2D inputs, got {lhs_shape} @ {rhs_shape}"
            )
        if lhs_shape[-1] != rhs_shape[-2]:
            raise ShapeInferenceError(f"MatMul inner dimension mismatch: {lhs_shape} @ {rhs_shape}")
        batch_l = lhs_shape[:-2]
        batch_r = rhs_shape[:-2]
        batch = _broadcast_shape(batch_l, batch_r) if (batch_l or batch_r) else ()
        return batch + (lhs_shape[-2], rhs_shape[-1]), out_dtype

    raise ShapeInferenceError(f"Unsupported binary op for inference: {op_type}")


def infer_unary(op_type: OpType, src_shape: Shape, src_dtype: str) -> tuple[Shape, str]:
    if op_type in (OpType.RELU, OpType.EXP, OpType.LOG, OpType.SIGMOID):
        return src_shape, src_dtype
    if op_type == OpType.TRANSPOSE:
        if len(src_shape) < 2:
            return src_shape, src_dtype
        return tuple(reversed(src_shape)), src_dtype
    raise ShapeInferenceError(f"Unsupported unary op for inference: {op_type}")


def infer_reduce(
    op_type: OpType,
    src_shape: Shape,
    src_dtype: str,
    axis: AxisLike,
    keepdims: bool,
) -> tuple[Shape, str, tuple[int, ...]]:
    if op_type not in (OpType.SUM, OpType.MEAN, OpType.MAX):
        raise ShapeInferenceError(f"Unsupported reduce op for inference: {op_type}")
    if src_dtype not in SUPPORTED_DTYPES:
        raise ShapeInferenceError(
            f"Unsupported dtype for reduce: {src_dtype}. Supported: {SUPPORTED_DTYPES}"
        )

    axes = normalize_reduce_axes(axis=axis, ndim=len(src_shape))
    if not axes:
        return src_shape, src_dtype, axes

    if keepdims:
        out_shape = tuple(1 if i in set(axes) else dim for i, dim in enumerate(src_shape))
    else:
        out_shape = tuple(dim for i, dim in enumerate(src_shape) if i not in set(axes))
    return out_shape, src_dtype, axes


def normalize_reduce_axes(axis: AxisLike, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))

    raw = (axis,) if isinstance(axis, int) else tuple(axis)
    norm: list[int] = []
    seen: set[int] = set()
    for a in raw:
        aa = int(a)
        if aa < 0:
            aa += ndim
        if aa < 0 or aa >= ndim:
            raise ShapeInferenceError(f"Axis {a} is out of range for ndim={ndim}")
        if aa not in seen:
            seen.add(aa)
            norm.append(aa)
    return tuple(sorted(norm))


def _broadcast_shape(lhs_shape: Shape, rhs_shape: Shape) -> Shape:
    rank = max(len(lhs_shape), len(rhs_shape))
    lhs = (1,) * (rank - len(lhs_shape)) + lhs_shape
    rhs = (1,) * (rank - len(rhs_shape)) + rhs_shape

    out = []
    for ld, rd in zip(lhs, rhs):
        if ld == rd:
            out.append(ld)
        elif ld == 1:
            out.append(rd)
        elif rd == 1:
            out.append(ld)
        else:
            raise ShapeInferenceError(f"Broadcast mismatch: {lhs_shape} vs {rhs_shape}")
    return tuple(out)


def _merge_dtype(lhs_dtype: str, rhs_dtype: str) -> str:
    if lhs_dtype not in SUPPORTED_DTYPES or rhs_dtype not in SUPPORTED_DTYPES:
        raise ShapeInferenceError(
            f"Unsupported dtype pair: {lhs_dtype} and {rhs_dtype}. Supported: {SUPPORTED_DTYPES}"
        )
    if "float64" in (lhs_dtype, rhs_dtype):
        return "float64"
    return "float32"

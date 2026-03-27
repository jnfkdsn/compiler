"""Symbolic Shape Engine using SymPy.

This module provides a professional symbolic shape inference engine
that replaces the simple string-based shape propagation with SymPy
algebraic expressions.

Features:
- Symbolic dimension variables (B, S, etc.)
- Algebraic simplification (S * 2 == S + S)
- Inequality solving for shape validation
- Shape constraint propagation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from sympy import (
        And,
        Eq,
    )
    from sympy import Expr as SymExpr
    from sympy import (
        Ge,
        Gt,
        Integer,
        Le,
        Lt,
        Ne,
        Not,
        Or,
        Rational,
    )
    from sympy import Symbol
    from sympy import Symbol as SymSymbol
    from sympy import (
        expand,
        factor,
        simplify,
        solve,
        symbols,
        sympify,
    )
    from sympy.core.numbers import NegativeOne, One, Zero

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    SymExpr = Any
    Symbol = Any


@dataclass(slots=True)
class SymbolicDim:
    """A symbolic dimension expression."""

    expr: Any  # SymPy expression when available, string otherwise
    name: str | None = None

    def __post_init__(self):
        if HAS_SYMPY and not isinstance(self.expr, SymExpr):
            if isinstance(self.expr, str):
                self.expr = sympify(self.expr)
            elif isinstance(self.expr, int):
                self.expr = Integer(self.expr)

    def __add__(self, other: SymbolicDim | int) -> SymbolicDim:
        if HAS_SYMPY:
            other_expr = other.expr if isinstance(other, SymbolicDim) else Integer(other)
            return SymbolicDim(self.expr + other_expr)
        return SymbolicDim(f"({self.expr} + {other})")

    def __radd__(self, other: int) -> SymbolicDim:
        return self.__add__(other)

    def __sub__(self, other: SymbolicDim | int) -> SymbolicDim:
        if HAS_SYMPY:
            other_expr = other.expr if isinstance(other, SymbolicDim) else Integer(other)
            return SymbolicDim(self.expr - other_expr)
        return SymbolicDim(f"({self.expr} - {other})")

    def __rsub__(self, other: int) -> SymbolicDim:
        if HAS_SYMPY:
            return SymbolicDim(Integer(other) - self.expr)
        return SymbolicDim(f"({other} - {self.expr})")

    def __mul__(self, other: SymbolicDim | int) -> SymbolicDim:
        if HAS_SYMPY:
            other_expr = other.expr if isinstance(other, SymbolicDim) else Integer(other)
            return SymbolicDim(self.expr * other_expr)
        return SymbolicDim(f"({self.expr} * {other})")

    def __rmul__(self, other: int) -> SymbolicDim:
        return self.__mul__(other)

    def __floordiv__(self, other: SymbolicDim | int) -> SymbolicDim:
        if HAS_SYMPY:
            other_expr = other.expr if isinstance(other, SymbolicDim) else Integer(other)
            return SymbolicDim(self.expr // other_expr)
        return SymbolicDim(f"({self.expr} // {other})")

    def __mod__(self, other: SymbolicDim | int) -> SymbolicDim:
        if HAS_SYMPY:
            other_expr = other.expr if isinstance(other, SymbolicDim) else Integer(other)
            return SymbolicDim(self.expr % other_expr)
        return SymbolicDim(f"({self.expr} % {other})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (SymbolicDim, int)):
            return False
        if HAS_SYMPY:
            other_expr = other.expr if isinstance(other, SymbolicDim) else Integer(other)
            return simplify(self.expr - other_expr) == 0
        return str(self.expr) == str(other)

    def __hash__(self) -> int:
        return hash(str(self.expr))

    def simplify(self) -> SymbolicDim:
        """Simplify the expression."""
        if HAS_SYMPY:
            return SymbolicDim(simplify(self.expr))
        return self

    def is_constant(self) -> bool:
        """Check if this is a constant (no symbolic variables)."""
        if HAS_SYMPY:
            return self.expr.is_number
        return str(self.expr).isdigit()

    def evaluate(self, bindings: dict[str, int]) -> int:
        """Evaluate the expression with given variable bindings."""
        if HAS_SYMPY:
            result = self.expr
            for name, value in bindings.items():
                result = result.subs(Symbol(name), value)
            return int(result)

        expr_str = str(self.expr)
        for name, value in bindings.items():
            expr_str = expr_str.replace(name, str(value))
        return int(eval(expr_str))

    def __str__(self) -> str:
        if HAS_SYMPY:
            return str(self.expr)
        return str(self.expr)

    def __repr__(self) -> str:
        return f"SymbolicDim({self.expr})"


@dataclass
class SymbolicShape:
    """A symbolic shape (tuple of symbolic dimensions)."""

    dims: tuple[SymbolicDim, ...]

    def __post_init__(self):
        if not isinstance(self.dims, tuple):
            self.dims = tuple(self.dims)

    @property
    def rank(self) -> int:
        return len(self.dims)

    def __getitem__(self, idx: int) -> SymbolicDim:
        return self.dims[idx]

    def __len__(self) -> int:
        return len(self.dims)

    def numel(self) -> SymbolicDim:
        """Compute total number of elements."""
        if not self.dims:
            return SymbolicDim(1)
        result = self.dims[0]
        for d in self.dims[1:]:
            result = result * d
        return result

    def evaluate(self, bindings: dict[str, int]) -> tuple[int, ...]:
        """Evaluate all dimensions with given bindings."""
        return tuple(d.evaluate(bindings) for d in self.dims)

    def __str__(self) -> str:
        return "(" + ", ".join(str(d) for d in self.dims) + ")"

    def __repr__(self) -> str:
        return f"SymbolicShape({self.dims})"


class SymbolicEngine:
    """Symbolic shape inference and constraint solving engine."""

    def __init__(self) -> None:
        self._symbols: dict[str, Symbol] = {}
        self._constraints: list[Any] = []
        self._bindings: dict[str, int] = {}

    def get_symbol(self, name: str) -> Symbol:
        """Get or create a symbolic variable."""
        if name not in self._symbols:
            if HAS_SYMPY:
                self._symbols[name] = Symbol(name, integer=True, positive=True)
            else:
                self._symbols[name] = name
        return self._symbols[name]

    def create_symbolic_dim(self, name: str) -> SymbolicDim:
        """Create a symbolic dimension variable."""
        return SymbolicDim(self.get_symbol(name), name=name)

    def create_symbolic_shape(self, names: list[str]) -> SymbolicShape:
        """Create a symbolic shape from dimension names."""
        return SymbolicShape(tuple(self.create_symbolic_dim(n) for n in names))

    def add_constraint(self, constraint: Any) -> None:
        """Add a shape constraint."""
        if HAS_SYMPY:
            self._constraints.append(constraint)

    def add_equality(self, dim1: SymbolicDim, dim2: SymbolicDim) -> None:
        """Add an equality constraint: dim1 == dim2."""
        if HAS_SYMPY:
            self.add_constraint(Eq(dim1.expr, dim2.expr))

    def add_divisibility(self, dim: SymbolicDim, divisor: int) -> None:
        """Add a divisibility constraint: dim % divisor == 0."""
        if HAS_SYMPY:
            self.add_constraint(Eq(dim.expr % divisor, 0))

    def add_positive(self, dim: SymbolicDim) -> None:
        """Add a positivity constraint: dim > 0."""
        if HAS_SYMPY:
            self.add_constraint(Gt(dim.expr, 0))

    def solve_constraints(self) -> dict[str, int] | None:
        """Solve all constraints and return possible solutions."""
        if not HAS_SYMPY or not self._constraints:
            return None

        try:
            all_constraints = And(*self._constraints)
            solutions = solve(all_constraints, list(self._symbols.values()))
            return solutions
        except Exception:
            return None

    def simplify_expr(self, expr: SymbolicDim) -> SymbolicDim:
        """Simplify a symbolic expression."""
        return expr.simplify()

    def are_equal(self, dim1: SymbolicDim, dim2: SymbolicDim) -> bool:
        """Check if two symbolic dimensions are provably equal."""
        if not HAS_SYMPY:
            return str(dim1) == str(dim2)

        diff = simplify(dim1.expr - dim2.expr)
        if diff == 0:
            return True

        eq_constraint = Eq(dim1.expr, dim2.expr)
        return eq_constraint in self._constraints

    def can_prove(self, condition: Any) -> bool:
        """Try to prove a condition given current constraints."""
        if not HAS_SYMPY:
            return False

        try:
            all_constraints = And(*self._constraints) if self._constraints else True
            combined = And(all_constraints, condition)
            return combined == condition
        except Exception:
            return False


class SymbolicShapeInference:
    """Symbolic shape inference for tensor operations."""

    def __init__(self) -> None:
        self.engine = SymbolicEngine()
        self._node_shapes: dict[int, SymbolicShape] = {}

    def set_input_shape(self, node_id: int, shape: SymbolicShape) -> None:
        """Set the symbolic shape for an input node."""
        self._node_shapes[node_id] = shape

    def set_input_shape_from_names(self, node_id: int, dim_names: list[str]) -> SymbolicShape:
        """Set symbolic shape from dimension name list."""
        shape = self.engine.create_symbolic_shape(dim_names)
        self._node_shapes[node_id] = shape
        return shape

    def get_shape(self, node_id: int) -> SymbolicShape | None:
        """Get the symbolic shape for a node."""
        return self._node_shapes.get(node_id)

    def infer_elementwise_shape(
        self, lhs_shape: SymbolicShape, rhs_shape: SymbolicShape
    ) -> SymbolicShape:
        """Infer shape for elementwise binary operations with broadcasting."""
        max_rank = max(lhs_shape.rank, rhs_shape.rank)

        lhs_padded = (SymbolicDim(1),) * (max_rank - lhs_shape.rank) + lhs_shape.dims
        rhs_padded = (SymbolicDim(1),) * (max_rank - rhs_shape.rank) + rhs_shape.dims

        result_dims = []
        for l, r in zip(lhs_padded, rhs_padded):
            if l.is_constant() and l.evaluate({}) == 1:
                result_dims.append(r)
            elif r.is_constant() and r.evaluate({}) == 1 or self.engine.are_equal(l, r):
                result_dims.append(l)
            else:
                if HAS_SYMPY:
                    self.engine.add_equality(l, r)
                result_dims.append(l)

        return SymbolicShape(tuple(result_dims))

    def infer_matmul_shape(self, a_shape: SymbolicShape, b_shape: SymbolicShape) -> SymbolicShape:
        """Infer shape for matrix multiplication."""
        if a_shape.rank < 2 or b_shape.rank < 2:
            raise ValueError("MatMul requires at least 2D inputs")

        batch_a = a_shape.dims[:-2]
        batch_b = b_shape.dims[:-2]

        batch_shape = self.infer_elementwise_shape(SymbolicShape(batch_a), SymbolicShape(batch_b))

        m = a_shape.dims[-2]
        k_a = a_shape.dims[-1]
        k_b = b_shape.dims[-2]
        n = b_shape.dims[-1]

        self.engine.add_equality(k_a, k_b)

        return SymbolicShape(batch_shape.dims + (m, n))

    def infer_reduce_shape(
        self, input_shape: SymbolicShape, axes: tuple[int, ...], keepdims: bool = False
    ) -> SymbolicShape:
        """Infer shape for reduction operations."""
        if not axes:
            return SymbolicShape((SymbolicDim(1),) if keepdims else ())

        result_dims = []
        for i, dim in enumerate(input_shape.dims):
            if i in axes:
                if keepdims:
                    result_dims.append(SymbolicDim(1))
            else:
                result_dims.append(dim)

        return SymbolicShape(tuple(result_dims))

    def infer_reshape_shape(
        self, input_shape: SymbolicShape, target_dims: list[SymbolicDim]
    ) -> SymbolicShape:
        """Infer and validate reshape shape."""
        input_numel = input_shape.numel()
        target_numel = SymbolicDim(1)
        unknown_idx = -1

        for i, d in enumerate(target_dims):
            if not d.is_constant():
                if unknown_idx >= 0:
                    raise ValueError("Only one dimension can be inferred in reshape")
                unknown_idx = i
            else:
                target_numel = target_numel * d

        if unknown_idx >= 0:
            inferred_dim = input_numel // target_numel
            target_dims = (
                target_dims[:unknown_idx] + [inferred_dim] + target_dims[unknown_idx + 1 :]
            )

        result_numel = SymbolicDim(1)
        for d in target_dims:
            result_numel = result_numel * d

        self.engine.add_equality(input_numel, result_numel)

        return SymbolicShape(tuple(target_dims))

    def infer_broadcast_shape(
        self, input_shape: SymbolicShape, target_shape: SymbolicShape
    ) -> SymbolicShape:
        """Infer shape for broadcast_to operation."""
        return self.infer_elementwise_shape(input_shape, target_shape)

    def infer_transpose_shape(
        self, input_shape: SymbolicShape, perm: tuple[int, ...] | None = None
    ) -> SymbolicShape:
        """Infer shape for transpose operation."""
        if perm is None:
            perm = tuple(reversed(range(input_shape.rank)))

        return SymbolicShape(tuple(input_shape.dims[i] for i in perm))

    def infer_slice_shape(
        self,
        input_shape: SymbolicShape,
        starts: tuple[int, ...],
        ends: tuple[int, ...],
        steps: tuple[int, ...] | None = None,
    ) -> SymbolicShape:
        """Infer shape for slice operation."""
        if steps is None:
            steps = (1,) * len(starts)

        result_dims = []
        for _i, (dim, start, end, step) in enumerate(zip(input_shape.dims, starts, ends, steps)):
            if dim.is_constant():
                d = dim.evaluate({})
                length = max(
                    0,
                    (
                        (min(end, d) - start + step - 1) // step
                        if step > 0
                        else (max(end, -1) - start + step + 1) // step
                    ),
                )
                result_dims.append(SymbolicDim(length))
            else:
                if step == 1:
                    slice_dim = dim - SymbolicDim(start) - SymbolicDim(end)
                else:
                    slice_dim = (dim - SymbolicDim(start)) // SymbolicDim(abs(step))
                result_dims.append(slice_dim.simplify())

        return SymbolicShape(tuple(result_dims))

    def validate_shape(self, shape: SymbolicShape, expected: SymbolicShape) -> bool:
        """Validate that a shape matches expected shape."""
        if shape.rank != expected.rank:
            return False

        return all(self.engine.are_equal(s, e) for s, e in zip(shape.dims, expected.dims))

    def get_strides(self, shape: SymbolicShape) -> tuple[SymbolicDim, ...]:
        """Compute contiguous strides for a shape."""
        if not shape.dims:
            return ()

        strides = [SymbolicDim(1)] * len(shape.dims)
        for i in range(len(shape.dims) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape.dims[i + 1]

        return tuple(strides)

    def to_string_repr(self, shape: SymbolicShape) -> tuple[str, ...]:
        """Convert symbolic shape to string representation for codegen."""
        return tuple(str(d) for d in shape.dims)


def create_symbolic_shape_from_input(input_id: int, rank: int) -> SymbolicShape:
    """Create symbolic shape for an input tensor."""
    inference = SymbolicShapeInference()
    dim_names = [f"in{input_id}_d{d}" for d in range(rank)]
    return inference.set_input_shape_from_names(input_id, dim_names)

"""Tensor Intermediate Representation (TIR).

TIR is a loop-nest IR that sits between high-level Graph IR and low-level C++ AST.
It explicitly represents:
- Loop nests with iteration variables
- Tensor access patterns (load/store)
- Buffer allocations
- Computation bodies

This design is inspired by TVM's TIR and Halide, enabling:
- Loop transformations (split, reorder, unroll, vectorize)
- Memory access optimization
- Parallelization annotations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


class TIRNode(ABC):
    """Base class for all TIR nodes."""

    @abstractmethod
    def accept(self, visitor: "TIRVisitor") -> Any:
        pass


class TIRExpr(TIRNode):
    """Base class for TIR expressions."""

    pass


class TIRStmt(TIRNode):
    """Base class for TIR statements."""

    pass


class LoopAnnotation(Enum):
    """Annotations for loop optimization hints."""

    NONE = auto()
    UNROLL = auto()
    VECTORIZE = auto()
    PARALLEL = auto()
    SIMD = auto()


@dataclass(slots=True)
class Var(TIRExpr):
    """A variable (iteration variable or buffer name)."""

    name: str
    dtype: str = "float32"

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_var(self)


@dataclass(slots=True)
class Const(TIRExpr):
    """A constant value."""

    value: Union[int, float, str]
    dtype: str = "int32"

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_const(self)


@dataclass(slots=True)
class Binary(TIRExpr):
    """Binary arithmetic operation."""

    lhs: TIRExpr
    op: str
    rhs: TIRExpr

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_binary(self)


@dataclass(slots=True)
class Unary(TIRExpr):
    """Unary operation."""

    op: str
    operand: TIRExpr

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_unary(self)


@dataclass(slots=True)
class Ternary(TIRExpr):
    """Ternary conditional expression."""

    cond: TIRExpr
    true_expr: TIRExpr
    false_expr: TIRExpr

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_ternary(self)


@dataclass(slots=True)
class CallExpr(TIRExpr):
    """Function call expression."""

    func: str
    args: List[TIRExpr] = field(default_factory=list)

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_call_expr(self)


@dataclass(slots=True)
class BufferLoad(TIRExpr):
    """Load a value from a buffer: buffer[indices...]."""

    buffer: Var
    indices: List[TIRExpr] = field(default_factory=list)

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_buffer_load(self)


@dataclass(slots=True)
class BufferStore(TIRStmt):
    """Store a value to a buffer: buffer[indices...] = value."""

    buffer: Var
    value: TIRExpr
    indices: List[TIRExpr] = field(default_factory=list)

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_buffer_store(self)


@dataclass(slots=True)
class For(TIRStmt):
    """A for loop with explicit iteration variable and bounds."""

    loop_var: Var
    start: TIRExpr
    stop: TIRExpr
    body: TIRStmt
    annotation: LoopAnnotation = LoopAnnotation.NONE

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_for(self)


@dataclass(slots=True)
class Block(TIRStmt):
    """A block of statements."""

    stmts: List[TIRStmt] = field(default_factory=list)

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_block(self)


@dataclass(slots=True)
class IfStmt(TIRStmt):
    """If statement."""

    cond: TIRExpr
    then_body: TIRStmt
    else_body: Optional[TIRStmt] = None

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_if_stmt(self)


@dataclass(slots=True)
class Allocate(TIRStmt):
    """Allocate a buffer."""

    buffer: Var
    shape: List[TIRExpr]
    dtype: str = "float32"
    body: Optional[TIRStmt] = None

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_allocate(self)


@dataclass(slots=True)
class LetStmt(TIRStmt):
    """Let binding: let var = value in body."""

    var: Var
    value: TIRExpr
    body: TIRStmt

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_let_stmt(self)


@dataclass(slots=True)
class AssertStmt(TIRStmt):
    """Assertion statement."""

    cond: TIRExpr
    message: str
    body: Optional[TIRStmt] = None

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_assert_stmt(self)


@dataclass(slots=True)
class ProducerStore(TIRStmt):
    """Store to a producer (output tensor)."""

    producer: Var
    value: TIRExpr
    indices: List[TIRExpr] = field(default_factory=list)

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_producer_store(self)


@dataclass(slots=True)
class AttrStmt(TIRStmt):
    """Attribute statement for annotations."""

    node: TIRNode
    attr_key: str
    value: TIRExpr
    body: TIRStmt

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_attr_stmt(self)


@dataclass(slots=True)
class Buffer:
    """Represents a tensor buffer with shape and strides."""

    name: str
    shape: List[TIRExpr]
    dtype: str = "float32"
    strides: Optional[List[TIRExpr]] = None
    data: Optional[Var] = None
    scope: str = "global"

    def __post_init__(self):
        if self.data is None:
            self.data = Var(self.name, self.dtype)


@dataclass(slots=True)
class PrimFunc(TIRNode):
    """A primitive function in TIR."""

    name: str
    params: List[Var]
    body: TIRStmt
    buffers: Dict[str, Buffer] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_prim_func(self)


@dataclass(slots=True)
class IRModule(TIRNode):
    """A module containing multiple PrimFuncs."""

    functions: Dict[str, PrimFunc] = field(default_factory=dict)

    def accept(self, visitor: "TIRVisitor") -> Any:
        return visitor.visit_ir_module(self)


class TIRVisitor(ABC):
    """Base visitor for TIR traversal."""

    @abstractmethod
    def visit_var(self, node: Var) -> Any:
        pass

    @abstractmethod
    def visit_const(self, node: Const) -> Any:
        pass

    @abstractmethod
    def visit_binary(self, node: Binary) -> Any:
        pass

    @abstractmethod
    def visit_unary(self, node: Unary) -> Any:
        pass

    @abstractmethod
    def visit_ternary(self, node: Ternary) -> Any:
        pass

    @abstractmethod
    def visit_call_expr(self, node: CallExpr) -> Any:
        pass

    @abstractmethod
    def visit_buffer_load(self, node: BufferLoad) -> Any:
        pass

    @abstractmethod
    def visit_buffer_store(self, node: BufferStore) -> Any:
        pass

    @abstractmethod
    def visit_for(self, node: For) -> Any:
        pass

    @abstractmethod
    def visit_block(self, node: Block) -> Any:
        pass

    @abstractmethod
    def visit_if_stmt(self, node: IfStmt) -> Any:
        pass

    @abstractmethod
    def visit_allocate(self, node: Allocate) -> Any:
        pass

    @abstractmethod
    def visit_let_stmt(self, node: LetStmt) -> Any:
        pass

    @abstractmethod
    def visit_assert_stmt(self, node: AssertStmt) -> Any:
        pass

    @abstractmethod
    def visit_producer_store(self, node: ProducerStore) -> Any:
        pass

    @abstractmethod
    def visit_attr_stmt(self, node: AttrStmt) -> Any:
        pass

    @abstractmethod
    def visit_prim_func(self, node: PrimFunc) -> Any:
        pass

    @abstractmethod
    def visit_ir_module(self, node: IRModule) -> Any:
        pass


class TIRTransformer(TIRVisitor):
    """Base transformer that returns nodes unchanged by default."""

    def visit_var(self, node: Var) -> TIRNode:
        return node

    def visit_const(self, node: Const) -> TIRNode:
        return node

    def visit_binary(self, node: Binary) -> TIRNode:
        return Binary(
            lhs=node.lhs.accept(self),
            op=node.op,
            rhs=node.rhs.accept(self),
        )

    def visit_unary(self, node: Unary) -> TIRNode:
        return Unary(op=node.op, operand=node.operand.accept(self))

    def visit_ternary(self, node: Ternary) -> TIRNode:
        return Ternary(
            cond=node.cond.accept(self),
            true_expr=node.true_expr.accept(self),
            false_expr=node.false_expr.accept(self),
        )

    def visit_call_expr(self, node: CallExpr) -> TIRNode:
        return CallExpr(func=node.func, args=[arg.accept(self) for arg in node.args])

    def visit_buffer_load(self, node: BufferLoad) -> TIRNode:
        return BufferLoad(
            buffer=node.buffer,
            indices=[idx.accept(self) for idx in node.indices],
        )

    def visit_buffer_store(self, node: BufferStore) -> TIRNode:
        return BufferStore(
            buffer=node.buffer,
            value=node.value.accept(self),
            indices=[idx.accept(self) for idx in node.indices],
        )

    def visit_for(self, node: For) -> TIRNode:
        return For(
            loop_var=node.loop_var,
            start=node.start.accept(self),
            stop=node.stop.accept(self),
            body=node.body.accept(self),
            annotation=node.annotation,
        )

    def visit_block(self, node: Block) -> TIRNode:
        return Block(stmts=[stmt.accept(self) for stmt in node.stmts])

    def visit_if_stmt(self, node: IfStmt) -> TIRNode:
        return IfStmt(
            cond=node.cond.accept(self),
            then_body=node.then_body.accept(self),
            else_body=node.else_body.accept(self) if node.else_body else None,
        )

    def visit_allocate(self, node: Allocate) -> TIRNode:
        return Allocate(
            buffer=node.buffer,
            shape=[s.accept(self) for s in node.shape],
            dtype=node.dtype,
            body=node.body.accept(self) if node.body else None,
        )

    def visit_let_stmt(self, node: LetStmt) -> TIRNode:
        return LetStmt(
            var=node.var,
            value=node.value.accept(self),
            body=node.body.accept(self),
        )

    def visit_assert_stmt(self, node: AssertStmt) -> TIRNode:
        return AssertStmt(
            cond=node.cond.accept(self),
            message=node.message,
            body=node.body.accept(self) if node.body else None,
        )

    def visit_producer_store(self, node: ProducerStore) -> TIRNode:
        return ProducerStore(
            producer=node.producer,
            value=node.value.accept(self),
            indices=[idx.accept(self) for idx in node.indices],
        )

    def visit_attr_stmt(self, node: AttrStmt) -> TIRNode:
        return AttrStmt(
            node=node.node,
            attr_key=node.attr_key,
            value=node.value.accept(self),
            body=node.body.accept(self),
        )

    def visit_prim_func(self, node: PrimFunc) -> TIRNode:
        return PrimFunc(
            name=node.name,
            params=node.params,
            body=node.body.accept(self),
            buffers=node.buffers,
            attrs=node.attrs,
        )

    def visit_ir_module(self, node: IRModule) -> TIRNode:
        return IRModule(functions={k: v.accept(self) for k, v in node.functions.items()})

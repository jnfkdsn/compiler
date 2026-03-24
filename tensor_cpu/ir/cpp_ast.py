"""Imperative C++ AST node definitions.

This module provides a strongly-typed AST representation for C++ code generation.
Instead of string concatenation, we build a tree of AST nodes that can be:
- Traversed and transformed (optimization passes)
- Pretty-printed with proper formatting
- Type-checked and validated

The design follows the Visitor pattern for extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union


class ASTNode(ABC):
    """Base class for all AST nodes."""

    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> Any:
        """Accept a visitor to process this node."""
        pass


class Expr(ASTNode):
    """Base class for expression nodes."""

    pass


class Stmt(ASTNode):
    """Base class for statement nodes."""

    pass


@dataclass(slots=True)
class Identifier(Expr):
    """A simple identifier (variable name)."""
    name: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_identifier(self)


@dataclass(slots=True)
class Literal(Expr):
    """A literal value (number, string, etc.)."""
    value: Union[str, int, float]
    dtype: Optional[str] = None

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_literal(self)


@dataclass(slots=True)
class BinaryOp(Expr):
    """Binary operation: lhs op rhs."""
    lhs: Expr
    op: str
    rhs: Expr

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_binary_op(self)


@dataclass(slots=True)
class UnaryOp(Expr):
    """Unary operation: op operand."""
    op: str
    operand: Expr

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_unary_op(self)


@dataclass(slots=True)
class TernaryOp(Expr):
    """Ternary conditional: cond ? true_expr : false_expr."""
    cond: Expr
    true_expr: Expr
    false_expr: Expr

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_ternary_op(self)


@dataclass(slots=True)
class Call(Expr):
    """Function call: func(args...)."""
    func: str
    args: List[Expr] = field(default_factory=list)

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_call(self)


@dataclass(slots=True)
class Index(Expr):
    """Array index access: base[index]."""
    base: Expr
    index: Expr

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_index(self)


@dataclass(slots=True)
class Cast(Expr):
    """Type cast: (type)expr."""
    target_type: str
    expr: Expr

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_cast(self)


@dataclass(slots=True)
class ExprStmt(Stmt):
    """Expression statement (expression followed by semicolon)."""
    expr: Expr

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_expr_stmt(self)


@dataclass(slots=True)
class Assign(Stmt):
    """Assignment statement: lhs = rhs; or lhs += rhs; etc."""
    lhs: Expr
    rhs: Expr
    op: str = "="

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_assign(self)


@dataclass(slots=True)
class VarDecl(Stmt):
    """Variable declaration: type name = init; or type name;."""
    var_type: str
    name: str
    init: Optional[Expr] = None
    is_const: bool = False

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_var_decl(self)


@dataclass(slots=True)
class Block(Stmt):
    """Block of statements: { stmts... }."""
    stmts: List[Stmt] = field(default_factory=list)

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_block(self)


@dataclass(slots=True)
class ForLoop(Stmt):
    """For loop: for (init; cond; update) body."""
    init: Optional[Stmt] = None
    cond: Optional[Expr] = None
    update: Optional[Stmt] = None
    body: Stmt = field(default_factory=lambda: Block())

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_for_loop(self)


@dataclass(slots=True)
class WhileLoop(Stmt):
    """While loop: while (cond) body."""
    cond: Expr
    body: Stmt = field(default_factory=lambda: Block())

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_while_loop(self)


@dataclass(slots=True)
class If(Stmt):
    """If statement: if (cond) then_stmt else else_stmt."""
    cond: Expr
    then_stmt: Stmt
    else_stmt: Optional[Stmt] = None

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_if(self)


@dataclass(slots=True)
class Return(Stmt):
    """Return statement: return expr; or return;."""
    expr: Optional[Expr] = None

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_return(self)


@dataclass(slots=True)
class FunctionDecl(Stmt):
    """Function declaration."""
    return_type: str
    name: str
    params: List[Tuple[str, str]]
    body: Block
    linkage: Optional[str] = None
    export: bool = False

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_function_decl(self)


@dataclass(slots=True)
class StructDecl(Stmt):
    """Struct declaration."""
    name: str
    fields: List[Tuple[str, str]]

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_struct_decl(self)


@dataclass(slots=True)
class Include(Stmt):
    """Include directive."""
    header: str
    is_system: bool = True

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_include(self)


@dataclass(slots=True)
class Define(Stmt):
    """Preprocessor define."""
    name: str
    value: Optional[str] = None

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_define(self)


@dataclass(slots=True)
class Pragma(Stmt):
    """Pragma directive."""
    content: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_pragma(self)


@dataclass(slots=True)
class RawCode(Stmt):
    """Raw C++ code (for cases that don't fit the AST model)."""
    code: str

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_raw_code(self)


@dataclass(slots=True)
class Program(ASTNode):
    """A complete C++ program (translation unit)."""
    includes: List[Include] = field(default_factory=list)
    defines: List[Define] = field(default_factory=list)
    pragmas: List[Pragma] = field(default_factory=list)
    decls: List[Stmt] = field(default_factory=list)

    def accept(self, visitor: "ASTVisitor") -> Any:
        return visitor.visit_program(self)


class ASTVisitor(ABC):
    """Base visitor for AST traversal."""

    @abstractmethod
    def visit_identifier(self, node: Identifier) -> Any:
        pass

    @abstractmethod
    def visit_literal(self, node: Literal) -> Any:
        pass

    @abstractmethod
    def visit_binary_op(self, node: BinaryOp) -> Any:
        pass

    @abstractmethod
    def visit_unary_op(self, node: UnaryOp) -> Any:
        pass

    @abstractmethod
    def visit_ternary_op(self, node: TernaryOp) -> Any:
        pass

    @abstractmethod
    def visit_call(self, node: Call) -> Any:
        pass

    @abstractmethod
    def visit_index(self, node: Index) -> Any:
        pass

    @abstractmethod
    def visit_cast(self, node: Cast) -> Any:
        pass

    @abstractmethod
    def visit_expr_stmt(self, node: ExprStmt) -> Any:
        pass

    @abstractmethod
    def visit_assign(self, node: Assign) -> Any:
        pass

    @abstractmethod
    def visit_var_decl(self, node: VarDecl) -> Any:
        pass

    @abstractmethod
    def visit_block(self, node: Block) -> Any:
        pass

    @abstractmethod
    def visit_for_loop(self, node: ForLoop) -> Any:
        pass

    @abstractmethod
    def visit_while_loop(self, node: WhileLoop) -> Any:
        pass

    @abstractmethod
    def visit_if(self, node: If) -> Any:
        pass

    @abstractmethod
    def visit_return(self, node: Return) -> Any:
        pass

    @abstractmethod
    def visit_function_decl(self, node: FunctionDecl) -> Any:
        pass

    @abstractmethod
    def visit_struct_decl(self, node: StructDecl) -> Any:
        pass

    @abstractmethod
    def visit_include(self, node: Include) -> Any:
        pass

    @abstractmethod
    def visit_define(self, node: Define) -> Any:
        pass

    @abstractmethod
    def visit_pragma(self, node: Pragma) -> Any:
        pass

    @abstractmethod
    def visit_raw_code(self, node: RawCode) -> Any:
        pass

    @abstractmethod
    def visit_program(self, node: Program) -> Any:
        pass


class ASTTransformer(ASTVisitor):
    """Base transformer that returns nodes unchanged by default."""

    def visit_identifier(self, node: Identifier) -> ASTNode:
        return node

    def visit_literal(self, node: Literal) -> ASTNode:
        return node

    def visit_binary_op(self, node: BinaryOp) -> ASTNode:
        return BinaryOp(
            lhs=node.lhs.accept(self),
            op=node.op,
            rhs=node.rhs.accept(self),
        )

    def visit_unary_op(self, node: UnaryOp) -> ASTNode:
        return UnaryOp(op=node.op, operand=node.operand.accept(self))

    def visit_ternary_op(self, node: TernaryOp) -> ASTNode:
        return TernaryOp(
            cond=node.cond.accept(self),
            true_expr=node.true_expr.accept(self),
            false_expr=node.false_expr.accept(self),
        )

    def visit_call(self, node: Call) -> ASTNode:
        return Call(func=node.func, args=[arg.accept(self) for arg in node.args])

    def visit_index(self, node: Index) -> ASTNode:
        return Index(base=node.base.accept(self), index=node.index.accept(self))

    def visit_cast(self, node: Cast) -> ASTNode:
        return Cast(target_type=node.target_type, expr=node.expr.accept(self))

    def visit_expr_stmt(self, node: ExprStmt) -> ASTNode:
        return ExprStmt(expr=node.expr.accept(self))

    def visit_assign(self, node: Assign) -> ASTNode:
        return Assign(lhs=node.lhs.accept(self), rhs=node.rhs.accept(self), op=node.op)

    def visit_var_decl(self, node: VarDecl) -> ASTNode:
        return VarDecl(
            var_type=node.var_type,
            name=node.name,
            init=node.init.accept(self) if node.init else None,
            is_const=node.is_const,
        )

    def visit_block(self, node: Block) -> ASTNode:
        return Block(stmts=[stmt.accept(self) for stmt in node.stmts])

    def visit_for_loop(self, node: ForLoop) -> ASTNode:
        return ForLoop(
            init=node.init.accept(self) if node.init else None,
            cond=node.cond.accept(self) if node.cond else None,
            update=node.update.accept(self) if node.update else None,
            body=node.body.accept(self),
        )

    def visit_while_loop(self, node: WhileLoop) -> ASTNode:
        return WhileLoop(
            cond=node.cond.accept(self),
            body=node.body.accept(self),
        )

    def visit_if(self, node: If) -> ASTNode:
        return If(
            cond=node.cond.accept(self),
            then_stmt=node.then_stmt.accept(self),
            else_stmt=node.else_stmt.accept(self) if node.else_stmt else None,
        )

    def visit_return(self, node: Return) -> ASTNode:
        return Return(expr=node.expr.accept(self) if node.expr else None)

    def visit_function_decl(self, node: FunctionDecl) -> ASTNode:
        return FunctionDecl(
            return_type=node.return_type,
            name=node.name,
            params=node.params,
            body=node.body.accept(self),
            linkage=node.linkage,
            export=node.export,
        )

    def visit_struct_decl(self, node: StructDecl) -> ASTNode:
        return node

    def visit_include(self, node: Include) -> ASTNode:
        return node

    def visit_define(self, node: Define) -> ASTNode:
        return node

    def visit_pragma(self, node: Pragma) -> ASTNode:
        return node

    def visit_raw_code(self, node: RawCode) -> ASTNode:
        return node

    def visit_program(self, node: Program) -> ASTNode:
        return Program(
            includes=[inc.accept(self) for inc in node.includes],
            defines=[d.accept(self) for d in node.defines],
            pragmas=[p.accept(self) for p in node.pragmas],
            decls=[decl.accept(self) for decl in node.decls],
        )

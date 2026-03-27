"""Convert TIR to C++ AST.

This module bridges the gap between Tensor IR and C++ AST,
enabling the final code generation stage.
"""

from __future__ import annotations

from typing import List, Optional

from .cpp_ast import (
    Assign,
    BinaryOp,
)
from .cpp_ast import Block as CppBlock
from .cpp_ast import (
    Call,
    Cast,
    Expr,
    ExprStmt,
    ForLoop,
    FunctionDecl,
    Identifier,
    If,
    Index,
    Literal,
    Program,
    Return,
    Stmt,
    TernaryOp,
    UnaryOp,
    VarDecl,
)
from .tir import (
    Allocate,
    Binary,
    Block,
    BufferLoad,
    BufferStore,
    CallExpr,
    Const,
    For,
    IfStmt,
    IRModule,
    LetStmt,
    LoopAnnotation,
    PrimFunc,
    Ternary,
    TIRExpr,
    TIRStmt,
    TIRVisitor,
    Unary,
    Var,
)


class TIRToCppConverter(TIRVisitor):
    """Convert TIR nodes to C++ AST nodes."""

    def __init__(self) -> None:
        self._let_vars: dict[str, Expr] = {}

    def visit_var(self, node: Var) -> Expr:
        if node.name in self._let_vars:
            return self._let_vars[node.name]
        return Identifier(node.name)

    def visit_const(self, node: Const) -> Expr:
        return Literal(node.value)

    def visit_binary(self, node: Binary) -> Expr:
        lhs = node.lhs.accept(self)
        rhs = node.rhs.accept(self)
        return BinaryOp(lhs, node.op, rhs)

    def visit_unary(self, node: Unary) -> Expr:
        operand = node.operand.accept(self)
        return UnaryOp(node.op, operand)

    def visit_ternary(self, node: Ternary) -> Expr:
        cond = node.cond.accept(self)
        true_expr = node.true_expr.accept(self)
        false_expr = node.false_expr.accept(self)
        return TernaryOp(cond, true_expr, false_expr)

    def visit_call_expr(self, node: CallExpr) -> Expr:
        args = [arg.accept(self) for arg in node.args]
        return Call(node.func, args)

    def visit_buffer_load(self, node: BufferLoad) -> Expr:
        base = node.buffer.accept(self)
        if len(node.indices) == 1:
            return Index(base, node.indices[0].accept(self))
        idx = self._compute_linear_index(node.buffer.name, node.indices)
        return Index(base, idx)

    def _compute_linear_index(self, buffer_name: str, indices: List[TIRExpr]) -> Expr:
        if len(indices) == 1:
            return indices[0].accept(self)
        result = indices[0].accept(self)
        for i in range(1, len(indices)):
            result = BinaryOp(result, "+", indices[i].accept(self))
        return result

    def visit_buffer_store(self, node: BufferStore) -> Stmt:
        base = node.buffer.accept(self)
        if len(node.indices) == 1:
            lhs = Index(base, node.indices[0].accept(self))
        else:
            idx = self._compute_linear_index(node.buffer.name, node.indices)
            lhs = Index(base, idx)
        rhs = node.value.accept(self)
        return Assign(lhs, rhs)

    def visit_for(self, node: For) -> Stmt:
        init = VarDecl(
            var_type="long long",
            name=node.loop_var.name,
            init=node.start.accept(self),
        )

        cond = BinaryOp(
            node.loop_var.accept(self),
            "<",
            node.stop.accept(self),
        )

        update = Assign(
            Identifier(node.loop_var.name),
            BinaryOp(Identifier(node.loop_var.name), "+", Literal(1)),
        )

        body = node.body.accept(self)
        if not isinstance(body, CppBlock):
            body = CppBlock(stmts=[body])

        return ForLoop(init=init, cond=cond, update=update, body=body)

    def visit_block(self, node: Block) -> Stmt:
        stmts = [stmt.accept(self) for stmt in node.stmts]
        return CppBlock(stmts=stmts)

    def visit_if_stmt(self, node: IfStmt) -> Stmt:
        cond = node.cond.accept(self)
        then_body = node.then_body.accept(self)
        if not isinstance(then_body, CppBlock):
            then_body = CppBlock(stmts=[then_body])

        else_body = None
        if node.else_body:
            else_body = node.else_body.accept(self)
            if not isinstance(else_body, CppBlock):
                else_body = CppBlock(stmts=[else_body])

        return If(cond, then_body, else_body)

    def visit_allocate(self, node: Allocate) -> Stmt:
        return CppBlock(stmts=[])

    def visit_let_stmt(self, node: LetStmt) -> Stmt:
        var_name = node.var.name
        value_expr = node.value.accept(self)
        self._let_vars[var_name] = value_expr

        body = node.body.accept(self)

        decl = VarDecl(
            var_type="long long" if node.var.dtype.startswith("int") else "float",
            name=var_name,
            init=value_expr,
        )

        if isinstance(body, CppBlock):
            return CppBlock(stmts=[decl] + body.stmts)
        return CppBlock(stmts=[decl, body])

    def visit_assert_stmt(self, node) -> Stmt:
        return CppBlock(stmts=[])

    def visit_producer_store(self, node) -> Stmt:
        base = node.producer.accept(self)
        if len(node.indices) == 1:
            lhs = Index(base, node.indices[0].accept(self))
        else:
            idx = self._compute_linear_index(node.producer.name, node.indices)
            lhs = Index(base, idx)
        rhs = node.value.accept(self)
        return Assign(lhs, rhs)

    def visit_attr_stmt(self, node) -> Stmt:
        return node.body.accept(self)

    def visit_prim_func(self, node: PrimFunc) -> FunctionDecl:
        params = []
        for p in node.params:
            params.append(("float*" if not p.dtype.startswith("int") else "long long", p.name))

        body = node.body.accept(self)
        if not isinstance(body, CppBlock):
            body = CppBlock(stmts=[body])

        return FunctionDecl(
            return_type="int",
            name=node.name,
            params=params,
            body=body,
        )

    def visit_ir_module(self, node: IRModule) -> Program:
        decls: List[Stmt] = []
        for func in node.functions.values():
            decls.append(func.accept(self))
        return Program(decls=decls)


def convert_tir_to_cpp(tir_node) -> Stmt:
    """Convert a TIR node to C++ AST."""
    converter = TIRToCppConverter()
    return tir_node.accept(converter)

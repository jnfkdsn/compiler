"""C++ code generator using Visitor pattern.

This module implements a CodegenVisitor that traverses the C++ AST
and produces formatted C++ source code.
"""

from __future__ import annotations

from .cpp_ast import (
    Assign,
    ASTVisitor,
    BinaryOp,
    Block,
    Call,
    Cast,
    Define,
    ExprStmt,
    ForLoop,
    FunctionDecl,
    Identifier,
    If,
    Include,
    Index,
    Literal,
    Pragma,
    Program,
    RawCode,
    Return,
    StructDecl,
    TernaryOp,
    UnaryOp,
    VarDecl,
    WhileLoop,
)


class CodegenVisitor(ASTVisitor):
    """Visitor that generates C++ source code from AST."""

    def __init__(self, indent_size: int = 4) -> None:
        self._indent_size = indent_size
        self._indent_level = 0
        self._lines: list[str] = []

    def _indent(self) -> str:
        return " " * (self._indent_level * self._indent_size)

    def _emit(self, line: str) -> None:
        if line:
            self._lines.append(f"{self._indent()}{line}")
        else:
            self._lines.append("")

    def _emit_raw(self, line: str) -> None:
        self._lines.append(line)

    def get_code(self) -> str:
        return "\n".join(self._lines)

    def visit_identifier(self, node: Identifier) -> str:
        return node.name

    def visit_literal(self, node: Literal) -> str:
        if node.dtype == "string":
            return f'"{node.value}"'
        return str(node.value)

    def visit_binary_op(self, node: BinaryOp) -> str:
        lhs = node.lhs.accept(self)
        rhs = node.rhs.accept(self)
        return f"({lhs} {node.op} {rhs})"

    def visit_unary_op(self, node: UnaryOp) -> str:
        operand = node.operand.accept(self)
        return f"({node.op}{operand})"

    def visit_ternary_op(self, node: TernaryOp) -> str:
        cond = node.cond.accept(self)
        true_expr = node.true_expr.accept(self)
        false_expr = node.false_expr.accept(self)
        return f"({cond} ? {true_expr} : {false_expr})"

    def visit_call(self, node: Call) -> str:
        args = ", ".join(arg.accept(self) for arg in node.args)
        return f"{node.func}({args})"

    def visit_index(self, node: Index) -> str:
        base = node.base.accept(self)
        index = node.index.accept(self)
        return f"{base}[{index}]"

    def visit_cast(self, node: Cast) -> str:
        expr = node.expr.accept(self)
        return f"(({node.target_type})({expr}))"

    def visit_expr_stmt(self, node: ExprStmt) -> None:
        expr = node.expr.accept(self)
        self._emit(f"{expr};")

    def visit_assign(self, node: Assign) -> None:
        lhs = node.lhs.accept(self)
        rhs = node.rhs.accept(self)
        self._emit(f"{lhs} {node.op} {rhs};")

    def visit_var_decl(self, node: VarDecl) -> None:
        prefix = "const " if node.is_const else ""
        if node.init:
            init = node.init.accept(self)
            self._emit(f"{prefix}{node.var_type} {node.name} = {init};")
        else:
            self._emit(f"{prefix}{node.var_type} {node.name};")

    def visit_block(self, node: Block) -> None:
        self._emit("{")
        self._indent_level += 1
        for stmt in node.stmts:
            stmt.accept(self)
        self._indent_level -= 1
        self._emit("}")

    def visit_for_loop(self, node: ForLoop) -> None:
        init_str = ""
        if node.init:
            init_code = self._render_stmt_inline(node.init)
            init_str = init_code.rstrip(";")

        cond_str = node.cond.accept(self) if node.cond else ""
        update_str = ""
        if node.update:
            update_code = self._render_stmt_inline(node.update)
            update_str = update_code.rstrip(";")

        self._emit(f"for ({init_str}; {cond_str}; {update_str}) {{")
        self._indent_level += 1
        node.body.accept(self)
        self._indent_level -= 1
        self._emit("}")

    def _render_stmt_inline(self, stmt) -> str:
        sub_visitor = CodegenVisitor(self._indent_size)
        sub_visitor._indent_level = 0
        stmt.accept(sub_visitor)
        return sub_visitor.get_code().strip()

    def visit_while_loop(self, node: WhileLoop) -> None:
        cond = node.cond.accept(self)
        self._emit(f"while ({cond}) {{")
        self._indent_level += 1
        node.body.accept(self)
        self._indent_level -= 1
        self._emit("}")

    def visit_if(self, node: If) -> None:
        cond = node.cond.accept(self)
        self._emit(f"if ({cond}) {{")
        self._indent_level += 1
        node.then_stmt.accept(self)
        self._indent_level -= 1
        if node.else_stmt:
            self._emit("} else {")
            self._indent_level += 1
            node.else_stmt.accept(self)
            self._indent_level -= 1
        self._emit("}")

    def visit_return(self, node: Return) -> None:
        if node.expr:
            expr = node.expr.accept(self)
            self._emit(f"return {expr};")
        else:
            self._emit("return;")

    def visit_function_decl(self, node: FunctionDecl) -> None:
        params = ", ".join(f"{ptype} {pname}" for ptype, pname in node.params)
        prefix = ""
        if node.linkage:
            prefix = f'{node.linkage} "C" '
        if node.export:
            prefix = "EXPORT " + prefix
        self._emit(f"{prefix}{node.return_type} {node.name}({params}) {{")
        self._indent_level += 1
        node.body.accept(self)
        self._indent_level -= 1
        self._emit("}")

    def visit_struct_decl(self, node: StructDecl) -> None:
        self._emit(f"struct {node.name} {{")
        self._indent_level += 1
        for ftype, fname in node.fields:
            self._emit(f"{ftype} {fname};")
        self._indent_level -= 1
        self._emit("};")

    def visit_include(self, node: Include) -> None:
        if node.is_system:
            self._emit_raw(f"#include <{node.header}>")
        else:
            self._emit_raw(f'#include "{node.header}"')

    def visit_define(self, node: Define) -> None:
        if node.value:
            self._emit_raw(f"#define {node.name} {node.value}")
        else:
            self._emit_raw(f"#define {node.name}")

    def visit_pragma(self, node: Pragma) -> None:
        self._emit_raw(f"#pragma {node.content}")

    def visit_raw_code(self, node: RawCode) -> None:
        for line in node.code.split("\n"):
            self._emit_raw(line)

    def visit_program(self, node: Program) -> str:
        for inc in node.includes:
            inc.accept(self)
        if node.includes:
            self._emit("")

        for define in node.defines:
            define.accept(self)
        for pragma in node.pragmas:
            pragma.accept(self)
        if node.defines or node.pragmas:
            self._emit("")

        for decl in node.decls:
            decl.accept(self)
            self._emit("")

        return self.get_code()


def generate_cpp(ast_node) -> str:
    """Convenience function to generate C++ code from any AST node."""
    visitor = CodegenVisitor()
    return ast_node.accept(visitor)

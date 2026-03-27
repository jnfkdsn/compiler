"""Intermediate Representations (IR) for the tensor compiler.

This package provides layered IRs:
- cpp_ast: Low-level C++ AST for imperative code generation
- cpp_codegen: C++ code generator using Visitor pattern
- tir: Tensor-level IR for loop nest optimization
- lowering: Graph IR to TIR lowering
- tir_to_cpp: TIR to C++ AST conversion
"""

from .cpp_ast import (
    Assign,
    ASTNode,
    ASTTransformer,
    ASTVisitor,
    BinaryOp,
    Block,
    Call,
    Cast,
    Define,
    Expr,
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
    Stmt,
    StructDecl,
    TernaryOp,
    UnaryOp,
    VarDecl,
    WhileLoop,
)
from .cpp_codegen import CodegenVisitor, generate_cpp
from .lowering import GraphLowering, LoweringContext
from .tir import (
    Allocate,
    Buffer,
    BufferLoad,
    BufferStore,
    IfStmt,
    IRModule,
    LetStmt,
    LoopAnnotation,
    PrimFunc,
    TIRExpr,
    TIRNode,
    TIRStmt,
    TIRTransformer,
    TIRVisitor,
)
from .tir import Binary as TIRBinary
from .tir import Block as TIRBlock
from .tir import CallExpr as TIRCallExpr
from .tir import Const as TIRConst
from .tir import For as TIRFor
from .tir import Ternary as TIRTernary
from .tir import Unary as TIRUnary
from .tir import Var as TIRVar
from .tir_to_cpp import TIRToCppConverter, convert_tir_to_cpp

__all__ = [
    "ASTNode",
    "ASTVisitor",
    "ASTTransformer",
    "Expr",
    "Stmt",
    "Identifier",
    "Literal",
    "BinaryOp",
    "UnaryOp",
    "TernaryOp",
    "Call",
    "Index",
    "Cast",
    "ExprStmt",
    "Assign",
    "VarDecl",
    "Block",
    "ForLoop",
    "WhileLoop",
    "If",
    "Return",
    "FunctionDecl",
    "StructDecl",
    "Include",
    "Define",
    "Pragma",
    "RawCode",
    "Program",
    "CodegenVisitor",
    "generate_cpp",
    "TIRNode",
    "TIRExpr",
    "TIRStmt",
    "TIRVisitor",
    "TIRTransformer",
    "TIRVar",
    "TIRConst",
    "TIRBinary",
    "TIRUnary",
    "TIRTernary",
    "TIRCallExpr",
    "BufferLoad",
    "BufferStore",
    "TIRFor",
    "TIRBlock",
    "IfStmt",
    "Allocate",
    "LetStmt",
    "Buffer",
    "PrimFunc",
    "IRModule",
    "LoopAnnotation",
    "GraphLowering",
    "LoweringContext",
    "TIRToCppConverter",
    "convert_tir_to_cpp",
]

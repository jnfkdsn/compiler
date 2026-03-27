"""Shared backend data structures for stable C++ code generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(slots=True)
class GeneratedKernel:
    source: str
    entry: str
    output_sym_shape: tuple[str, ...] = ()
    input_ranks: tuple[int, ...] = ()
    workspace_slots: tuple[tuple[str, tuple[str, ...]], ...] = ()
    exact_input_shapes: tuple[tuple[int, ...], ...] = ()


@dataclass(slots=True)
class CppLine:
    text: str


@dataclass(slots=True)
class CppFor:
    init: str
    cond: str
    inc: str
    body: list[CppStmt]


CppStmt = Union[CppLine, CppFor]


def render_cpp_stmts(stmts: list[CppStmt], indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    for stmt in stmts:
        if isinstance(stmt, CppLine):
            lines.append(f"{pad}{stmt.text}")
            continue
        if isinstance(stmt, CppFor):
            lines.append(f"{pad}for ({stmt.init}; {stmt.cond}; {stmt.inc}) {{")
            lines.extend(render_cpp_stmts(stmt.body, indent=indent + 4))
            lines.append(f"{pad}}}")
            continue
        raise TypeError(f"Unsupported CppStmt: {type(stmt)!r}")
    return lines


__all__ = [
    "GeneratedKernel",
    "CppLine",
    "CppFor",
    "CppStmt",
    "render_cpp_stmts",
]

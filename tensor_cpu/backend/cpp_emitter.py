"""Final C++ source emission for stable generated kernels."""

from __future__ import annotations

from typing import List

from ..abi import AbiStatus
from ..ir.graph import Node
from ..ir.ops import OpType


class CppEmitterMixin:
    _compute_dtype: str | None

    def _render_cpp(
        self,
        *,
        ordered: List[Node],
        inputs: List[Node],
        output_node: Node,
        declarations: List[str],
        body: List[str],
        workspace_slots: List[tuple[str, str]] = (),
    ) -> str:
        max_rank = 8
        output_numel = self._numel_expr(output_node)
        output_name = "out_ptr"
        if output_node.op_type == OpType.INPUT:
            output_name = f"arg{inputs.index(output_node)}"
        signature = ", ".join(
            [
                "const TensorDesc* inputs",
                "long long num_inputs",
                "TensorDesc* out_desc",
                "void* workspace",
            ]
        )

        ctype = "double" if self._compute_dtype == "float64" else "float"
        input_bindings = [
            f"{ctype}* arg{i} = static_cast<{ctype}*>(inputs[{i}].data);"
            for i in range(len(inputs))
        ]

        dim_declarations: List[str] = []
        for idx, node in enumerate(inputs):
            for d in range(node.rank):
                dim_declarations.append(f"const long long in{idx}_d{d} = inputs[{idx}].shape[{d}];")

        ws_declarations: List[str] = []
        if workspace_slots:
            ws_declarations.append(f"{ctype}* ws = static_cast<{ctype}*>(workspace);")
            offset_parts: List[str] = []
            for slot_name, numel_expr in workspace_slots:
                if not offset_parts:
                    ws_declarations.append(f"{ctype}* {slot_name} = ws;")
                else:
                    ws_declarations.append(
                        f"{ctype}* {slot_name} = ws + ({' + '.join(offset_parts)});"
                    )
                offset_parts.append(numel_expr)

        input_guards: List[str] = []
        for i, node in enumerate(inputs):
            input_guards.append(
                f"if (inputs[{i}].data == nullptr) return {int(AbiStatus.INPUT_DATA_NULL_BASE) + i};"
            )
            input_guards.append(
                f"if (inputs[{i}].rank != {node.rank}LL) return {int(AbiStatus.INPUT_RANK_MISMATCH_BASE) + i};"
            )

        output_guards: List[str] = [
            f"if (out_desc == nullptr) return {int(AbiStatus.OUT_DESC_NULL)};",
            f"if (out_desc->data == nullptr) return {int(AbiStatus.OUT_DATA_NULL)};",
            f"if (out_desc->rank != {output_node.rank}LL) return {int(AbiStatus.OUT_RANK_MISMATCH)};",
        ]

        return "\n".join(
            [
                "#include <algorithm>",
                "#include <cfloat>",
                "#include <cstddef>",
                "#include <cmath>",
                "#include <cstdint>",
                "#include <immintrin.h>",
                "#ifdef _OPENMP",
                "#include <omp.h>",
                "#endif",
                "",
                "#ifdef _WIN32",
                "#define EXPORT __declspec(dllexport)",
                "#else",
                "#define EXPORT",
                "#endif",
                "",
                f"constexpr int kMaxRank = {max_rank};",
                "struct TensorDesc {",
                "    void* data;",
                "    long long numel;",
                "    long long rank;",
                "    long long shape[kMaxRank];",
                "    long long strides[kMaxRank];",
                "};",
                "",
                f'extern "C" EXPORT int run_kernel({signature}) {{',
                f"    if (num_inputs != {len(inputs)}LL) return {int(AbiStatus.INPUT_COUNT_MISMATCH)};",
                *[f"    {line}" for line in input_guards],
                *[f"    {line}" for line in input_bindings],
                *[f"    {line}" for line in dim_declarations],
                *[f"    {line}" for line in output_guards],
                f"    {ctype}* out_ptr = static_cast<{ctype}*>(out_desc->data);",
                *[f"    {line}" for line in ws_declarations],
                *[f"    {line}" for line in declarations],
                *[f"    {line}" for line in body],
                f"    if ({output_name} != out_ptr) {{",
                f"        for (long long i = 0; i < {output_numel}; ++i) out_ptr[i] = {output_name}[i];",
                "    }",
                "    return 0;",
                "}",
            ]
        )

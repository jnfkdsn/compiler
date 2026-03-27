"""Matmul and fused-matmul lowering for the stable C++ backend."""

from __future__ import annotations

from ..ir.graph import Node
from ..ir.ops import OpType
from .common import CppFor, CppLine, CppStmt


class MatmulLoweringMixin:
    use_hpc_template: bool
    _requires_exact_input_shapes: bool

    def _emit_matmul(self, node: Node, names: dict[int, str]) -> list[str]:
        a_node = self.graph.get_node(node.inputs[0])
        b_node = self.graph.get_node(node.inputs[1])
        a = names[a_node.id]
        b = names[b_node.id]
        c = names[node.id]
        a_sym = self._sym.get(a_node.id, ())
        b_sym = self._sym.get(b_node.id, ())
        c_sym = self._sym.get(node.id, ())

        m_dim = a_sym[-2]
        k_dim = a_sym[-1]
        n_dim = b_sym[-1]
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)

        if len(a_sym) == 2 and len(b_sym) == 2:
            if self.use_hpc_template:
                self._requires_exact_input_shapes = True
                return self._emit_hpc_matmul(
                    a, b, c, int(a_node.shape[-2]), int(b_node.shape[-1]), int(a_node.shape[-1])
                )
            return self._emit_structured(
                [
                    CppFor(
                        init="long long i = 0",
                        cond=f"i < ({m_dim})",
                        inc="++i",
                        body=[
                            CppFor(
                                init="long long j = 0",
                                cond=f"j < ({n_dim})",
                                inc="++j",
                                body=[
                                    CppLine(f"{ctype} acc = {zero};"),
                                    CppFor(
                                        init="long long kk = 0",
                                        cond=f"kk < ({k_dim})",
                                        inc="++kk",
                                        body=[
                                            CppLine(
                                                f"acc += {a}[i * ({k_dim}) + kk] * {b}[kk * ({n_dim}) + j];"
                                            )
                                        ],
                                    ),
                                    CppLine(f"{c}[i * ({n_dim}) + j] = acc;"),
                                ],
                            )
                        ],
                    )
                ]
            )

        batch_c = c_sym[:-2]
        batch_a = a_sym[:-2]
        batch_b = b_sym[:-2]
        rank_bc = len(batch_c)
        batch_size = " * ".join(f"({d})" for d in batch_c) if batch_c else "1"
        padded_a = ("1",) * (rank_bc - len(batch_a)) + batch_a
        padded_b = ("1",) * (rank_bc - len(batch_b)) + batch_b

        c_bs = self._sym_strides_from(batch_c)
        a_bs = self._sym_strides_from(padded_a)
        b_bs = self._sym_strides_from(padded_b)
        mat_a = f"(({m_dim}) * ({k_dim}))"
        mat_b = f"(({k_dim}) * ({n_dim}))"
        mat_c = f"(({m_dim}) * ({n_dim}))"

        body: list[CppStmt] = [
            CppLine("long long a_off = 0;"),
            CppLine("long long b_off = 0;"),
            CppLine("long long rem = batch;"),
        ]
        for d in range(rank_bc):
            body.append(CppLine(f"long long bc{d} = rem / ({c_bs[d]}); rem %= ({c_bs[d]});"))
            if padded_a[d] != "1":
                body.append(CppLine(f"a_off += bc{d} * ({a_bs[d]}) * {mat_a};"))
            if padded_b[d] != "1":
                body.append(CppLine(f"b_off += bc{d} * ({b_bs[d]}) * {mat_b};"))
        body.append(CppLine(f"const long long c_off = batch * {mat_c};"))
        body.append(
            CppFor(
                init="long long i = 0",
                cond=f"i < ({m_dim})",
                inc="++i",
                body=[
                    CppFor(
                        init="long long j = 0",
                        cond=f"j < ({n_dim})",
                        inc="++j",
                        body=[
                            CppLine(f"{ctype} acc = {zero};"),
                            CppFor(
                                init="long long kk = 0",
                                cond=f"kk < ({k_dim})",
                                inc="++kk",
                                body=[
                                    CppLine(
                                        f"acc += {a}[a_off + i * ({k_dim}) + kk] * {b}[b_off + kk * ({n_dim}) + j];"
                                    )
                                ],
                            ),
                            CppLine(f"{c}[c_off + i * ({n_dim}) + j] = acc;"),
                        ],
                    )
                ],
            )
        )
        return self._emit_structured(
            [
                CppFor(
                    init="long long batch = 0",
                    cond=f"batch < ({batch_size})",
                    inc="++batch",
                    body=body,
                )
            ]
        )

    def _emit_fused_matmul(self, node: Node, names: dict[int, str]) -> list[str]:
        a_node = self.graph.get_node(node.inputs[0])
        b_node = self.graph.get_node(node.inputs[1])
        bias_node = self.graph.get_node(node.inputs[2])

        a_sym = self._sym.get(a_node.id, ())
        b_sym = self._sym.get(b_node.id, ())
        if len(a_sym) != 2 or len(b_sym) != 2:
            raise ValueError(f"Fused matmul requires 2D symbolic inputs, got {a_sym} and {b_sym}")

        a = names[a_node.id]
        b = names[b_node.id]
        bias = names[bias_node.id]
        out = names[node.id]

        m_dim = a_sym[-2]
        k_dim = a_sym[-1]
        n_dim = b_sym[-1]
        with_relu = node.op_type == OpType.FUSED_MATMUL_BIAS_RELU
        ctype = self._cpp_type(node)
        zero = self._cpp_zero(node)

        inner_body: list[CppStmt] = [
            CppLine(f"{ctype} acc = {zero};"),
            CppFor(
                init="long long kk = 0",
                cond=f"kk < ({k_dim})",
                inc="++kk",
                body=[CppLine(f"acc += {a}[i * ({k_dim}) + kk] * {b}[kk * ({n_dim}) + j];")],
            ),
            CppLine(f"{ctype} v = acc + {bias}[j];"),
        ]
        if with_relu:
            inner_body.append(CppLine(f"v = v > {zero} ? v : {zero};"))
        inner_body.append(CppLine(f"{out}[i * ({n_dim}) + j] = v;"))

        return self._emit_structured(
            [
                CppFor(
                    init="long long i = 0",
                    cond=f"i < ({m_dim})",
                    inc="++i",
                    body=[
                        CppFor(
                            init="long long j = 0",
                            cond=f"j < ({n_dim})",
                            inc="++j",
                            body=inner_body,
                        )
                    ],
                )
            ]
        )

    def _emit_hpc_matmul(self, a: str, b: str, c: str, m: int, n: int, k: int) -> list[str]:
        return [
            "{",
            "constexpr int BM = 64;",
            "constexpr int BN = 64;",
            "constexpr int BK = 64;",
            f"for (int i = 0; i < {m} * {n}; ++i) {c}[i] = 0.0f;",
            "#pragma omp parallel for schedule(static)",
            f"for (int ii = 0; ii < {m}; ii += BM) {{",
            f"  for (int kk = 0; kk < {k}; kk += BK) {{",
            f"    for (int jj = 0; jj < {n}; jj += BN) {{",
            f"      const int i_max = (ii + BM < {m}) ? (ii + BM) : {m};",
            f"      const int k_max = (kk + BK < {k}) ? (kk + BK) : {k};",
            f"      const int j_max = (jj + BN < {n}) ? (jj + BN) : {n};",
            "      for (int i = ii; i < i_max; ++i) {",
            "        for (int kk2 = kk; kk2 < k_max; ++kk2) {",
            f"          const float a_val = {a}[i * {k} + kk2];",
            "          int j = jj;",
            "          #if defined(__AVX512F__)",
            "          {",
            "            const __m512 a_vec = _mm512_set1_ps(a_val);",
            "            for (; j + 15 < j_max; j += 16) {",
            f"              __m512 c_vec = _mm512_loadu_ps(&{c}[i * {n} + j]);",
            f"              const __m512 b_vec = _mm512_loadu_ps(&{b}[kk2 * {n} + j]);",
            "              c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);",
            f"              _mm512_storeu_ps(&{c}[i * {n} + j], c_vec);",
            "            }",
            "          }",
            "          #elif defined(__AVX2__)",
            "          {",
            "            const __m256 a_vec = _mm256_set1_ps(a_val);",
            "            for (; j + 7 < j_max; j += 8) {",
            f"              __m256 c_vec = _mm256_loadu_ps(&{c}[i * {n} + j]);",
            f"              const __m256 b_vec = _mm256_loadu_ps(&{b}[kk2 * {n} + j]);",
            "              c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);",
            f"              _mm256_storeu_ps(&{c}[i * {n} + j], c_vec);",
            "            }",
            "          }",
            "          #endif",
            "          #pragma omp simd",
            "          for (int j2 = j; j2 < j_max; ++j2) {",
            f"            {c}[i * {n} + j2] += a_val * {b}[kk2 * {n} + j2];",
            "          }",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "}",
        ]

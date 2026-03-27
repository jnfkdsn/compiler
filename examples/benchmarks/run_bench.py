#!/usr/bin/env python
"""Performance benchmark suite for Tensor CPU AI Compiler.

This harness supports three use cases:
1. quick smoke checks for CI
2. broader workload coverage for manual benchmarking
3. optimization experiments comparing feature toggles on identical graphs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, ".")

from tensor_cpu import JITEngine, Tensor, TraceContext

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-4


@dataclass(frozen=True)
class EngineConfig:
    """Compilation knobs for a single benchmark case."""

    use_hpc_template: bool = False
    enable_memory_planner: bool = True
    passes: tuple[str, ...] = ()

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "use_hpc_template": self.use_hpc_template,
            "enable_memory_planner": self.enable_memory_planner,
            "passes": list(self.passes),
        }


@dataclass(frozen=True)
class RuntimeControls:
    """Execution environment controls for reproducibility."""

    threads: Optional[int] = None
    cpu_affinity: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class CaseSpec:
    """A single benchmark case specification."""

    key: str
    label: str
    kind: str
    shape: Tuple[int, ...]
    warmup: int
    repeats: int
    engine: EngineConfig
    build: Callable[[], tuple[Any, tuple[np.ndarray, ...], np.ndarray, Dict[str, Any]]]


@dataclass(frozen=True)
class ExperimentSpec:
    """Derived comparison between two benchmark results."""

    name: str
    baseline_key: str
    candidate_key: str
    metric: str = "median_ms"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    shape: Tuple[int, ...]
    time_ms: float
    compile_ms: float
    median_ms: float
    p95_ms: float
    max_abs_err: float
    gflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "time_ms": self.time_ms,
            "compile_ms": self.compile_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "max_abs_err": self.max_abs_err,
        }
        if self.gflops is not None:
            payload["gflops"] = self.gflops
        if self.bandwidth_gbps is not None:
            payload["bandwidth_gbps"] = self.bandwidth_gbps
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def _parse_cpu_affinity(spec: Optional[str]) -> Optional[tuple[int, ...]]:
    if spec is None or not spec.strip():
        return None
    cpus: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid CPU affinity range: {token}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    if not cpus:
        return None
    return tuple(sorted(cpus))


def _apply_runtime_controls(controls: RuntimeControls) -> Dict[str, Any]:
    applied: Dict[str, Any] = {
        "requested_threads": controls.threads,
        "requested_cpu_affinity": list(controls.cpu_affinity) if controls.cpu_affinity else None,
        "applied_cpu_affinity": None,
        "notes": [],
    }

    if controls.threads is not None:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = str(controls.threads)
        applied["notes"].append("Pinned thread count through OMP/BLAS environment variables.")

    if controls.cpu_affinity is not None:
        if hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, set(controls.cpu_affinity))
            applied["applied_cpu_affinity"] = sorted(os.sched_getaffinity(0))
        else:
            applied["notes"].append("CPU affinity control is not supported on this platform.")

    return applied


def _current_cpu_affinity() -> Optional[list[int]]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def _collect_environment(runtime_controls: Dict[str, Any]) -> Dict[str, Any]:
    compiler = None
    for candidate in ("clang++", "g++", "cl"):
        if shutil.which(candidate):
            compiler = candidate
            break
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "compiler": compiler or "not_found",
        "cpu_count": os.cpu_count(),
        "current_cpu_affinity": _current_cpu_affinity(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "mkldnn_threads": os.environ.get("MKL_NUM_THREADS", "unset"),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "tensor_passes": os.environ.get("TENSOR_PASSES", ""),
        "runtime_controls": runtime_controls,
    }


def _summarize_times(times_ms: Sequence[float]) -> Tuple[float, float, float]:
    mean_ms = float(sum(times_ms) / len(times_ms))
    median_ms = float(np.median(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    return mean_ms, median_ms, p95_ms


def compute_matmul_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def _max_abs_err(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def _compile_module(graph: Any, engine_config: EngineConfig):
    engine = JITEngine(
        use_hpc_template=engine_config.use_hpc_template,
        enable_memory_planner=engine_config.enable_memory_planner,
        passes=list(engine_config.passes),
    )
    compile_start = time.perf_counter()
    module = engine.compile_graph(graph)
    compile_ms = (time.perf_counter() - compile_start) * 1000
    return module, compile_ms


def _validate_and_time(
    module: Any,
    inputs: tuple[np.ndarray, ...],
    expected: np.ndarray,
    *,
    warmup: int,
    repeats: int,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> tuple[float, float, float, float]:
    expected_arr = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(module.run(*inputs), dtype=np.float32)
    max_err = _max_abs_err(actual, expected_arr)
    if not np.allclose(actual, expected_arr, atol=atol, rtol=rtol):
        raise AssertionError(
            f"Benchmark validation failed: max_abs_err={max_err:.6g}, "
            f"shape={actual.shape}, expected_shape={expected_arr.shape}"
        )

    for _ in range(max(warmup - 1, 0)):
        module.run(*inputs)

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        module.run(*inputs)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    return _summarize_times(times_ms) + (max_err,)


def _run_case(case: CaseSpec) -> BenchmarkResult:
    graph, inputs, expected, extra_metadata = case.build()
    module, compile_ms = _compile_module(graph, case.engine)
    avg_time_ms, median_ms, p95_ms, max_err = _validate_and_time(
        module,
        inputs,
        expected,
        warmup=case.warmup,
        repeats=case.repeats,
    )

    metadata = {"kind": case.kind, **case.engine.to_metadata(), **extra_metadata}
    result = BenchmarkResult(
        name=case.key,
        shape=case.shape,
        time_ms=avg_time_ms,
        compile_ms=compile_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        max_abs_err=max_err,
        metadata=metadata,
    )

    if case.kind in {"matmul", "fused_matmul"}:
        m, k, n = case.shape
        flops = compute_matmul_flops(m, n, k)
        result.gflops = flops / (avg_time_ms * 1e-3) / 1e9
    elif case.kind == "elementwise":
        numel = int(np.prod(case.shape))
        logical_accesses = metadata.get("logical_accesses", 3)
        result.bandwidth_gbps = numel * 4 * logical_accesses / (avg_time_ms * 1e-3) / 1e9

    return result


def _build_matmul_case(
    key: str,
    label: str,
    shape: tuple[int, int, int],
    *,
    warmup: int,
    repeats: int,
    engine: EngineConfig,
) -> CaseSpec:
    m, k, n = shape

    def build() -> tuple[Any, tuple[np.ndarray, ...], np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(42)
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        with TraceContext() as tc:
            at = Tensor.from_numpy(a, name="a")
            bt = Tensor.from_numpy(b, name="b")
            (at @ bt).mark_as_output()
            graph = tc.graph
        expected = a @ b
        return graph, (a, b), expected, {"m": m, "k": k, "n": n}

    return CaseSpec(
        key=key,
        label=label,
        kind="matmul",
        shape=shape,
        warmup=warmup,
        repeats=repeats,
        engine=engine,
        build=build,
    )


def _build_fused_case(
    key: str,
    label: str,
    shape: tuple[int, int, int],
    *,
    warmup: int,
    repeats: int,
    engine: EngineConfig,
) -> CaseSpec:
    m, k, n = shape

    def build() -> tuple[Any, tuple[np.ndarray, ...], np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(42)
        x = rng.standard_normal((m, k), dtype=np.float32)
        w = rng.standard_normal((k, n), dtype=np.float32)
        b = rng.standard_normal((n,), dtype=np.float32)
        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            wt = Tensor.from_numpy(w, name="w")
            bt = Tensor.from_numpy(b, name="b")
            (((xt @ wt) + bt).relu()).mark_as_output()
            graph = tc.graph
        expected = np.maximum((x @ w) + b, 0.0)
        return graph, (x, w, b), expected, {"m": m, "k": k, "n": n}

    return CaseSpec(
        key=key,
        label=label,
        kind="fused_matmul",
        shape=shape,
        warmup=warmup,
        repeats=repeats,
        engine=engine,
        build=build,
    )


def _build_elementwise_case(
    key: str,
    label: str,
    shape: tuple[int, ...],
    op: str,
    *,
    warmup: int,
    repeats: int,
    engine: EngineConfig,
) -> CaseSpec:
    def build() -> tuple[Any, tuple[np.ndarray, ...], np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(shape, dtype=np.float32)
        y = rng.standard_normal(shape, dtype=np.float32)
        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            yt = Tensor.from_numpy(y, name="y")
            if op == "add":
                out = xt + yt
                expected = x + y
                inputs = (x, y)
                logical_accesses = 3
            elif op == "mul":
                out = xt * yt
                expected = x * y
                inputs = (x, y)
                logical_accesses = 3
            elif op == "relu":
                out = xt.relu()
                expected = np.maximum(x, 0.0)
                inputs = (x,)
                logical_accesses = 2
            else:
                raise ValueError(f"Unknown elementwise op: {op}")
            out.mark_as_output()
            graph = tc.graph
        return graph, inputs, expected, {"op": op, "logical_accesses": logical_accesses}

    return CaseSpec(
        key=key,
        label=label,
        kind="elementwise",
        shape=shape,
        warmup=warmup,
        repeats=repeats,
        engine=engine,
        build=build,
    )


def _build_chain_case(
    key: str,
    label: str,
    shape: tuple[int, int],
    depth: int,
    *,
    warmup: int,
    repeats: int,
    engine: EngineConfig,
) -> CaseSpec:
    def build() -> tuple[Any, tuple[np.ndarray, ...], np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(shape, dtype=np.float32)
        y = rng.standard_normal(shape, dtype=np.float32)
        z = rng.standard_normal(shape, dtype=np.float32)
        expected = x.copy()
        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            yt = Tensor.from_numpy(y, name="y")
            zt = Tensor.from_numpy(z, name="z")
            cur = xt
            for idx in range(depth):
                phase = idx % 3
                if phase == 0:
                    cur = cur + yt
                    expected = expected + y
                elif phase == 1:
                    cur = cur * zt
                    expected = expected * z
                else:
                    cur = cur.relu()
                    expected = np.maximum(expected, 0.0)
            cur.mark_as_output()
            graph = tc.graph
        accesses = 1 + depth * 3
        return graph, (x, y, z), expected, {"depth": depth, "logical_accesses": accesses}

    return CaseSpec(
        key=key,
        label=label,
        kind="elementwise",
        shape=shape,
        warmup=warmup,
        repeats=repeats,
        engine=engine,
        build=build,
    )


def _build_reduce_case(
    key: str,
    label: str,
    shape: tuple[int, int],
    op: str,
    axis: int,
    *,
    warmup: int,
    repeats: int,
    engine: EngineConfig,
) -> CaseSpec:
    def build() -> tuple[Any, tuple[np.ndarray, ...], np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(shape, dtype=np.float32)
        with TraceContext() as tc:
            xt = Tensor.from_numpy(x, name="x")
            if op == "sum":
                out = xt.sum(axis=axis)
                expected = x.sum(axis=axis, dtype=np.float32)
            elif op == "mean":
                out = xt.mean(axis=axis)
                expected = x.mean(axis=axis, dtype=np.float32)
            else:
                raise ValueError(f"Unknown reduce op: {op}")
            out.mark_as_output()
            graph = tc.graph
        return graph, (x,), np.asarray(expected, dtype=np.float32), {"op": op, "axis": axis}

    return CaseSpec(
        key=key,
        label=label,
        kind="reduce",
        shape=shape,
        warmup=warmup,
        repeats=repeats,
        engine=engine,
        build=build,
    )


def _profile_sizes(quick: bool) -> Dict[str, Any]:
    if quick:
        return {
            "warmup": 1,
            "repeats": 3,
            "matmuls": [(128, 128, 128)],
            "fused": [(128, 128, 128)],
            "elementwise": [((1024, 1024), "add")],
            "chains": [((512, 512), 6)],
            "reductions": [((1024, 512), "sum", 1)],
            "experiment_matmul": (128, 128, 128),
            "experiment_chain_shape": (512, 512),
            "experiment_chain_depth": 6,
        }
    return {
        "warmup": 3,
        "repeats": 10,
        "matmuls": [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 64, 256)],
        "fused": [(128, 128, 128), (256, 256, 256), (512, 512, 512)],
        "elementwise": [((1024, 1024), "add"), ((2048, 2048), "mul"), ((2048, 2048), "relu")],
        "chains": [((1024, 1024), 8)],
        "reductions": [((2048, 512), "sum", 1), ((1024, 1024), "mean", 0)],
        "experiment_matmul": (256, 256, 256),
        "experiment_chain_shape": (1024, 1024),
        "experiment_chain_depth": 8,
    }


def _base_suite(quick: bool) -> list[CaseSpec]:
    sizes = _profile_sizes(quick)
    warmup = sizes["warmup"]
    repeats = sizes["repeats"]
    default_engine = EngineConfig(
        use_hpc_template=True, enable_memory_planner=True, passes=("pipeline",)
    )
    elementwise_engine = EngineConfig(
        use_hpc_template=False, enable_memory_planner=True, passes=("pipeline",)
    )

    cases: list[CaseSpec] = []
    for m, k, n in sizes["matmuls"]:
        cases.append(
            _build_matmul_case(
                key=f"matmul_{m}x{k}x{n}",
                label=f"matmul {m}x{k}x{n}",
                shape=(m, k, n),
                warmup=warmup,
                repeats=repeats,
                engine=default_engine,
            )
        )
    for m, k, n in sizes["fused"]:
        cases.append(
            _build_fused_case(
                key=f"fused_mlp_{m}x{k}x{n}",
                label=f"fused mlp {m}x{k}x{n}",
                shape=(m, k, n),
                warmup=warmup,
                repeats=repeats,
                engine=default_engine,
            )
        )
    for shape, op in sizes["elementwise"]:
        cases.append(
            _build_elementwise_case(
                key=f"{op}_{shape[0]}x{shape[1]}",
                label=f"{op} {shape[0]}x{shape[1]}",
                shape=shape,
                op=op,
                warmup=warmup,
                repeats=repeats,
                engine=elementwise_engine,
            )
        )
    for shape, depth in sizes["chains"]:
        cases.append(
            _build_chain_case(
                key=f"chain_depth{depth}_{shape[0]}x{shape[1]}",
                label=f"elementwise chain depth={depth} {shape[0]}x{shape[1]}",
                shape=shape,
                depth=depth,
                warmup=warmup,
                repeats=repeats,
                engine=elementwise_engine,
            )
        )
    for shape, op, axis in sizes["reductions"]:
        cases.append(
            _build_reduce_case(
                key=f"{op}_axis{axis}_{shape[0]}x{shape[1]}",
                label=f"{op} axis={axis} {shape[0]}x{shape[1]}",
                shape=shape,
                op=op,
                axis=axis,
                warmup=warmup,
                repeats=repeats,
                engine=elementwise_engine,
            )
        )
    return cases


def _experiment_suite(quick: bool) -> tuple[list[CaseSpec], list[ExperimentSpec]]:
    sizes = _profile_sizes(quick)
    warmup = sizes["warmup"]
    repeats = sizes["repeats"]
    matmul_shape = sizes["experiment_matmul"]
    chain_shape = sizes["experiment_chain_shape"]
    chain_depth = sizes["experiment_chain_depth"]

    cases = [
        _build_matmul_case(
            key=f"experiment_matmul_hpc_off_{matmul_shape[0]}x{matmul_shape[1]}x{matmul_shape[2]}",
            label="matmul hpc off",
            shape=matmul_shape,
            warmup=warmup,
            repeats=repeats,
            engine=EngineConfig(
                use_hpc_template=False, enable_memory_planner=True, passes=("pipeline",)
            ),
        ),
        _build_matmul_case(
            key=f"experiment_matmul_hpc_on_{matmul_shape[0]}x{matmul_shape[1]}x{matmul_shape[2]}",
            label="matmul hpc on",
            shape=matmul_shape,
            warmup=warmup,
            repeats=repeats,
            engine=EngineConfig(
                use_hpc_template=True, enable_memory_planner=True, passes=("pipeline",)
            ),
        ),
        _build_chain_case(
            key=f"experiment_chain_passes_off_d{chain_depth}_{chain_shape[0]}x{chain_shape[1]}",
            label="chain passes off",
            shape=chain_shape,
            depth=chain_depth,
            warmup=warmup,
            repeats=repeats,
            engine=EngineConfig(use_hpc_template=False, enable_memory_planner=True, passes=()),
        ),
        _build_chain_case(
            key=f"experiment_chain_passes_on_d{chain_depth}_{chain_shape[0]}x{chain_shape[1]}",
            label="chain passes on",
            shape=chain_shape,
            depth=chain_depth,
            warmup=warmup,
            repeats=repeats,
            engine=EngineConfig(
                use_hpc_template=False, enable_memory_planner=True, passes=("pipeline",)
            ),
        ),
        _build_chain_case(
            key=f"experiment_chain_mem_off_d{chain_depth}_{chain_shape[0]}x{chain_shape[1]}",
            label="chain memory planner off",
            shape=chain_shape,
            depth=chain_depth,
            warmup=warmup,
            repeats=repeats,
            engine=EngineConfig(
                use_hpc_template=False, enable_memory_planner=False, passes=("pipeline",)
            ),
        ),
        _build_chain_case(
            key=f"experiment_chain_mem_on_d{chain_depth}_{chain_shape[0]}x{chain_shape[1]}",
            label="chain memory planner on",
            shape=chain_shape,
            depth=chain_depth,
            warmup=warmup,
            repeats=repeats,
            engine=EngineConfig(
                use_hpc_template=False, enable_memory_planner=True, passes=("pipeline",)
            ),
        ),
    ]

    experiments = [
        ExperimentSpec(
            name="matmul_hpc_speedup",
            baseline_key=cases[0].key,
            candidate_key=cases[1].key,
        ),
        ExperimentSpec(
            name="chain_pipeline_speedup",
            baseline_key=cases[2].key,
            candidate_key=cases[3].key,
        ),
        ExperimentSpec(
            name="chain_memory_planner_speedup",
            baseline_key=cases[4].key,
            candidate_key=cases[5].key,
        ),
    ]
    return cases, experiments


def _run_suite(cases: Iterable[CaseSpec]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        result = _run_case(case)
        results[case.key] = result.to_payload()
        metric_bits = [
            f"run_mean={result.time_ms:.3f} ms",
            f"median={result.median_ms:.3f} ms",
            f"p95={result.p95_ms:.3f} ms",
            f"compile={result.compile_ms:.3f} ms",
            f"max_err={result.max_abs_err:.3g}",
        ]
        if result.gflops is not None:
            metric_bits.append(f"{result.gflops:.2f} GFLOPS")
        if result.bandwidth_gbps is not None:
            metric_bits.append(f"{result.bandwidth_gbps:.2f} GB/s")
        print(f"  {case.key}: " + ", ".join(metric_bits))
    return results


def _build_experiment_summaries(
    results: Dict[str, Dict[str, Any]],
    experiments: Sequence[ExperimentSpec],
) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for experiment in experiments:
        baseline = results.get(experiment.baseline_key)
        candidate = results.get(experiment.candidate_key)
        if baseline is None or candidate is None:
            continue
        baseline_metric = float(baseline[experiment.metric])
        candidate_metric = float(candidate[experiment.metric])
        speedup = baseline_metric / candidate_metric if candidate_metric > 0 else float("inf")
        delta_pct = (
            ((candidate_metric - baseline_metric) / baseline_metric * 100.0)
            if baseline_metric > 0
            else 0.0
        )
        summaries[experiment.name] = {
            "metric": experiment.metric,
            "baseline_key": experiment.baseline_key,
            "candidate_key": experiment.candidate_key,
            "baseline_value": baseline_metric,
            "candidate_value": candidate_metric,
            "speedup": speedup,
            "delta_pct": delta_pct,
        }
    return summaries


def run_all_benchmarks(
    quick: bool, include_experiments: bool
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    results: Dict[str, Any] = {}
    print("=" * 60)
    print("Tensor CPU AI Compiler Performance Benchmarks")
    print("=" * 60)

    print("\n[Core Workloads]")
    results.update(_run_suite(_base_suite(quick)))

    experiment_summaries: Dict[str, Any] = {}
    if include_experiments:
        print("\n[Optimization Experiments]")
        experiment_cases, experiment_specs = _experiment_suite(quick)
        experiment_results = _run_suite(experiment_cases)
        results.update(experiment_results)
        experiment_summaries = _build_experiment_summaries(experiment_results, experiment_specs)
        for name, summary in experiment_summaries.items():
            print(
                f"  {name}: {summary['candidate_key']} vs {summary['baseline_key']} "
                f"=> speedup={summary['speedup']:.3f}x, delta={summary['delta_pct']:.1f}%"
            )

    print("\n" + "=" * 60)
    return results, experiment_summaries


def build_benchmark_report(
    results: Dict[str, Any],
    *,
    profile: str,
    include_experiments: bool,
    runtime_controls: Dict[str, Any],
    experiments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": profile,
        "include_experiments": include_experiments,
        "environment": _collect_environment(runtime_controls),
        "methodology": {
            "timing_unit": "milliseconds",
            "compile_ms": "one-time JIT compile latency",
            "time_ms": "mean runtime across measured repeats",
            "median_ms": "median runtime across measured repeats",
            "p95_ms": "95th percentile runtime across measured repeats",
            "max_abs_err": "max absolute error against a NumPy reference output",
            "correctness_guard": (
                "Every benchmark case validates one compiled execution against a NumPy reference "
                f"with atol={DEFAULT_ATOL} and rtol={DEFAULT_RTOL} before timing."
            ),
            "cpu_thread_note": (
                "Use --threads and --cpu-affinity for reproducible runs. CI smoke runs should use "
                "single-thread mode unless the benchmark is explicitly about thread scaling."
            ),
        },
        "results": results,
        "experiments": experiments or {},
    }


def _unwrap_results(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "results" in payload and isinstance(payload["results"], dict):
        return payload["results"]
    return payload


def _comparison_policy(
    payload: Dict[str, Any], threshold_override: float | None
) -> Tuple[str, float]:
    comparison = payload.get("comparison", {})
    metric = comparison.get("metric", "median_ms")
    threshold = threshold_override
    if threshold is None:
        threshold = comparison.get("regression_threshold", 0.05)
    return str(metric), float(threshold)


def compare_with_baseline(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    threshold: float | None = None,
) -> List[str]:
    regressions = []

    current_results = _unwrap_results(current)
    baseline_results = _unwrap_results(baseline)
    metric, active_threshold = _comparison_policy(baseline, threshold)

    for key, current_val in current_results.items():
        if key not in baseline_results:
            continue
        baseline_val = baseline_results[key]
        if metric in current_val and metric in baseline_val:
            current_time = float(current_val[metric])
            baseline_time = float(baseline_val[metric])
            if current_time > baseline_time * (1 + active_threshold):
                regression_pct = (current_time - baseline_time) / baseline_time * 100
                regressions.append(
                    f"{key} [{metric}]: {current_time:.3f}ms vs {baseline_time:.3f}ms "
                    f"(+{regression_pct:.1f}% slower)"
                )
    return regressions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tensor CPU benchmarks")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--compare", "-c", help="Compare with baseline JSON file")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        help="Regression threshold override (for example: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a reduced benchmark suite for CI smoke checks"
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="Run optimization toggle experiments (HPC, passes, memory planner).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Pin OMP/BLAS thread environment variables to a fixed value.",
    )
    parser.add_argument(
        "--cpu-affinity",
        help="Pin the benchmark process to CPUs, for example: '0' or '0-3,6'.",
    )
    args = parser.parse_args()

    controls = RuntimeControls(
        threads=args.threads,
        cpu_affinity=_parse_cpu_affinity(args.cpu_affinity),
    )
    applied_controls = _apply_runtime_controls(controls)

    results, experiments = run_all_benchmarks(args.quick, args.experiments)
    report = build_benchmark_report(
        results,
        profile="quick" if args.quick else "full",
        include_experiments=args.experiments,
        runtime_controls=applied_controls,
        experiments=experiments,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nResults written to {args.output}")

    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        regressions = compare_with_baseline(report, baseline, args.threshold)
        if regressions:
            print("\n" + "!" * 60)
            print("PERFORMANCE REGRESSIONS DETECTED:")
            for regression in regressions:
                print(f"  - {regression}")
            print("!" * 60)
            return 1
        print("\nNo performance regressions detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

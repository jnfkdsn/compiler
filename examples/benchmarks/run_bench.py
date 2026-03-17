#!/usr/bin/env python3
"""Simple benchmark harness for tensor_cpu: matmul microbenchmark.

Usage example:
  python examples/benchmarks/run_bench.py --m 512 --k 512 --n 512 --warmup 3 --repeat 20 --passes pipeline --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import psutil
except Exception:
    psutil = None


def add_examples_to_path():
    # Ensure examples folder is on sys.path so we can import tensor_cpu package
    here = Path(__file__).resolve()
    examples_dir = here.parents[1]
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))


def build_and_compile_matmul(m: int, k: int, n: int, passes: list | None = None):
    from tensor_cpu.tracer import TraceContext
    from tensor_cpu.tensor import Tensor
    from tensor_cpu.runtime import JITEngine

    a0 = np.zeros((m, k), dtype=np.float32)
    b0 = np.zeros((k, n), dtype=np.float32)

    with TraceContext() as tc:
        a = Tensor.from_numpy(a0, name="a")
        b = Tensor.from_numpy(b0, name="b")
        c = a @ b
        c = c.mark_as_output()
        _ = c
        graph = tc.graph

    # Create engine; allow passes control via environment or caller
    engine = JITEngine(passes=passes)
    t0 = time.perf_counter()
    mod = engine.compile_graph(graph)
    compile_time = time.perf_counter() - t0
    return mod, compile_time


def build_and_compile_mlp(batch: int, in_dim: int, hidden: int, out_dim: int, passes: list | None = None):
    from tensor_cpu.tracer import TraceContext
    from tensor_cpu.tensor import Tensor
    from tensor_cpu.runtime import JITEngine

    x0 = np.zeros((batch, in_dim), dtype=np.float32)
    w1 = np.random.randn(in_dim, hidden).astype(np.float32)
    b1 = np.random.randn(hidden).astype(np.float32)
    w2 = np.random.randn(hidden, out_dim).astype(np.float32)
    b2 = np.random.randn(out_dim).astype(np.float32)

    with TraceContext() as tc:
        x = Tensor.from_numpy(x0, name="x")
        W1 = Tensor.from_numpy(w1, name="w1")
        B1 = Tensor.from_numpy(b1, name="b1")
        W2 = Tensor.from_numpy(w2, name="w2")
        B2 = Tensor.from_numpy(b2, name="b2")

        h = x @ W1
        # broadcast bias
        h = h + B1
        h = h.relu()
        out = h @ W2
        out = out + B2
        out = out.mark_as_output()
        graph = tc.graph

    engine = JITEngine(passes=passes)
    t0 = time.perf_counter()
    mod = engine.compile_graph(graph)
    compile_time = time.perf_counter() - t0
    # inputs order: x, w1, b1, w2, b2
    inputs = (x0, w1, b1, w2, b2)
    return mod, compile_time, inputs


def build_and_compile_chain(batch: int, dim: int, length: int, passes: list | None = None):
    from tensor_cpu.tracer import TraceContext
    from tensor_cpu.tensor import Tensor
    from tensor_cpu.runtime import JITEngine

    x0 = np.zeros((batch, dim), dtype=np.float32)

    with TraceContext() as tc:
        x = Tensor.from_numpy(x0, name="x")
        v = x
        for i in range(length):
            v = v * 1.1
            v = v + 0.5
            v = v.relu()
        out = v.mark_as_output()
        graph = tc.graph

    engine = JITEngine(passes=passes)
    t0 = time.perf_counter()
    mod = engine.compile_graph(graph)
    compile_time = time.perf_counter() - t0
    return mod, compile_time, (x0,)


def measure_run(mod, a, b, warmup: int, repeat: int):
    # Warmup
    for _ in range(warmup):
        mod.run(a, b)

    times = []
    peak_rss = 0
    proc = psutil.Process() if psutil is not None else None
    for _ in range(repeat):
        if proc:
            rss_before = proc.memory_info().rss
        t0 = time.perf_counter()
        mod.run(a, b)
        t1 = time.perf_counter()
        if proc:
            rss_after = proc.memory_info().rss
            peak_rss = max(peak_rss, rss_before, rss_after)
        times.append((t1 - t0) * 1000.0)

    import statistics

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.pstdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_rss": peak_rss,
    }


def measure_run_inputs(mod, inputs: tuple, warmup: int, repeat: int):
    # Warmup
    for _ in range(warmup):
        mod.run(*inputs)

    times = []
    peak_rss = 0
    proc = psutil.Process() if psutil is not None else None
    for _ in range(repeat):
        if proc:
            rss_before = proc.memory_info().rss
        t0 = time.perf_counter()
        mod.run(*inputs)
        t1 = time.perf_counter()
        if proc:
            rss_after = proc.memory_info().rss
            peak_rss = max(peak_rss, rss_before, rss_after)
        times.append((t1 - t0) * 1000.0)

    import statistics

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.pstdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_rss": peak_rss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="matmul", choices=("matmul","mlp","chain"))
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512, help="hidden size for mlp")
    parser.add_argument("--chain-length", type=int, default=20, help="length of elementwise chain")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--passes", type=str, default="", help="comma-separated passes to enable (e.g. 'pipeline')")
    parser.add_argument("--out", type=str, default="bench_results.csv")
    args = parser.parse_args()

    add_examples_to_path()

    passes_list = [p.strip() for p in args.passes.split(",") if p.strip()] if args.passes else None
    # also set env for backward compatibility/tools
    if args.passes:
        os.environ["TENSOR_PASSES"] = args.passes

    model = args.model
    if model == "matmul":
        m, k, n = args.m, args.k, args.n
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        print(f"Compiling matmul {m}x{k} * {k}x{n} (passes={args.passes})...")
        mod, compile_time = build_and_compile_matmul(m, k, n, passes=passes_list)
        inputs = (a, b)
        meta = {"model": "matmul", "m": m, "k": k, "n": n}
    elif model == "mlp":
        batch = args.m
        in_dim = args.k
        hidden = args.hidden
        out_dim = args.n
        print(f"Compiling MLP batch={batch} in={in_dim} hidden={hidden} out={out_dim} (passes={args.passes})...")
        mod, compile_time, inputs = build_and_compile_mlp(batch, in_dim, hidden, out_dim, passes=passes_list)
        meta = {"model": "mlp", "batch": batch, "in": in_dim, "hidden": hidden, "out": out_dim}
    else:  # chain
        batch = args.m
        dim = args.k
        length = args.chain_length
        print(f"Compiling chain batch={batch} dim={dim} len={length} (passes={args.passes})...")
        mod, compile_time, inputs = build_and_compile_chain(batch, dim, length, passes=passes_list)
        meta = {"model": "chain", "batch": batch, "dim": dim, "length": length}

    print(f"Compile time: {compile_time:.3f}s")

    print("Running benchmark...")
    res = measure_run_inputs(mod, inputs, warmup=args.warmup, repeat=args.repeat)

    row = {**meta, "passes": os.environ.get("TENSOR_PASSES", ""), "compile_time_s": compile_time, **res}

    out_path = Path(args.out)
    header = list(row.keys())
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("Result:")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

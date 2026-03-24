# Benchmarks

Run the benchmark harness from the repository root:

```bash
python examples/benchmarks/run_bench.py --quick --threads 1 --output benchmark-results.json
python examples/benchmarks/run_bench.py --quick --threads 1 --compare examples/benchmarks/baseline.quick.json
python examples/benchmarks/run_bench.py --quick --experiments --threads 1 --output benchmark-experiments.json
python examples/benchmarks/run_bench.py --threads 1 --cpu-affinity 0-3 --output benchmark-full.json
python examples/benchmarks/run_bench.py --quick --compare examples/benchmarks/baseline.quick.json --threshold 0.10
```

Notes:

- `--quick` is intended for CI smoke runs.
- `--experiments` runs paired optimization studies for HPC matmul, the pass pipeline, and the memory planner.
- Full runs use larger shapes and more repeats, and add tall-skinny matmul, elementwise chains, and reductions.
- Every case validates one compiled execution against a NumPy reference before timing and reports `max_abs_err`.
- Output JSON includes environment metadata, runtime controls, compile latency, mean runtime, median runtime, p95 runtime, and optional experiment summaries.
- Regression comparison defaults to the metric and threshold stored in the baseline file.
- The checked-in quick baseline currently compares `median_ms` with a 35% regression threshold.
- For reproducible measurements, prefer `--threads 1` and `--cpu-affinity ...` unless you are explicitly testing thread scaling.
- The quick baseline does not cover experiment-only cases; if you want regression checks for optimization experiments, first generate a matching baseline with `--quick --experiments`.

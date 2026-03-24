"""Tests for the standalone benchmark harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_benchmark_module():
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "run_bench.py"
    spec = importlib.util.spec_from_file_location("tensor_cpu_benchmark_harness", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def test_parse_cpu_affinity_supports_ranges():
    assert bench._parse_cpu_affinity("0-2,4,6-7") == (0, 1, 2, 4, 6, 7)
    assert bench._parse_cpu_affinity("3") == (3,)
    assert bench._parse_cpu_affinity(None) is None


def test_compare_with_baseline_uses_metric_from_baseline():
    current = {"results": {"matmul": {"median_ms": 12.0, "compile_ms": 100.0}}}
    baseline = {
        "comparison": {"metric": "median_ms", "regression_threshold": 0.10},
        "results": {"matmul": {"median_ms": 10.0, "compile_ms": 10.0}},
    }

    regressions = bench.compare_with_baseline(current, baseline)
    assert regressions == ["matmul [median_ms]: 12.000ms vs 10.000ms (+20.0% slower)"]

    no_regressions = bench.compare_with_baseline(current, baseline, threshold=0.25)
    assert no_regressions == []


def test_build_experiment_summaries_reports_speedup():
    results = {
        "off": {"median_ms": 10.0},
        "on": {"median_ms": 5.0},
    }
    experiments = [
        bench.ExperimentSpec(
            name="pipeline_speedup",
            baseline_key="off",
            candidate_key="on",
        )
    ]

    summary = bench._build_experiment_summaries(results, experiments)
    assert summary["pipeline_speedup"]["speedup"] == 2.0
    assert summary["pipeline_speedup"]["delta_pct"] == -50.0


def test_build_benchmark_report_embeds_runtime_controls_and_experiments():
    report = bench.build_benchmark_report(
        {"case": {"median_ms": 1.0}},
        profile="quick",
        include_experiments=True,
        runtime_controls={"requested_threads": 1, "requested_cpu_affinity": [0], "notes": []},
        experiments={"exp": {"speedup": 1.25}},
    )

    assert report["schema_version"] == 2
    assert report["profile"] == "quick"
    assert report["include_experiments"] is True
    assert report["experiments"]["exp"]["speedup"] == 1.25
    assert report["environment"]["runtime_controls"]["requested_threads"] == 1

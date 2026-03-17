"""Unit tests for passes: constant folding and CSE, plus a small perf comparison."""

from __future__ import annotations

import sys
import time
import numpy as np

# ensure examples/ is on path so `import tensor_cpu` resolves when running from examples/
sys.path.insert(0, ".")

from tensor_cpu.graph import Graph
from tensor_cpu.ops import OpType
from tensor_cpu.passes.constfold import constant_fold
from tensor_cpu.passes.cse import cse
from tensor_cpu.runtime import JITEngine
from tensor_cpu.static_graph import StaticGraph


def test_constant_fold_simple():
    g = Graph()
    c1 = g.add_node(op_type=OpType.CONST, name="c1", shape=(), dtype="float32", attrs={"value": 1.5})
    c2 = g.add_node(op_type=OpType.CONST, name="c2", shape=(), dtype="float32", attrs={"value": 2.0})
    add = g.add_node(op_type=OpType.ADD, name="add", inputs=[c1.id, c2.id], shape=(), dtype="float32")
    g.mark_output(add.id)

    num = constant_fold(g)
    assert num == 1, f"expected 1 folded node, got {num}"

    # After folding there should be only const nodes (the add replaced)
    ops = [n.op_type for n in g.nodes()]
    assert OpType.ADD not in ops
    consts = [n for n in g.nodes() if n.op_type == OpType.CONST]
    assert len(consts) >= 1

    print("  PASS: constant_fold simple")


def test_cse_simple():
    g = Graph()
    x = g.add_node(op_type=OpType.INPUT, name="x", shape=(4,), dtype="float32")
    c = g.add_node(op_type=OpType.CONST, name="one", shape=(), dtype="float32", attrs={"value": 1.0})
    a1 = g.add_node(op_type=OpType.ADD, name="add1", inputs=[x.id, c.id], shape=(4,), dtype="float32")
    a2 = g.add_node(op_type=OpType.ADD, name="add2", inputs=[x.id, c.id], shape=(4,), dtype="float32")
    # two identical adds
    g.mark_output(a2.id)

    removed = cse(g)
    assert removed == 1, f"expected 1 removed by CSE, got {removed}"

    adds = [n for n in g.nodes() if n.op_type == OpType.ADD]
    assert len(adds) == 1
    print("  PASS: cse simple")


def _build_elementwise_chain(n_ops: int = 30):
    # build a static graph doing repeated elementwise ops
    sg = StaticGraph()
    inp = sg.input("x", shape=(16, 16), dtype="float32")
    cur = inp
    for i in range(n_ops):
        if i % 2 == 0:
            cur = cur * cur
        else:
            cur = cur + cur
    cur.mark_as_output()
    return sg.graph


def test_pipeline_perf_compare():
    g = _build_elementwise_chain(24)

    # baseline: no pipeline
    eng_plain = JITEngine(passes=None)
    t0 = time.perf_counter()
    mod_plain = eng_plain.compile_graph(g)
    t1 = time.perf_counter()
    run_plain = 0.0
    # run a few times to warm up
    for _ in range(1000):
        t_s = time.perf_counter()
        _ = mod_plain.run(np.random.randn(16, 16).astype(np.float32))
        run_plain += time.perf_counter() - t_s

    # with pipeline
    eng_opt = JITEngine(passes=["pipeline"])
    t2 = time.perf_counter()
    mod_opt = eng_opt.compile_graph(g)
    t3 = time.perf_counter()
    run_opt = 0.0
    for _ in range(1000):
        t_s = time.perf_counter()
        _ = mod_opt.run(np.random.randn(16, 16).astype(np.float32))
        run_opt += time.perf_counter() - t_s

    compile_plain = t1 - t0
    compile_opt = t3 - t2
    print(f"  compile time: plain={compile_plain:.3f}s, opt={compile_opt:.3f}s")
    print(f"  run time (3 iters): plain={run_plain:.4f}s, opt={run_opt:.4f}s")

    # No strict assert: just sanity check modules produced
    assert mod_plain is not None and mod_opt is not None
    print("  PASS: pipeline perf compare (times printed)")


def main():
    test_constant_fold_simple()
    test_cse_simple()
    test_pipeline_perf_compare()


if __name__ == "__main__":
    main()

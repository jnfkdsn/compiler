**整体框架**
现在这个项目最清晰的理解方式，不是按目录扫，而是按“数据怎么流”来理解。稳定主路径已经比较明确了，核心就是：

`Tensor -> TraceContext -> Graph -> optimize_graph -> CppCodegen -> JITEngine/runtime`

对应到代码里：

- 用户入口在 [tensor.py](/home/jnfkdsn/tensor/tensor_cpu/tensor.py) 和 [__init__.py](/home/jnfkdsn/tensor/tensor_cpu/__init__.py)。`Tensor` 同时支持 eager NumPy 执行和 tracing。
- tracing 在 [tracer.py](/home/jnfkdsn/tensor/tensor_cpu/frontend/tracer.py)。它负责把 eager 过程中发生的算子，追加成 `Graph` 里的 `Node`。
- 核心 IR 在 [graph.py](/home/jnfkdsn/tensor/tensor_cpu/ir/graph.py)、[ops.py](/home/jnfkdsn/tensor/tensor_cpu/ir/ops.py)、[shape_inference.py](/home/jnfkdsn/tensor/tensor_cpu/ir/shape_inference.py)。这里是编译器真正的“中间表示”。
- 图优化在 [pipeline.py](/home/jnfkdsn/tensor/tensor_cpu/passes/pipeline.py) 和 `passes/` 下其他文件。当前主 pipeline 很朴素：`const fold -> cse -> fusion -> dce`。
- 后端 codegen 在 [codegen.py](/home/jnfkdsn/tensor/tensor_cpu/backend/codegen.py)。它现在已经拆成了 `shape_solver / memory_planner / op_lowering / cpp_emitter` 这些职责。
- 运行时在 [runtime.py](/home/jnfkdsn/tensor/tensor_cpu/runtime.py)。它负责把生成的 C++ 写到临时文件、调用本地编译器、用 `ctypes` 加载 `.so/.dll`，再把 NumPy array 通过 ABI 描述符喂给 native kernel。
- 自动微分和训练编译在 [vjp.py](/home/jnfkdsn/tensor/tensor_cpu/autodiff/vjp.py)、[train_jit.py](/home/jnfkdsn/tensor/tensor_cpu/autodiff/train_jit.py)、[jit.py](/home/jnfkdsn/tensor/tensor_cpu/nn/jit.py)。这部分不是“核心编译链路”，但它体现了这个项目不只是算子 demo，而是想把训练闭环也打通。

你要特别记住一件事：**主路径是稳定的，但不是全能的**。当前明确限制在 [architecture.md](/home/jnfkdsn/tensor/docs/architecture.md) 和 [execution-path.md](/home/jnfkdsn/tensor/docs/execution-path.md) 里写得很清楚：
- 单输出
- 默认路径不支持 control flow lowering
- 默认 runtime 还是 `ctypes`
- HPC 2D matmul 需要 exact traced shapes

experimental 模块在 [experimental.md](/home/jnfkdsn/tensor/docs/experimental.md) 已经明确隔离了。面试时不要把它们讲成“已经融入主链”。

**你应该怎么学**
如果你的目标是“达到面试要求”，我建议按这个顺序学，不要一上来钻 experimental：

1. 先把主路径跑通一遍  
先读 [README.md](/home/jnfkdsn/tensor/README.md)、[architecture.md](/home/jnfkdsn/tensor/docs/architecture.md)、[execution-path.md](/home/jnfkdsn/tensor/docs/execution-path.md)。  
然后直接看 [test_jit_core.py](/home/jnfkdsn/tensor/examples/test/test_jit_core.py)。这是最好的入口，因为里面把 naive JIT、fusion、HPC、symbolic shape、ABI guard 都串起来了。

2. 再看“图是怎么来的”  
重点读 [tensor.py](/home/jnfkdsn/tensor/tensor_cpu/tensor.py)、[tracer.py](/home/jnfkdsn/tensor/tensor_cpu/frontend/tracer.py)、[graph.py](/home/jnfkdsn/tensor/tensor_cpu/ir/graph.py)。  
你必须能讲清楚：
- eager 和 trace 是怎么共存的
- 一个 `Tensor` op 为什么既会做 NumPy 计算，又会往 Graph 里加节点
- `Node` 里为什么要保存 `shape/rank/numel/strides/attrs`

3. 再看“图怎么被优化和编译”  
重点读 [pipeline.py](/home/jnfkdsn/tensor/tensor_cpu/passes/pipeline.py)、[fusion.py](/home/jnfkdsn/tensor/tensor_cpu/passes/fusion.py)、[codegen.py](/home/jnfkdsn/tensor/tensor_cpu/backend/codegen.py)、[runtime.py](/home/jnfkdsn/tensor/tensor_cpu/runtime.py)。  
这里你要能说清：
- 哪些优化是 graph-level 的
- 哪些信息在 codegen 阶段才需要
- 为什么 runtime 要做 rank/shape guard
- 为什么 HPC matmul 不能随便复用动态 shape

4. 最后看“这个项目不只是前端图，而是带训练链路”  
读 [vjp.py](/home/jnfkdsn/tensor/tensor_cpu/autodiff/vjp.py)、[train_jit.py](/home/jnfkdsn/tensor/tensor_cpu/autodiff/train_jit.py)、[jit.py](/home/jnfkdsn/tensor/tensor_cpu/nn/jit.py)、[test_training.py](/home/jnfkdsn/tensor/examples/test/test_training.py)。  
你不一定要把每个梯度公式背下来，但要能解释：
- 为什么 `vjp.py` 是单一梯度真源
- eager backward 和 graph-mode backward 是怎么共享规则的
- `JITTrainer` 编译的是哪几块内容

**最有效的学习动作**
不要只读代码，最好边看边做这 4 件事：

- 跑 [test_jit_core.py](/home/jnfkdsn/tensor/examples/test/test_jit_core.py)，把每个 test 对应到主路径的一个阶段。
- 自己 trace 一个小图，比如 `relu(x @ w + b)`，打印 `graph.to_debug_string()`，再看生成的 C++。
- 用 benchmark 跑一次 `--quick --experiments`，理解 `HPC on/off`、`passes on/off`、`memory planner on/off` 在测什么，入口在 [run_bench.py](/home/jnfkdsn/tensor/examples/benchmarks/run_bench.py)。
- 看一遍 [test_passes.py](/home/jnfkdsn/tensor/examples/test/test_passes.py)，理解 pass 前后图和性能的差异。

**什么叫“达到面试要求”**
如果你是拿这个项目去面试，我认为最低要做到这几件事：

- 能用 2-3 分钟讲清主路径和每层职责。
- 能解释一次完整例子：从 `Tensor` 表达式到 `Graph`，再到 pass，再到 C++ kernel，再到 runtime 执行。
- 能明确说出当前限制，而不是过度包装。
- 能讲一个“正确性保护”的故事，比如 rank guard、exact traced shape guard、benchmark 的 NumPy reference 校验。
- 能讲一个“性能/工程化”的故事，比如 fusion、memory planner、HPC matmul、benchmark baseline、CI smoke。

**面试官最可能追问你的点**
你最好提前准备这 6 个问题：

- tracing 和 eager 为什么能放在同一个 `Tensor` API 里？
- `Graph` 为什么选择这么轻的 IR，而不是一开始就上更复杂的多层 IR？
- `optimize_graph` 现在做的优化为什么是这几种？
- codegen 为什么拆成 `shape solver / memory planner / lowering / emitter`？
- runtime 为什么先用 `ctypes`，它的局限是什么？
- 你为什么把 control flow / layered backend / pybind 放在 experimental，而不是直接并入 stable？

**你现在不要花太多时间的地方**
如果目标是先过面试，先别把精力主要放在这些地方：
- `experimental/control_flow.py`
- `experimental/tracing_v2.py`
- `experimental/runtime_pybind.py`
- 还没接入默认执行路径的 layered/backend 试验代码

这些东西更适合你后面讲“未来演进方向”，不适合当你当前能力陈述的主线。

**一句话学习路线**
先吃透 stable path，再补 autodiff/training，再准备性能和限制的陈述，最后把 experimental 当作“下一步计划”而不是“当前已完成能力”。

如果你愿意，我下一步可以直接给你做一份“面试准备清单”，按“3 天速成版”列出每天该看哪些文件、该跑哪些测试、该准备哪些回答。
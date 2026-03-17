**你要看的关键文件与函数**

0. Dispatcher 统一调度层与 JIT matmul 解耦
- `examples/tensor_cpu/dispatcher.py` — 线程安全 trace 状态（`threading.local()`）、可插拔 eager kernel 注册（`set_eager_binary/reset_eager_binary`）、集中式梯度规则（`BINARY_GRAD/UNARY_GRAD`）
- `examples/tensor_cpu/jit_matmul.py` — JIT matmul 编译与缓存，通过 dispatcher 插件机制挂载，前端 tensor.py 完全无感知

1. 反向图构建与代码生成
- `examples/tensor_cpu/train_jit.py:92` `build_backward_graph(...)`
- `examples/tensor_cpu/train_jit.py:374` `compile_training_step(...)`
- 新增反向所需算子:
  - `examples/tensor_cpu/ops.py:19` `BROADCAST_TO`
  - `examples/tensor_cpu/ops.py:21` `RELU_GRAD`
- 对应 codegen:
  - `examples/tensor_cpu/codegen.py:279` `_emit_relu_grad(...)`
  - `examples/tensor_cpu/codegen.py:299` `_emit_broadcast_to(...)`

2. Optimizer update kernel 化
- SGD kernel 编译:
  - `examples/tensor_cpu/train_jit.py:281` `compile_sgd_update_kernel(...)`
- Adam kernels 编译:
  - `examples/tensor_cpu/train_jit.py:319` `compile_adam_update_kernels(...)`
- 训练 step 封装:
  - `examples/tensor_cpu/train_jit.py:363` `CompiledTrainingStep`

3. 内存规划与生命周期管理
- Codegen 增加内存规划开关与复用逻辑:
  - `examples/tensor_cpu/codegen.py:54` `CppCodegen(..., enable_memory_planner=True)`
  - `examples/tensor_cpu/codegen.py:129` `_compute_use_count(...)`
  - `examples/tensor_cpu/codegen.py:90`/`111` 处在生成阶段做 temp buffer 复用
- Runtime 透传开关:
  - `examples/tensor_cpu/runtime.py:131` `JITEngine(..., enable_memory_planner=True)`
  - `examples/tensor_cpu/runtime.py:139` 传给 `CppCodegen`

4. 包导出
- `examples/tensor_cpu/__init__.py:8` 导出 `compile_training_step` / `compile_sgd_update_kernel` / `compile_adam_update_kernels`

5. 全图训练 JIT 示例
- `examples/train_full_jit_linear_demo.py:46` `compile_training_step(...)`
- `examples/train_full_jit_linear_demo.py:52` `compile_sgd_update_kernel(...)`
- `examples/train_full_jit_linear_demo.py:70` 成功标志输出

---

**补充说明（当前实现边界）**
- 当前 backward graph 构建器要求损失是**标量输出**。
- 已覆盖你当前训练链路常见算子（add/sub/mul/div/matmul/transpose/relu/exp/log/sigmoid/sum/mean）的反向。
- 这是可运行的“全图训练 JIT 基础版”，下一步可以继续把它接到 attention 全流程（把 attention loss 前向图直接交给 `compile_training_step`，并用 Adam kernels 驱动参数更新）。

Made changes.


推荐你按“先会用，再懂实现，最后看优化和训练编译”的顺序学。

1. train_nn_demo.py
先看最小训练闭环，知道框架怎么被使用。

2. tensor.py
看 `Tensor` 的 eager 运算和 `backward()`。重构后算子方法极度精简（每个仅 1-2 行），核心逻辑在 `_dispatch_binary/_dispatch_unary` 中。

3. tracer.py
理解 `TraceContext`，知道图是怎么被记录的。现在 tracer 委托 dispatcher 管理 trace 状态。

3.5. dispatcher.py
理解统一调度层：线程安全 trace 状态、eager kernel 注册表、梯度规则集中管理。这是重构后的架构核心。

4. graph.py
理解 `Graph/Node` 数据结构，打通 IR 心智模型。

5. shape_inference.py
看 shape/dtype 规则（已支持 float64、N-D transpose、batch matmul），理解为什么图构建阶段就能发现错误。

6. milestone2_jit_demo.py
看最小 JIT 流程（trace -> compile -> run）。

7. codegen.py
看图到 C++ 的映射逻辑（先看 `_emit_add/_emit_matmul`，再看 `_render_cpp`）。

8. runtime.py
看 kernel.cpp 写入、编译、`ctypes` 调用和 ABI 校验。

9. abi.py
看错误码体系，理解 runtime 报错如何定位。

10. fusion.py
11. dce.py
12. pipeline.py
这三步看中端优化如何落地。

13. milestone3_fusion_demo.py
把前面 pass 和运行结果关联起来。

14. milestone4_hpc_benchmark.py
看性能侧入口，再回头看 codegen.py 里的 HPC matmul 分支。

15. train_complex_nn_demo.py
看 `BatchNorm/LayerNorm/Dropout` 在真实训练中的用法。

16. train_attention_text_demo.py
看 attention 文本任务是怎么接入现有框架的。

17. train_jit.py
重点看训练期编译：backward 图构建、optimizer kernel 化、`compile_training_step`。

18. train_full_jit_linear_demo.py
验证“全图训练 JIT”完整闭环。

19. jit.py
看高层自动编译训练接口 `JITTrainer` 和 `LazyJITTrainer`。

20. train_nn_auto_jit_demo.py
21. train_nn_lazy_jit_demo.py
对比自动编译和惰性+缓存执行的体验。

22. lazy.py
23. lazy_eval_demo.py
专门学习惰性求值语义。

24. static_graph.py
25. static_graph_demo.py
26. train_static_graph_linear_demo.py
最后学严格静态图路线（define -> compile -> run -> train）。

如果你愿意，我可以再给你做一版“7天学习计划”（每天看哪些文件、跑哪些脚本、做哪些改动练习）。





## 项目学习路线

本项目是一个**纯 CPU 动态 AI 编译器**，从 Python 前端到 C++ JIT 后端完整实现。以下按依赖关系和复杂度递进排列，分 6 个阶段。

---

### 阶段一：基础概念 — IR 数据结构

**目标**：理解编译器中间表示（IR）的核心抽象——"操作类型"和"计算图"

| 顺序 | 文件 | 学什么 |
|---|---|---|
| 1 | ops.py | `OpType` 枚举，了解所有支持的算子（算术、矩阵乘、激活、归约、融合类型） |
| 2 | graph.py | `Node` dataclass（id/op_type/inputs/shape/dtype/attrs）和 `Graph` 类（添加节点、拓扑排序、节点替换、输出标记） |
| 3 | shape_inference.py | 静态形状推断引擎：广播规则、matmul 形状推导、归约维度计算、dtype 提升 |

**达到目标**：能手动构建一个 `Graph`，添加 INPUT → MATMUL → ADD → RELU 节点链，调用 `topological_sort()` 得到正确执行序列。

---

### 阶段二：前端图捕获 — Tracer 与 Tensor

**目标**：理解如何用 Python 运算符重载自动捕获计算图

| 顺序 | 文件 | 学什么 |
|---|---|---|
| 4 | tracer.py | `TraceContext` 上下文管理器，线程本地 trace 状态，`__enter__`/`__exit__` 生命周期 |
| 5 | dispatcher.py | 双模式分发器：eager 模式走 NumPy 计算，trace 模式走图节点构建；线程本地状态管理 |
| 6 | tensor.py | `Tensor` 类核心：`from_numpy()`、`__add__`/`__matmul__` 等魔术方法、`_dispatch_binary`/`_dispatch_unary`/`_dispatch_reduce` 分发逻辑、`backward()` 反向传播 |

**达到目标**：能写出以下代码并理解内部每一步发生了什么：

```python
with TraceContext() as tc:
    x = Tensor.from_numpy(x_np, name="x")
    w = Tensor.from_numpy(w_np, name="w")
    out = (x @ w).relu().mark_as_output()
    graph = tc.graph  # 捕获到完整计算图
```

**验证**：运行 test/test_jit_core.py 中的 `test_naive_jit` 前半部分（图构建部分）。

---

### 阶段三：后端编译 — CodeGen + JIT Runtime

**目标**：理解图 → C++ 源码 → 动态库 → ctypes 调用的完整链路

| 顺序 | 文件 | 学什么 |
|---|---|---|
| 7 | abi.py | ABI 状态码定义，内核调用的错误校验约定 |
| 8 | codegen.py | `CppCodegen` 类：遍历拓扑序生成 C++ 代码；naive 循环模板 vs HPC 分块模板（loop tiling + OpenMP + SIMD）；符号形状参数化 |
| 9 | runtime.py | `JITEngine`：调用 g++/clang++ 编译 .so；`JITModule`：ctypes 加载动态库、ABI 校验、zero-copy 数据交互；内存规划器 |

**达到目标**：能解释一张计算图如何变成 C++ 代码，编译出 .so 文件，再通过 ctypes 传入 NumPy 数组获取计算结果。

**验证**：完整运行 test/test_jit_core.py（6 个测试全部通过）。

---

### 阶段四：图优化 — Pass 系统

**目标**：理解编译器优化 pass 如何变换计算图

| 顺序 | 文件 | 学什么 |
|---|---|---|
| 10 | passes/dce.py | 死代码消除：从输出节点反向可达性分析，删除无用节点 |
| 11 | passes/fusion.py | 算子融合：模式匹配 `MATMUL+ADD`→`FUSED_MATMUL_BIAS`、`MATMUL+ADD+RELU`→`FUSED_MATMUL_BIAS_RELU`；节点替换 |
| 12 | passes/pipeline.py | 优化管线编排：`optimize_graph()` 先 fusion 再 DCE，返回统计信息 |

**达到目标**：能解释融合 pass 如何将 3 个节点合并为 1 个，为什么这能减少内存带宽开销。

**验证**：运行 `test_fusion`，确认 `fused_subgraphs >= 1`。

---

### 阶段五：自动微分 — VJP + 训练 JIT

**目标**：理解 reverse-mode AD 的两种实现（eager NumPy 路径 + graph-level 符号路径）

| 顺序 | 文件 | 学什么 |
|---|---|---|
| 13 | vjp.py | **重点文件**。`VJPRule(eager, graph)` 统一注册表；每个算子的前向/反向数学推导（如 $\frac{\partial}{\partial A}(AB) = G B^T$）；`apply_vjp()` 用于图级反向 |
| 14 | train_jit.py | `build_backward_graph()` 从前向图构建反向图；`compile_training_step()` 编译前向+反向+优化器内核；`compile_sgd_update_kernel()` / `compile_adam_update_kernels()` |

**达到目标**：能解释 `loss.backward()` 走 eager 路径时如何调用 `_VJP_REGISTRY[op].eager`，以及 `build_backward_graph()` 走 graph 路径时如何调用 `_VJP_REGISTRY[op].graph` 构建符号反向图。

**验证**：运行 test/test_training.py 中的 `test_eager_backward`（含数值梯度校验）和 `test_full_jit_training_step`。

---

### 阶段六：上层框架 — nn / optim / 高级 API

**目标**：理解类 PyTorch 的上层抽象如何构建在编译器之上

| 顺序 | 文件 | 学什么 |
|---|---|---|
| 15 | nn/modules.py | `Module` 基类（参数收集、zero_grad、train/eval）；`Linear`/`ReLU`/`BatchNorm1d`/`LayerNorm`/`Dropout`/`SelfAttention` 实现；`mse_loss` |
| 16 | optim/sgd.py | SGD 实现：动量、Nesterov、weight decay |
| 17 | optim/adam.py | Adam 实现：一阶/二阶矩估计、偏差校正 |
| 18 | nn/jit.py | `JITTrainer`：首次 step 自动 trace+compile，后续复用编译内核；`LazyJITTrainer`：按 rank 缓存编译结果 |
| 19 | lazy.py | `LazyTensor`：thunk 延迟求值，`eval()` 触发物化 |
| 20 | static_graph.py | `StaticGraph` + `SymbolicTensor`：严格的先定义后编译范式 |
| 21 | jit/api.py | `jit.trace()` 高层 API：对函数或 Module 进行 trace 并编译为 C++ |
| 22 | jit_matmul.py | 可插拔 JIT matmul：替换 NumPy 的 `@` 运算符为 JIT 编译的 C++ 内核 |

**达到目标**：能用本框架完成一个完整的训练流程——定义模型、选择优化器、执行训练循环、观察 loss 下降。理解 eager/JIT/lazy 三种执行模式的区别与适用场景。

**验证**：运行 test/test_training.py（7 个测试）和 test/test_advanced.py（9 个测试）全部通过。

---

### 总览依赖图

```
ops.py ──► graph.py ──► shape_inference.py
                │
        tracer.py ◄── dispatcher.py ◄── vjp.py
                │
            tensor.py
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
codegen.py  runtime.py   abi.py
    │           │
    ▼           ▼
passes/     train_jit.py
    │           │
    ▼           ▼
nn/modules ── nn/jit ── optim/
    │
    ▼
lazy.py ── static_graph.py ── jit/api.py
```

核心学习路径就是沿着这个依赖图**自底向上**：先掌握 IR 数据结构，再理解图捕获，然后是编译后端，接着是优化 pass，再到自动微分，最后是上层框架。共 22 个文件，3 个测试文件作为每阶段的验证工具。
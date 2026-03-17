# Tensor CPU AI Compiler 详细实现手册（小白友好版）

本文档面向第一次接触“深度学习框架 + 编译器”的读者，目标是讲清楚三件事：

1. 这个项目到底在做什么。
2. 一段 Python 训练代码是怎样一步步走到 C++ 执行的。
3. 当前实现到了什么程度，和工业级方案还差什么。

如果你只记一句话：
这个项目不是简单“调用 NumPy 训练”，而是在构建一条可扩展的编译执行链路：
`Tensor API -> Graph IR -> Pass 优化 -> C++ CodeGen -> JIT Runtime -> (训练期) 反向图与优化器内核`。

---

## 1. 项目全景图

你可以把它想象成一条流水线。

1. 前端（Frontend）
- 你写 `x @ w + b`、`relu()`、`mean()` 这样的 Python 代码。
- 框架用 `Tensor` 封装数据和运算。

2. 图层（IR）
- 在 tracing 模式下，运算会被记录成一张有向无环图（Graph）。
- 图里每个节点是 `Node`，有 `op_type/shape/dtype/inputs` 等元信息。

3. 中端优化（Pass）
- 在图级别做优化，比如融合和死代码消除。

4. 后端（CodeGen）
- 把图翻译成 C++ kernel 源码。
- 支持普通循环、融合算子、HPC matmul（OpenMP + AVX）。

5. 运行时（Runtime）
- 调用本地编译器（`g++/clang++`）编译成 `.so`。
- 用 `ctypes` 加载并执行 `run_kernel(...)`。

关键目录：
- `examples/tensor_cpu/dispatcher.py` — 统一调度层（trace 状态、eager kernel、梯度规则）
- `examples/tensor_cpu/tensor.py` — 前端 Tensor API
- `examples/tensor_cpu/tracer.py` — 图追踪上下文
- `examples/tensor_cpu/graph.py` — 图 IR 数据结构
- `examples/tensor_cpu/jit_matmul.py` — JIT matmul 编译与缓存
- `examples/tensor_cpu/passes/*` — 图优化 pass
- `examples/tensor_cpu/codegen.py` — C++ 代码生成
- `examples/tensor_cpu/runtime.py` — JIT 编译运行时

---

## 2. 前端：Dispatcher、Tensor 与 Autograd 是怎么工作的

### 2.1 Dispatcher：统一调度层（新架构核心）

在重构后，项目引入了 `dispatcher.py` 作为统一调度中心，解决了旧架构的三大问题：

1. **线程安全**：trace 状态使用 `threading.local()` 存储，替代了旧的全局单例 `_TRACE_STATE`，多线程并发 trace 不会互相干扰。
2. **Eager 分发**：所有基础运算（add/mul/matmul 等）的 NumPy 实现注册在 dispatcher 中，`tensor.py` 不再硬编码具体实现。JIT matmul 通过 `set_eager_binary(OpType.MATMUL, ...)` 插入 dispatcher，tensor.py 完全不知道编译器的存在。
3. **梯度规则统一**：`BINARY_GRAD` / `UNARY_GRAD` 字典是 eager 反向传播的唯一数据源，避免了之前每个算子方法里手写闭包求导的代码膨胀。

关键文件：`examples/tensor_cpu/dispatcher.py`

### 2.2 Tensor 的职责（精简后）

重构后 `Tensor` 的算子方法极度精简，每个算子只有 1-2 行：

```python
def __add__(self, other):
    return _dispatch_binary(OpType.ADD, self, self._ensure_tensor(other))
```

内部的 `_dispatch_binary` / `_dispatch_unary` 统一完成三件事：
1. 调 `dispatcher.eager_binary(op_type, ...)` 执行 NumPy 计算
2. 若 tracing，调 `add_binary_node(...)` 记录图节点
3. 从 `dispatcher.binary_grad_rule(op_type)` 获取梯度规则并绑定 `_backward`

关键文件：`examples/tensor_cpu/tensor.py`

### 2.3 为什么还要 tracing

当 `dispatcher.is_tracing()` 为 `True` 时，`Tensor` 在 eager 计算的同时，还会往图里加节点。

例如：
- `__matmul__` -> `add_binary_node(OpType.MATMUL, ...)`
- `sum(axis=..., keepdims=...)` -> `add_reduce_node(...)`
- `transpose()` -> `add_transpose_node(...)`

这样就把 Python 运算"录制"成了 Graph IR，为后续编译做准备。

关键文件：
- `examples/tensor_cpu/tracer.py`（委托 dispatcher 管理 trace 状态）
- `examples/tensor_cpu/tensor.py`

## 3. IR：图结构、shape 推导、为什么重要

### 3.1 Graph / Node 元数据

`Node` 的核心字段：
- `op_type`: 算子类型
- `inputs`: 输入节点 id
- `shape/dtype`
- `attrs`: 算子属性（如 reduce 的 axis/keepdims）
- `rank/numel/strides`: 连续内存布局信息

关键文件：`examples/tensor_cpu/graph.py`

### 3.2 Shape Inference 做什么

图构建时就检查并推导 shape，避免把错误拖到 C++ 才暴露。

当前支持：
- 二元算子（含 broadcast、**N-D batch matmul**）
- 一元算子（包括 **N-D transpose**，不再限制 2D）
- 归约算子（sum/mean 的 axis/keepdims）
- dtype 提升：支持 `float32` 和 `float64`，混合运算自动提升到 float64

关键文件：`examples/tensor_cpu/shape_inference.py`

---

## 4. Pass：图级优化

当前中端优化包括：

1. 融合
- `MatMul + Add (+ ReLU)` -> 融合节点
- 减少中间张量和 kernel 调度开销

2. DCE（死代码消除）
- 从输出反向可达分析，删除无用节点

关键文件：
- `examples/tensor_cpu/passes/fusion.py`
- `examples/tensor_cpu/passes/dce.py`
- `examples/tensor_cpu/passes/pipeline.py`

---

## 5. CodeGen：从图到 C++

### 5.1 生成逻辑

`CppCodegen.generate()` 会：

1. 拓扑排序节点
2. 为输入、常量、中间 buffer 分配名字
3. 按节点类型调用 `_emit_*` 生成 C++ 代码片段
4. 用 `_render_cpp(...)` 生成完整 `run_kernel(...)`

关键文件：`examples/tensor_cpu/codegen.py`

### 5.2 结构化 codegen（你最近做的升级）

现在不是纯粹拼大字符串，而是：
- 先构建结构化语句（`CppLine`、`CppFor`）
- 再统一渲染成 C++ 文本

这让后续做 loop transform 更容易。

### 5.3 内存规划（生命周期复用）

`enable_memory_planner=True` 时：
- 先统计每个临时节点被使用次数
- 当节点最后一次被消费后，把它的 buffer slot 放回 free-list
- 新节点优先复用同 numel 的空闲 slot

这是一个简化版 memory planner，已经能降低临时内存占用。

关键位置：`examples/tensor_cpu/codegen.py`

---

## 6. Runtime：C++ 编译与调用

`JITEngine.compile_graph(graph)` 做了以下事：

1. 调 `CppCodegen.generate()` 拿到 `kernel.source`
2. 写入临时目录 `kernel.cpp`
3. 调本地编译器生成动态库（Linux 通常是 `libkernel.so`）
4. `ctypes.CDLL` 加载动态库
5. 绑定 `run_kernel` 函数签名

执行时：
- `JITModule.run(*inputs)` 把 `numpy` 数组打包成 `TensorDesc`
- 调 C++ `run_kernel`
- 校验 ABI 状态码，不为 0 则抛可读错误

关键文件：`examples/tensor_cpu/runtime.py`

---

## 7. ABI：为什么要这么严格

`TensorDesc` 包含：
- `data`
- `numel`
- `rank`
- `shape[kMaxRank]`
- `strides[kMaxRank]`

在 kernel 入口会校验输入输出：
- 输入数量、shape、stride、rank、numel
- 输出描述符合法性

这样做的意义：
- 出错更早、更可定位
- 避免 silent wrong result

关键文件：
- `examples/tensor_cpu/abi.py`
- `examples/tensor_cpu/runtime.py`
- `examples/tensor_cpu/codegen.py`（生成的 guard 代码）

---

## 8. 训练期 JIT：从“只前向编译”到“全图训练编译”

你目前有三层训练执行方式。

### 8.1 纯 eager 训练

- 直接 `Tensor` 运算 + `backward()` + Python optimizer
- 优点：简单
- 缺点：热算子不一定最快

### 8.2 算子级 JIT 训练

- 以 `matmul` 为例，训练时首次遇到某 shape 编译一次，后续缓存复用
- 这是“热路径加速”的渐进方案
- JIT matmul 已从 tensor.py 中解耦，独立为 `jit_matmul.py`
- 通过 dispatcher 的 `set_eager_binary` 机制插入，tensor.py 完全不知道编译细节

关键文件：`examples/tensor_cpu/jit_matmul.py`

### 8.3 全图训练 JIT（基础版）

你已经实现：

1. 反向图构建
- `build_backward_graph(...)` 对 forward graph 做 reverse-mode 自动微分
- 为参数生成梯度图并编译

2. 优化器 update kernel 化
- SGD update kernel
- Adam 的 `m/v/param` 更新 kernels

3. 训练 step 封装
- `compile_training_step(...)` 返回可执行的 loss + grad compiled modules

关键文件：`examples/tensor_cpu/train_jit.py`

示例：
- `examples/train_full_jit_linear_demo.py`

---

## 9. nn 层自动编译：像 PyTorch 一样调用 Module

你已经实现了 `nn.JITTrainer` 与 `nn.LazyJITTrainer`，让用户不用手写 compile 细节。

使用方式：

```python
trainer = nn.JITTrainer(model, loss_fn=nn.mse_loss, optimizer="sgd", lr=5e-2)
for step in range(200):
    loss = trainer.step(x, y)
```

内部发生了什么：
- 第一次 `step` 自动 trace 并编译 forward/loss/backward/update kernels
- 后续 `step` 直接走编译结果
- 参数更新自动写回 `model` 参数

`nn.LazyJITTrainer` 进一步支持：
- 惰性构图 + 首次自动编译
- 以输入 shape 为 key 的编译缓存复用
- 新 shape 自动新增编译缓存条目

关键文件：
- `examples/tensor_cpu/nn/jit.py`
- `examples/train_nn_auto_jit_demo.py`
- `examples/train_nn_lazy_jit_demo.py`

---

## 10. Attention 与文本任务在本项目中的位置

当前 attention demo 主要证明：

1. Attention 前向图可被 trace + JIT 编译执行
- 包括 `matmul`、`transpose`、`softmax(组合实现)` 等

2. 训练可收敛
- 位置编码加入后，token accuracy 显著提升

关键文件：`examples/train_attention_text_demo.py`

注意：
- 当前文本生成效果仍是 toy 级别，不代表工业级 LLM 训练器。

---

## 11. 为什么不是“直接 NumPy 就行”

这是初学者最常见问题。

确实：
- 如果只求“把模型跑起来”，NumPy 足够。

但这个项目目标是：
- 搭建可控的编译器链路
- 在图级做优化和后端替换
- 把训练与执行语义从 Python 解释层迁移到编译内核

NumPy 是数值库，不是编译器中间表示系统。
你现在做的是“框架 + 编译后端”的基础设施。

---

## 12. 当前能力边界（实话实说）

已经完成：
- 前后端闭环
- **Dispatcher 统一调度层**：线程安全的 trace 状态、可插拔 eager kernel、集中式梯度规则
- **JIT matmul 解耦**：编译器后端细节与前端 Tensor API 完全分离
- JIT runtime + ABI
- 基础图优化
- 训练期 JIT 基础版
- nn 自动编译训练入口
- **放宽类型系统**：shape inference 支持 float64、N-D transpose、batch matmul

还未工业级：
- 动态 shape 多版本缓存策略还较基础
- backward 覆盖算子仍在扩展中
- 调度/并行策略和成本模型较简化
- 缺少大规模 benchmark 与 profiling 基建
- 缺少生产级错误恢复与长生命周期内存池

---

## 13. 推荐阅读顺序（给小白）

1. 先看 `examples/train_nn_demo.py`
- 理解 Tensor + nn + optim 的最小训练闭环

2. 再看 `examples/milestone2_jit_demo.py`
- 理解 tracing 到 JIT 的基础路径

3. 再看 `examples/tensor_cpu/runtime.py`
- 理解 C++ 编译和 ctypes 调用

4. 再看 `examples/tensor_cpu/codegen.py`
- 理解图如何翻译为 C++

5. 再看 `examples/tensor_cpu/train_jit.py`
- 理解训练全图 JIT 的核心思想

6. 最后看 `examples/train_nn_auto_jit_demo.py`
- 理解“像 PyTorch 一样调用 Module，但内部自动编译”

---

## 14. 一句话总结

这个项目已经从“玩具张量库”跨过门槛，进入“可演进的编译器型训练框架原型”：
有 IR、有 Pass、有 CodeGen、有 Runtime、有 ABI、有训练期 JIT、有 nn 自动编译入口。

它还不是工业级终态，但路线是对的，而且实现深度在简历与面试中已经有很强说服力。

---

## 15. 新增：完整惰性求值（Lazy Evaluation）

为了解决“算子一调用就立刻执行”的 eager 限制，项目新增了独立惰性层：

- 文件：`examples/tensor_cpu/lazy.py`
- 核心类型：`LazyTensor`

### 15.1 它怎么工作

`LazyTensor` 不在每一步立即计算，而是把计算包装成 thunk（延迟函数）。

例如：
- `pred = (x @ w) + b`
- `loss = lazy_mse_loss(pred, y)`

上面两行只是在“拼表达式”，还没有真正执行数值运算。

真正执行发生在以下时刻：
- 调用 `loss.eval()`
- 访问 `loss.data`
- 调用 `loss.backward()`

这就是完整的惰性求值语义：表达式先构建、再按需物化。

### 15.2 当前支持能力

`LazyTensor` 已支持常见算子：
- 二元：`add/sub/mul/div/matmul`
- 一元：`relu/exp/log/sigmoid/transpose`
- 归约：`sum/mean`
- 组合：`softmax`

并提供惰性损失函数：
- `lazy_mse_loss`
- `lazy_binary_cross_entropy`

### 15.3 示例

完整示例：`examples/lazy_eval_demo.py`

示例输出会先打印：
- `lazy graph built; not evaluated yet`

然后在首次访问 `loss.data` / `backward()` 时才实际计算，证明惰性语义生效。

---

## 16. 新增：严格静态图前端与纯静态图训练

除了动态图 + tracing 方案，项目新增了严格静态图前端：

- 文件：`examples/tensor_cpu/static_graph.py`
- 核心类型：`StaticGraph`、`SymbolicTensor`

### 16.1 严格静态图语义

`SymbolicTensor` 不持有 eager 数值结果，只代表图中的符号节点。

流程固定为：
1. `input(...)` 定义输入
2. 通过算子构图（只记录，不执行）
3. `mark_as_output()` 指定输出
4. `compile()` 编译
5. `run()` 执行

这就是典型 `define-and-run`（先定义图，再运行）。

### 16.2 纯静态图训练 demo

新增示例：`examples/train_static_graph_linear_demo.py`

它展示了完整静态图训练闭环：
- 用 `StaticGraph` 定义 loss 图
- 一次性编译 forward+backward
- 使用编译后的 SGD update kernel 更新参数

示例结果：
- `final_mse=0.000398`
- `Pure static-graph training demo passed.`

这意味着项目现在同时支持：
- 动态图前端（eager + trace）
- 惰性前端（LazyTensor）
- 严格静态图前端（StaticGraph）

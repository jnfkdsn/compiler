AI 编译器是一个跨越深度学习算法、传统编译原理和底层计算机体系结构的交叉领域。面试官非常喜欢从你的自研项目出发，逐渐由浅入深，扩展到工业界主流框架（如 PyTorch 2.0、TVM、XLA、MLIR）的设计理念上。

为了应对这种“扩展提问”，我为你总结了 **AI 编译器核心知识点**，并制定了一份 **进阶学习计划**。

---

### 一、 面试常考的 AI 编译器核心知识点

AI 编译器通常分为 **前端（Frontend）、中端（Graph IR & Passes）、后端（CodeGen & 硬件内核）**。你需要掌握以下概念：

#### 1. 前端与 IR（中间表示）
*   **计算图的获取方式**：
    *   **Tracing（符号追踪）**：像你的项目，通过跑一遍样例数据把路径记录下来（缺点：控制流 `if/else` 容易丢失）。
    *   **AST 解析（抽象语法树）**：像 TorchScript，直接解析 Python 代码语法，保留控制流，但对 Python 动态特性支持差。
    *   **Bytecode 拦截（字节码层）**：像 TorchDynamo（PyTorch 2.0 的核心），在 Python 虚拟机层面拦截字节码编译，兼顾动态性和捕获率（**面试必考**）。
*   **多层 IR 设计**：为什么工业界不用一棵树走到底？（参考 MLIR 的 Dialect 思想，分为 High-level Graph IR 和 Low-level Loop IR，方便在不同层级做不同优化）。

#### 2. 中端图优化（Graph-level Optimization）
你的项目做了简易的 Fusion 和 DCE，面试官可能还会问：
*   **常见的代数化简（Algebraic Simplification）**：比如 $x \times 0 \rightarrow 0$，$x \times 1 \rightarrow x$，常量折叠（Constant Folding）。
*   **算子融合（Operator Fusion）的分类**：
    *   **Element-wise 融合**：如 `Add + ReLU`。
    *   **Reduce 融合**：如 `MatMul + Bias + ReLU`。
    *   **Horizontal 融合**：把没有依赖关系、且输入相同的小算子合并成一个大算子（如 Transformer 中的 Q, K, V 矩阵乘法合并）。
*   **内存规划（Memory Planning）**：除了你的 Buffer 复用，工业界如何做 Liveness Analysis（活跃变量分析），以及 In-place 更新（如 `x = x + 1` 直接覆盖）。

#### 3. 后端内核优化（Kernel-level Optimization）与 CodeGen
*   **Loop 变换（Loop Transformations）**：
    *   **Tiling（分块）**：解决 Cache Line 命中率问题，把大矩阵切成小块放入 L1/L2 缓存。
    *   **Unrolling（展开）**：减少 for 循环跳转开销，增加指令级并行。
    *   **Vectorization（向量化）**：你的 AVX/SIMD 优化。
*   **Auto-tuning（自动调优）**：像 TVM 的 AutoTVM/Ansor。工业界如何不靠人手写，而是靠遗传算法或机器学习模型，在巨大的搜索空间（不同 Tiling size、不同展开参数）中搜出最快的那份代码？

#### 4. 动态 Shape 难题 (Dynamic Shape)
*   **为什么动态 Shape 是 AI 编译器的噩梦？**
    你的项目中使用了基于 Shape 的 Cache（同一个 Shape 编译一次）。如果 batch_size 不断变化（如 NLP 中句子长度不一），会导致疯狂触发重编译。工业界现在的最新进展（PyTorch 2.0 的 `dynamic_shapes=True`）是如何做符号化 Shape 推导的。

---

### 二、 AI 编译器进阶学习计划（3 个月~半年）

这套计划旨在让你从“手写了小轮子”过渡到“深刻理解工业界大轮子”。

#### 阶段一：夯实算子底层与系统架构 (1-4 周)
*   **目标**：不仅能写 CPU 的 AVX，还要懂 GPU。
*   **学习内容**：
    *   了解 GPU 体系结构（SM、Warp、Shared Memory、Global Memory）。
    *   学习基础 CUDA 编程（不用非常精通，但必须要理解 CUDA 中计算 MatMul 怎么利用 Shared Memory 做 Tiling）。
    *   **进阶**：了解 OpenAI 提出的 **Triton 语言**。Triton 极大降低了写 GPU Kernel 的门槛，它是当前 PyTorch 2.0 (Inductor 后端) 生成 GPU 代码的核心底座。

#### 阶段二：深入 TVM 与传统 AI 编译基建 (第 2 个月)
*   **目标**：理解从 Graph 到 Loop 代码生成的标准范式。
*   **学习内容**：
    *   学习经典论文《TVM: An Automated End-to-End Optimizing Compiler for Deep Learning》。
    *   了解 **Halide** 语言的核心思想：**算法（Algorithm）与调度（Schedule）分离**。即“算什么”和“怎么算在硬件上最快”解耦。
    *   推荐用 TVM Python API 跑几个小例子，看看它的 `te.compute`（Tensor Expression）是怎么一句话定义计算，并用 `te.schedule` 切分运算的。

#### 阶段三：理解现代编译基础设施 MLIR (第 3 个月)
*   **目标**：MLIR 是当前各大厂（Google、苹果、字节等）自研 AI 芯片/编译器的基石。
*   **学习内容**：
    *   了解 MLIR 的核心：Dialect（方言）。理解它是如何让高层大图（Tensor 级）一步步 lowering 到低层（LLVM IR）。
    *   重点看 `Linalg` 方言（处理线性代数和循环）和 `Vector` 方言。

#### 阶段四：跟上最前沿 PyTorch 2.0 架构 (第 4 个月)
*   **目标**：理解大厂现状，能在面试中与面试官谈笑风生。
*   **学习内容**：
    *   研究 `TorchDynamo`：它是怎么无缝拦截 Python 并获取计算图的？和 `TorchScript` 有什么区别？
    *   研究 `TorchInductor`：它是如何作为后端，把小算子融合成 Triton 脚本（GPU）或者 C++ OpenMP 脚本（CPU）的。

---

### 三、 推荐学习资源（黄金推荐）

1.  **必看公开课**：陈天奇博士（TVM/XGBoost作者）的 **《Machine Learning Compilation》(MLC 机器学习编译)**。
    *   这是目前全网最好的 AI 编译器入门到精通课程。直接搜 `MLC 陈天奇`。这门课会带着你用 Python 重新认识一遍 Tensor、Loop、Tiling 以及自动代码生成。
2.  **必读文章/文档**：
    *   **《A Roaming Guide to PyTorch 2.0》** 或者国内的 PyTorch 2.0 源码解析博客，弄懂 Dynamo 和 Inductor。
    *   **Triton 官方教程**的 Matmul 篇，理解现代 GPU Kernel 是怎么用 Python 写的。
3.  **书籍/论文**：
    *   《深入理解LLVM》或相关 LLVM 官方教程的 Kaleidoscope 章节（补充传统编译前端 AST 和 Pass 的认识）。

**面试实战建议**：
在面试时，你可以主动抛出这些概念：“在写这个 `tensor_cpu` 的过程时，我遇到了如何处理动态分支的问题。通过调研，我了解到了工业界像 TorchScript 的 AST 解析路线和 TorchDynamo 的 Bytecode 路线；由于精力有限，我的项目采用了轻量级的 Tracing 方案，但也实现了基于 Shape 的 Cache……” 这种回答能让你显得技术视野极大。

需要我帮你把这份知识点和计划整理成一份 Markdown 笔记存进 `memories` 或项目的 `docs` 目录中，方便你随时复习吗？
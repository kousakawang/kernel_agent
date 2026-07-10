# Skill: Triton/CuTe DSL/CUDA Codegen Strategy

此 skill 指导 Kernel Agent 选择实现路径。

## 默认路径

1. Triton prototype/tuning
   - 最快得到可改可测的候选实现。
   - 适合矩阵、attention 子块、elementwise/fusion、layout transform。

2. CuTe DSL prototype/tuning
   - 和 Triton 一样属于首选 JIT 路径。
   - 适合需要更细 tile、向量化读取、shared memory、pipeline 或 MMA 组织控制的 kernel。
   - 不应因为某个现有 SGLang CuTe backend 对特定路径要求 SM100，就默认禁止在 SM90/H20 上写自定义 CuTe DSL kernel。

3. CUDA extension / CUTLASS path
   - 当 Triton/CuTe DSL 达不到目标，或需要非 JIT 集成、特定库调用、PTX/SASS 级控制时使用。
   - 这类路径编译、集成、调试成本更高，Phase 1 中应由 benchmark/profile 证据触发。

## 实现原则

- 先实现正确语义，不急于融合所有路径。
- 明确 prefill/decode 是否需要分开 kernel。
- 对动态 shape 做有限 specialization，不为所有 shape 写复杂分支。
- 对 dtype、layout、stride 的假设写入 `kernel_constraints.md`。
- 若要求 contiguous、padding、metadata 预计算，生成 `FrameworkChangeRequest`。

## 常见优化方向

- 改善 memory coalescing。
- 减少 global memory round trip。
- 融合轻量 epilogue。
- 预计算或压缩 metadata。
- 减少临时 tensor。
- 对 hot shape 做 block/tile/vectorization specialization。
- 避免不必要 dtype cast。

## 不应做的事

- 为了 benchmark 只支持一个随机 seed。
- 依赖输入内容分布作弊。
- 修改 golden 或 tolerance 来制造通过。
- 把框架侧必须知道的限制藏在代码里。

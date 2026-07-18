# Skill: Model Profiling

此 skill 用于 Phase 2：从模型/服务 workload 中定位热点模块和候选算子。第一阶段不实现自动 profiler，只约定采集和解析方法。

## Profile 目标

- 区分 prefill、decode、vision encoder、sampling、通信、调度开销。
- 找出 GPU time 占比最高的 kernel/module。
- 找出 CPU/scheduler 或 memory allocator 造成的非 kernel 瓶颈。
- 给出每个热点是否适合交给 Kernel Agent。

## 推荐采集层次

1. 服务日志：确认 workload、batch、token、cache、cuda graph、吞吐。
2. torch profiler：建立 Python module 到 CUDA kernel 的粗映射。
3. nsys：看 timeline、launch gap、同步、CPU/GPU overlap。
4. ncu：只对候选热点 kernel 做硬件指标分析。

## 输出要求

每个热点需要记录：

- 名称和源码入口。
- 场景：prefill、decode、mixed、vision、communication。
- GPU time 占比和调用频率。
- 关键 shape。
- 现有 backend。
- 是否已有可独立 UT。
- 是否适合 kernel 优化。
- 估计端到端收益上限。

## 不适合交给 Kernel Agent 的情况

- 主要瓶颈是调度、排队、I/O、tokenizer、Python overhead。
- 单 kernel 占比极低，优化上限不足。
- 正确性依赖完整模型状态，短期无法构造独立 UT。
- 需要先做框架级策略变更，例如 batching、cache、parallelism。

# Skill: Kernel Optimization Loop

此 skill 定义 Kernel Agent 的内部优化循环。

## Loop

1. Read
   - 读取 `task_pack/` 中的 `task.yaml`、`shape_list.json`、reference、candidate 接口、correctness、benchmark、baseline 和约束。
   - 确认语义、shape、dtype、layout、目标硬件。

2. Accept or Block
   - 如果 task pack 不完整、reference 不清、shape 缺失，输出 `task_acceptance_review.md`。
   - 如果任务可执行，明确实现策略和第一轮目标。

3. Implement
   - 首选最小正确实现。
   - 保留候选实现和 baseline 可对比。

4. Correctness
   - 跑所有 required shape。
   - 失败时先修精度，再优化性能。

5. Benchmark
   - warmup + repeat + synchronize。
   - 记录 median/mean/std 或 p50/p90。

6. Profile
   - 对热 shape 跑 NCU。
   - 提取瓶颈指标和代码位置。

7. Analyze and Modify
   - 基于指标做一轮明确改动。
   - 不同时修改太多独立因素。

8. Deliver or Request Framework Change
   - 达标：输出 `KernelDeliveryPackage`。
   - 需要框架配合：输出 `FrameworkChangeRequest`。
   - 无法达标：输出瓶颈解释和替代建议。

## 记录要求

每轮至少记录：

- candidate ID。
- 修改点。
- correctness 结果。
- benchmark 摘要。
- profiler 摘要。
- 保留/放弃原因。

## 停止条件

- 达到 `target_perf`。
- correctness 无法满足且原因明确。
- NCU 显示已接近硬件上限。
- 需要框架改造才能继续。
- 预算耗尽，需要人工决策。

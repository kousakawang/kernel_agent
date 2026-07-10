# Skill: NVIDIA Nsight Compute Analysis

此 skill 用于 NVIDIA/H20 上的单 kernel profile 诊断。第一阶段只作为分析指南，不自动运行 profiler。

## 推荐采集

模板命令：

```bash
bash scripts/run_ncu.sh <case_id>
```

必要时缩小 section：

```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats \
  python benchmark.py --case-id <case_id> --target candidate
```

## 关键指标

- SM utilization：是否有足够并行度。
- Achieved occupancy：是否受 registers/shared memory/block size 限制。
- DRAM throughput：是否接近显存带宽。
- L2 hit rate / throughput：是否 cache 友好。
- Warp stall reasons：memory dependency、barrier、not selected、long scoreboard。
- Tensor core / FP pipe utilization：是否用到合适计算单元。
- Registers per thread：是否限制 occupancy。
- Shared memory bank conflict：是否有访存冲突。
- Launch overhead：小 shape 是否被 launch 支配。

## 诊断映射

- DRAM 高、SM 低：优先减少读写、融合、改善 coalescing。
- L2 命中低：检查 layout、访问顺序、重用距离。
- Long scoreboard 高：隐藏 memory latency，增加并行度或减少依赖链。
- Barrier 高：减少同步或改变 tile 组织。
- Occupancy 低：检查 registers、shared memory、num warps。
- 小 shape 慢：考虑 persistent、fusion、CUDA graph 或 framework batching。

## 输出要求

在 `benchmark_report.md` 或 `kernel_delivery_package.md` 中记录：

- NCU 命令。
- 目标 case。
- 主要瓶颈。
- 指标摘要。
- 和代码改动对应的解释。

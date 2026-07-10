# Skill: Hotspot Prioritization

此 skill 用于把 profile 结果变成优化优先级列表。

## 排序维度

- `gpu_time_share`：GPU 时间占比越高，优先级越高。
- `call_frequency`：调用频率越高，稳定收益越重要。
- `shape_stability`：shape 越稳定，越适合专门优化。
- `e2e_upper_bound`：按 Amdahl's law 估计收益上限。
- `ut_feasibility`：能否构造独立 golden/UT。
- `integration_risk`：接入是否需要大范围框架改造。
- `hardware_signal`：是否有明确硬件瓶颈，例如带宽、occupancy、tensor core 未用满。

## 推荐决策

- `P0`：热点占比高，UT 可构造，接入风险低。
- `P1`：热点占比高，但需要框架配合或 profile 还不完整。
- `P2`：局部可优化，但端到端收益不确定。
- `Reject`：不适合作为 kernel 任务。

## 输出格式

使用 `templates/hotspot_report.md`。每个候选必须给出：

- 优先级。
- 证据。
- 转成 `KernelRequestSpec` 的路径。
- 暂不处理的原因。

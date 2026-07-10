# Skill: Audit and Anti-Cheat

此 skill 用于 Phase 4：独立复测 kernel correctness/performance，避免 benchmark 假阳性或 reward hacking。

## Correctness Audit

- 随机 seed 多次变化。
- shape sweep 覆盖 hot、edge、non-contiguous 可选场景。
- fresh process 运行，避免缓存旧输出。
- candidate 输出不能复用 golden buffer。
- 检查输入是否被非法修改。
- 对 tolerance 异常放宽进行人工标记。

## Performance Audit

- warmup/repeat 固定。
- 多轮运行看稳定性。
- baseline 和 candidate 同进程/同条件对比。
- 检查 CUDA synchronize。
- 检查是否只优化单个 shape 导致其他 required shape 退化。
- 对极小 latency 检查 launch/cuda graph/cache 影响。

## Red Flags

- 修改 golden 或测试 tolerance。
- 根据固定随机 seed 特化输出。
- 返回缓存结果。
- benchmark 只覆盖单 shape。
- 候选实现 silently fallback 到 baseline，却报告 candidate 时间。

## 输出

使用 `templates/audit_report.md`。

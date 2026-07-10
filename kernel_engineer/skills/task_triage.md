# Skill: Task Triage

此 skill 在 Phase 1 用于接收并验收单个 `task_pack/`；后续 Phase 2 可以扩展为多个候选任务之间的优先级选择。

## 接受任务的条件

- `task.yaml` 字段完整。
- `shape_list.json` 至少包含一个 required hot case。
- `reference_impl.py` 和 `candidate_impl.py` 接口一致。
- correctness harness 有独立 reference，不依赖候选 kernel。
- benchmark harness 可以比较 reference 和 candidate。
- `env_manifest.yaml` 明确 Triton、CuTe DSL、CUDA extension、NCU 的 `available` 与 `usable_for_task`。
- 至少一个 hot shape 明确。
- baseline 和 target 可比较。
- 支持范围和允许框架改造边界清楚。

## 优先级评分

- 端到端收益上限高。
- micro benchmark 目标明确。
- shape 稳定且集中。
- 接入风险低。
- 目标硬件和工具链已支持。
- 有明确 profiler 证据。

## 输出

使用 `templates/task_acceptance_review.md`。

可能结论：

- `ACCEPT`：可开始实现。
- `NEEDS_MORE_INFO`：需要 Framework Engineer 补齐信息。
- `REJECT_FOR_NOW`：暂不适合 kernel 侧处理。

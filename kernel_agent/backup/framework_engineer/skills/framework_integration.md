# Skill: Framework Integration

此 skill 用于 Phase 3：把 Kernel Agent 交付的实现接入 SGLang，并保留 fallback 和验收路径。

## 接入原则

- 新 kernel 必须有 feature flag 或 backend option。
- 必须保留原 backend fallback。
- 不允许 silent fallback；fallback 需要日志或报告可见。
- 接入前先确认 `KernelDeliveryPackage` 的 shape/dtype/layout 约束。
- 若需要 workspace、metadata、layout 变化，必须先评审 `FrameworkChangeRequest`。

## SGLang 首期关注点

- backend registry 或模型配置如何选择新 kernel。
- prefill/decode 是否走不同路径。
- CUDA graph capture 是否受影响。
- cache、metadata、workspace 生命周期是否和 scheduler 兼容。
- tensor parallel、batch、chunked prefill 是否覆盖。

## 接入交付物

- `integration_plan.md`
- 代码变更说明。
- fallback 条件。
- e2e 验证命令。
- rollback 方法。

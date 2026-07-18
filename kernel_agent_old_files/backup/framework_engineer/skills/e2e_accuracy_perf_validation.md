# Skill: E2E Accuracy and Performance Validation

此 skill 用于判断 kernel 优化是否真正改善模型场景，而不是只改善 micro benchmark。

## 性能验收

记录并对比：

- E2E latency。
- Throughput。
- TTFT / TPOT，如 workload 适用。
- GPU memory。
- batch、token、cache 命中、cuda graph 状态。
- 服务日志中的异常或 fallback。

## 精度验收

根据场景选择：

- 固定 prompt 输出一致性。
- logits 或 hidden state 对比。
- 任务级指标。
- 采样场景下使用 deterministic 配置。

## 判定规则

- correctness 失败则直接拒绝。
- micro benchmark 变快但 e2e 无收益，需要解释原因。
- e2e 变快但精度/稳定性下降，不能接受。
- 如果收益只在少数 shape 出现，需要明确启用条件和 fallback。

## 报告

使用 `templates/e2e_verification_report.md`。

# Skill: Framework Feedback

此 skill 定义 Kernel Agent 何时向 Framework Engineer Agent 提出 `FrameworkChangeRequest`。

## 需要反馈的场景

- 输入 layout/stride 导致无法 coalesced load。
- 每次 kernel 内重复构造 metadata，适合框架侧预计算。
- 需要 persistent workspace 或 ping-pong buffer。
- prefill/decode 应拆成不同 kernel 或不同 metadata。
- 权重或 cache 需要重排才能使用高效访问模式。
- 当前融合边界不合理，kernel launch 或 global memory round trip 过多。
- 需要额外 padding/alignment 才能利用硬件特性。

## 反馈要求

`FrameworkChangeRequest` 必须说明：

- 要框架改变什么。
- kernel 侧为什么需要。
- 预计收益和证据。
- 对 correctness、fallback、兼容性的影响。
- 对 UT/spec 的影响。
- 如果框架不改，kernel 侧还能做到什么程度。

## 不应反馈的内容

- “希望框架更快”这类不具体诉求。
- 没有 benchmark/profile 证据的侵入式改造。
- 会改变模型语义但没有明确验收方法的改造。

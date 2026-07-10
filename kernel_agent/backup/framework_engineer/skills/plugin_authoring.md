# Skill: Framework Plugin Authoring

此 skill 用于为新的框架、模型族或业务场景增加 Framework Engineer 插件。

## 何时新增插件

- 新框架，例如 vLLM、TensorRT-LLM、自研 serving。
- 新模型族有特殊结构或 cache/metadata 机制。
- 新 workload 需要特殊 e2e 验收。
- 新 profiler 或日志格式需要专门解析。

## 编写步骤

1. 说明适用范围和不适用范围。
2. 写清模型/模块源码入口。
3. 写清启动命令、workload 命令和 profile 命令。
4. 定义如何判断热点是否适合作为 kernel 任务。
5. 定义如何接入 Kernel Agent 的交付物。
6. 定义 accuracy/perf 验收标准。

## 约束

- 插件不能改变 `KernelRequestSpec` 的核心字段。
- 插件可以增加框架专属字段，但必须放在 `framework_extra`。
- 插件文档必须包含 fallback 和 rollback 方法。

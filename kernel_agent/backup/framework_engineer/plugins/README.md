# Framework Engineer Plugins

Framework plugins describe how to profile, extract kernel requests, integrate kernels, and validate e2e behavior for a specific framework or model family.

第一阶段默认插件目标是 SGLang + Qwen3.5 + NVIDIA/H20。这里先定义插件规范，不实现插件加载系统。

## Plugin Shape

每个插件目录建议包含：

```text
plugins/<framework_or_model>/
  README.md
  profile.md
  request_extraction.md
  integration.md
  validation.md
  commands.md
```

## Required Content

- 支持的框架版本或 commit 范围。
- 模型/模块入口。
- 推荐 profile 命令和日志解析方法。
- 如何从热点转成 `KernelRequestSpec`。
- kernel 接入方式和 fallback 策略。
- e2e 验证 workload 和精度检查方法。

## Extension Rule

新增框架或场景时，优先新增 plugin 文档，不修改核心 prompt。

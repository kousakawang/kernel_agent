# Kernel Agent Plugins

Kernel plugins describe hardware, DSL, compiler, profiler, and existing kernel library knowledge. 第一阶段默认 NVIDIA/H20 + Triton/CUDA + Nsight，不实现插件加载系统。

## Plugin Shape

每个插件目录建议包含：

```text
plugins/<hardware_or_dsl>/
  README.md
  hardware_spec.md
  programming_model.md
  compiler.md
  profiler.md
  examples.md
```

## Required Content

- 适用硬件和软件版本。
- memory hierarchy、compute units、preferred alignment。
- 推荐 DSL/API 和编译方式。
- profiler 命令和指标解释。
- 已有 kernel example。
- 常见性能陷阱。

## Extension Rule

新增硬件或 DSL 时，优先新增 plugin 文档和 skill，不修改核心 Kernel Engineer prompt。

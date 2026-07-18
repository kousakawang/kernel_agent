# Kernel Agent Archive

当前 active 工程已经顶层化：

```text
framework_engineer/
kernel_engineer/
```

`kernel_agent/` 只保留历史设计、旧模板、调研参考和示例 task pack 的归档内容。默认不要把这里的文件作为 Phase 1.2 运行上下文。

## Active Entry Points

Framework Engineer:

```bash
python -m framework_engineer.cli validate-config --config <config.py>
python -m framework_engineer.cli run-phase1 --config <config.py>
```

Kernel Engineer:

```text
kernel_engineer/prompts/kernel_engineer.md
kernel_engineer/skills/task_pack_optimization_protocol.md
```

双方唯一交互物是 Framework Engineer 生成的独立 `task_pack/`。

## Backup

历史内容位于：

```text
kernel_agent/backup/
```

包括早期设计草稿、shape-only 模板、Qwen 专用示例、Phase 2+ 设想和外部参考 repo。需要恢复能力时，先审阅 backup，再按当前 snapshot task-pack 协议重新引入 active 目录。

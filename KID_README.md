# KID 使用指南

KID（Kernel Interface Decomposition）由一个 KID Agent 对外统一执行，内部包含确定性的
Runtime Capture CLI 和需要源码判断的 Semantic Resolution 两个阶段。最终产物是交给
`source_locate` 的 `kernel-interface-decomposition/v2` JSON。

## 1. 准备单 backend 配置

每次任务只处理一个 backend 和一条 `cmd/test_cmd`。准备目录：

```text
<workspace>/config/<backend>/
├── runtime_capture_config.json
└── semantic_resolver_config.json
```

可参考完整 golden：
[`example_kernels/nsys_poc_kid_golden`](example_kernels/nsys_poc_kid_golden)。不要直接用 capture
覆盖 golden；复制配置并把 `workdir`、`output_dir`、Runtime/local path mapping、源码根目录和
各输出路径改到新的 workspace。

Runtime 配置中的 `target` 是用户指定的 high-level Python 接口。`cmd=null` 时直接 profile
`test_cmd`；有服务命令时，CLI 等待 `ready` 后才开始 Nsight 和 Runtime capture。当前必须禁用
CUDA Graph，推荐 `selection.sampling=unique_decomposition`。

## 2. 启动 KID Agent

在 Codex 中提交：

```text
请阅读 kernel_agent/framework_engineer/prompts/start_kid.md，
执行 KID，配置目录是：<workspace>/config/<backend>
```

Agent 默认重新运行 Runtime capture，然后依次完成 `prepare`、semantic decisions/notes、
`finalize` 和 `validate`。只有明确告诉 Agent 复用已有 Runtime 时，才会跳过 GPU capture。

Runtime 阶段必须在装有 CUDA 和 Nsight Systems 的环境运行。本工作区通常使用：

- `python`：封装的远端 GPU runner，会同步仓库并在 GPU 容器执行；
- `python3`：本地解释器，用于 context/finalize/validate 等无 GPU helper。

若本机本身具备 GPU/Nsight，也可以让 Agent 全部使用同一个 Python。

## 3. 最小端到端验收

第一次验证总入口时，建议复制 golden 配置到新 workspace，并把 Runtime 配置中的
`test_cmd` 缩减为一个 PyTorch case：

```bash
python kernel_agent/framework_engineer/kernel_interface_decomposer/nsys_poc.py \
  --worker --cases pytorch_native
```

不要预填或复制 golden decisions，让 Agent 根据新生成的 context 作判断。预期最终只有一个
`torch.matmul` semantic target，`archetype=pytorch_dispatch`、`provider=pytorch`、coverage 为
100%，且 call site 来自本轮 Runtime stack edge。耗时和 Runtime ID 不作固定比较。

## 4. 产物与成功条件

```text
<workspace>/
├── config/<backend>/...
├── cli_log/<backend>/
│   ├── runtime_capture.schema.json
│   ├── capture_events/events_<pid>.jsonl
│   ├── trace/profile.sqlite
│   └── logs/...
├── ref/<backend>/
│   ├── semantic_resolver_context.json
│   ├── semantic_resolver_decisions.json
│   └── kid_semantic_resolver_notes.md
└── output/<backend>/decomposition.schema.json
```

任务只有在 Runtime artifact 自校验通过、所有 direct kernel owner 被唯一分配且 Semantic
Resolver `validate` 返回 0 后才算完成。`source_locate` 只读取最终
`output/<backend>/decomposition.schema.json`。

## 5. 分阶段排错

必要时可手工重放确定性步骤：

```bash
python -m framework_engineer.kernel_interface_decomposer \
  capture <config-dir>/runtime_capture_config.json

python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  prepare <config-dir>/semantic_resolver_config.json
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  finalize <config-dir>/semantic_resolver_config.json
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  validate <config-dir>/semantic_resolver_config.json
```

详细文件责任边界见
[`example_kernels/nsys_poc_kid_golden/ARTIFACT_GUIDE.md`](example_kernels/nsys_poc_kid_golden/ARTIFACT_GUIDE.md)，
KID 到 source locate 的字段合同见
[`kernel_agent_kadai/KID_to_source_locate_handoff.md`](kernel_agent_kadai/KID_to_source_locate_handoff.md)。

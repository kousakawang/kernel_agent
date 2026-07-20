# Source Locate Golden

这个目录是 2026-07-20 基于最新 KID-v2 输出和当前 source-location 实现重放得到的完整样例。

本 golden 按 testcase 建立独立 workspace。当前 testcase ID 是 `all_backends`，它的 KID 输入是 `input/all_backends/decomposition.kid.schema.json`，所有运行产物都位于 `workspaces/all_backends/`。

要确认本次 source locate 的实际结果，优先查看：

- Agent 正式定位结果：[`workspaces/all_backends/agent/located.schema.json`](workspaces/all_backends/agent/located.schema.json)
- Agent 搜索说明：[`workspaces/all_backends/agent/ref/locate_agent_notes.md`](workspaces/all_backends/agent/ref/locate_agent_notes.md)
- extract 后交付 schema：[`workspaces/all_backends/extract/decomposition.extracted.schema.json`](workspaces/all_backends/extract/decomposition.extracted.schema.json)
- 最终源码包：[`workspaces/all_backends/extract/kernel_sources/`](workspaces/all_backends/extract/kernel_sources/)

## 公共 Agent 文件

Prompt 和 Skill 属于 source-locate 的公共实现，不复制进单个 golden case：

- 启动 Prompt：[`framework_engineer/prompts/start_source_locate.md`](../../framework_engineer/prompts/start_source_locate.md)
- 搜索 Skill：[`framework_engineer/skills/source_locate.md`](../../framework_engineer/skills/source_locate.md)

每个 testcase 只保存自己的输入、配置、Agent decisions 和运行产物。

## 目录结构

```text
source_locate_golden/
├── config/
│   └── all_backends/
│       └── third_party_manifest.json        # 允许成为正式 hit 的源码仓
├── input/
│   └── all_backends/
│       └── decomposition.kid.schema.json    # 此 testcase 的 KID-v2 输入
└── workspaces/
    └── all_backends/
        ├── locate/
        │   └── locate_candidates.schema.json
        ├── agent/
        │   ├── source_locate_decisions.json
        │   ├── located.schema.json
        │   └── ref/locate_agent_notes.md
        ├── extract/
        │   ├── decomposition.extracted.schema.json
        │   └── kernel_sources/<low_level_id>/...
        └── validation/
            ├── agent_evaluation.json
            └── artifact_check.json
```

这里的对应关系是固定的：

```text
input/<testcase>/decomposition.kid.schema.json
config/<testcase>/...
workspaces/<testcase>/...
```

新增 testcase 时，应创建相同名字的三个目录，不与 `all_backends` 共用 workspace。

## 各阶段产物

1. `locate/locate_candidates.schema.json`：公开 locate CLI 生成的 Python interface 候选。
2. `agent/source_locate_decisions.json`：Agent 私有工作文件，包含四层状态、hits 和判断理由。
3. `agent/located.schema.json`：finalize 后、extract 前的正式 source-location schema；不再包含 `locate_candidates`。
4. `agent/ref/locate_agent_notes.md`：由 finalize 从 decisions 生成的可读定位说明。
5. `extract/decomposition.extracted.schema.json`：在 located schema 基础上增加各 target 的 `kernel_sources_dir`。
6. `extract/kernel_sources/`：最终交付的完整源码副本、缺失层占位文件和 `read_hints.txt`。

## 本次结果摘要

- 输入包含 10 个 low-level target。
- locate CLI 唯一定位 9 个 Python interface；`torch_sdpa` 因 manifest 未提供 PyTorch 源码仓而为 `not_found`。
- Agent 对其余 9 个 target 完成四层源码判断；确实不存在的 binding/header 标为 `not_applicable`。
- `torch_sdpa` 的四层均为 `missed`，并在最终源码包中生成占位说明。
- extract 为全部 10 个 target 生成目录、27 份源码副本、13 份占位文件和 10 份 `read_hints.txt`。

## 如何重放 all_backends testcase

以下命令从 `kernel_agent/` 目录执行。源码位置记录在 `config/all_backends/third_party_manifest.json`；若工作区移动，需要先更新 manifest 和 decisions 中的绝对路径。

### 1. locate CLI

```bash
python3 -m framework_engineer.source_location.cli locate \
  --schema example_kernels/source_locate_golden/input/all_backends/decomposition.kid.schema.json \
  --manifest example_kernels/source_locate_golden/config/all_backends/third_party_manifest.json \
  --sglang-repo-root /Users/bytedance/Desktop/infra_agent/sglang \
  --out example_kernels/source_locate_golden/workspaces/all_backends/locate/locate_candidates.schema.json
```

### 2. source_locate Agent

使用公共 [`start_source_locate.md`](../../framework_engineer/prompts/start_source_locate.md) 启动 Agent。Agent 按公共 [`source_locate.md`](../../framework_engineer/skills/source_locate.md) 搜索源码，将工作决策写到当前 testcase workspace，然后调用 finalize：

```bash
python3 -m framework_engineer.source_location.agent_helper finalize \
  --schema example_kernels/source_locate_golden/workspaces/all_backends/locate/locate_candidates.schema.json \
  --decisions example_kernels/source_locate_golden/workspaces/all_backends/agent/source_locate_decisions.json \
  --manifest example_kernels/source_locate_golden/config/all_backends/third_party_manifest.json \
  --sglang-repo-root /Users/bytedance/Desktop/infra_agent/sglang \
  --out example_kernels/source_locate_golden/workspaces/all_backends/agent/located.schema.json \
  --notes-out example_kernels/source_locate_golden/workspaces/all_backends/agent/ref/locate_agent_notes.md
```

Agent 到这里结束，不调用 extract。

### 3. extract CLI

extract 会原地向输入 schema 写入 `kernel_sources_dir`。为了保留 Agent 的 located schema，先复制到 workspace 的 extract 阶段，再执行：

```bash
cp example_kernels/source_locate_golden/workspaces/all_backends/agent/located.schema.json \
  example_kernels/source_locate_golden/workspaces/all_backends/extract/decomposition.extracted.schema.json

python3 -m framework_engineer.source_location.cli extract \
  --schema example_kernels/source_locate_golden/workspaces/all_backends/extract/decomposition.extracted.schema.json \
  --workspace-out example_kernels/source_locate_golden/workspaces/all_backends/extract
```

## 边界说明

- schema 中的 source hit 和 `kernel_sources_dir` 使用本次运行机器上的绝对路径，这是当前契约要求。
- manifest 只允许 SGLang、内嵌 sgl-kernel，以及 `status=ok` 的 FlashInfer、DeepGEMM、sgl-attn 仓成为正式 hit。
- `provider`、KID capture 分类和 runtime kernel name 只作为 KID 输入信息保留，不用于 source-location 分派。
- `end_line` 不写回 schema，只体现在各 target 的 `read_hints.txt` 中。

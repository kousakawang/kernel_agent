# Start Source Locate Agent

你是 source_locate Agent。按照 `framework_engineer/skills/source_locate.md` 的标准，从 locate
candidate schema 出发完成四层源码定位，并严格停在 extract 之前。

## 用户需要提供

- `<candidate-schema>`：`locate` CLI 的输出；
- `<third-party-manifest>`；
- `<sglang-repo-root>`；
- `<decisions-out>`；
- `<located-schema-out>`；
- `<notes-out>`。

不要通过对话重新收集 schema 已经包含的 KID 字段。缺少输入文件、schema contract 非法或源码根
不存在时，报告具体错误并停止。

## 执行步骤

### 1. 阅读约束

完整阅读：

```text
framework_engineer/skills/source_locate.md
framework_engineer/source_location/example/README.md
```

确认本任务不修改 KID semantic target、不修改源码、不调用 extract。

### 2. 枚举 targets

从 candidate schema 读取所有 `low_level_id`。逐个运行：

```bash
python -m framework_engineer.source_location.agent_helper inspect-target \
  --schema <candidate-schema> \
  --kernel-id <low-level-id> \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root>
```

检查 locator candidate、call site、合法 roots 和 skipped roots。不要因为一个 target 缺源码而中止
其余 targets。

### 3. 逐 target 阅读和搜索

先打开 candidate definition 及其直接调用，再根据已经看见的 symbol 使用私有搜索：

```bash
python -m framework_engineer.source_location.agent_helper search \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root> \
  --mode literal \
  --query <qualified-or-source-symbol>
```

需要时将 mode 替换为 `registration`、`loader` 或 `build`。可以直接使用 `rg` 和源码阅读工具补充，
但所有正式 hit 仍必须落在 helper 给出的合法 roots 中。

每找到一个节点都必须验证源码中的真实调用边，然后按照以下顺序记录：

- `interface_definition`：semantic interface 本身；
- `py_cpp_binding`：Python→native/JIT/FFI；
- `kernel_impl`：host→dispatch/template→device kernel；
- `kernel_header`：独立声明 header。

### 4. 写 decisions

按照 `source-locate-agent-decisions/v1` 写 `<decisions-out>`。每个 kernel 和每层都必须有 summary、
rationale；每个 hit 都必须有 symbol 和 reason。

不确定但已有可靠 hit 时使用 `best_effort`，没有可靠 hit时使用 `missed`，确认该层不存在时才使用
`not_applicable`。不得使用 `ambiguous/not_found` 作为最终状态。

### 5. Finalize

```bash
python -m framework_engineer.source_location.agent_helper finalize \
  --schema <candidate-schema> \
  --decisions <decisions-out> \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root> \
  --out <located-schema-out> \
  --notes-out <notes-out>
```

如果失败，修正 decisions 中报告的具体 contract/path/line 问题后重试。不要直接手改 finalized schema
来绕过 helper。

### 6. 可选 Golden 评测

若任务提供 `<golden-schema>`，运行：

```bash
python -m framework_engineer.source_location.agent_helper evaluate \
  --actual <located-schema-out> \
  --decisions <decisions-out> \
  --golden <golden-schema> \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root>
```

失败时检查缺失或乱序的核心链。允许添加有源码证据且带 reason 的 helper hits，不允许删除或重排
Golden 核心链来迎合输出。

## 最终报告

只报告：

- processed/resolved/best-effort/missed target 数量；
- located schema、decisions 和 notes 路径；
- skipped/missing repos；
- 需要人工补仓或 KID 重新解析的 target；
- Golden evaluate 是否通过（若执行）。

不要调用 `framework_engineer.source_location.cli extract`。外层工作流负责下一阶段。

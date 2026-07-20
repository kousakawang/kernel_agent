# Source Locate Agent 使用说明

Source Locate 负责在 KID 已经确定 semantic low-level target 后，为每个 target 定位四层源码：

- `interface_definition`：semantic Python interface 定义；
- `py_cpp_binding`：Python→C++/FFI/JIT 桥接链；
- `kernel_impl`：host launcher→dispatch/template→核心 device kernel；
- `kernel_header`：独立的声明/API header。

完整流程是：

```text
KID v2 schema
  → locate CLI
  → candidate schema
  → source_locate Agent
  → decisions JSON
  → finalize helper
  → located schema + locate_agent_notes.md
  → extract CLI
  → extracted schema + kernel_sources/
```

`locate` 和 `extract` 是两个公开 CLI。Agent 使用的 `inspect-target/search/finalize/evaluate`
位于私有 `agent_helper` 中。

## 1. 运行方式

Source Locate Agent 采用“入口 Prompt + Skill + 本地工具”的运行方式，不包含独立的 Python LLM
runner。用户需要在能够读取本仓库、访问源码目录并执行 shell 命令的宿主 Agent 中启动任务。

Agent 的两个入口文件是：

```text
framework_engineer/prompts/start_source_locate.md
framework_engineer/skills/source_locate.md
```

入口 Prompt 负责执行顺序，Skill 定义四层语义、状态标准、搜索边界和停止条件。

## 2. 依赖

### 必需软件

- Python 3；当前实现只依赖 Python 标准库。
- `rg`（ripgrep）；私有 search helper 使用它执行限定源码根内的搜索。
- 一个能够阅读 Prompt/Skill、访问本地文件并运行命令的宿主 Agent。

在 `kernel_agent` 仓库根目录检查：

```bash
python3 --version
rg --version
PYTHONPATH=. python3 -m framework_engineer.source_location.cli --help
PYTHONPATH=. python3 -m framework_engineer.source_location.agent_helper --help
```

公开 CLI 的 help 应只包含 `locate` 和 `extract`；私有 helper 应包含
`inspect-target/search/finalize/evaluate`。

### 不需要的依赖

- 不需要 GPU。
- 不需要启动 SGLang 服务。
- 不需要编译 SGLang、sgl-kernel 或第三方 kernel。
- 不需要安装或修改任何被定位的 Python package。

Source Locate 只读取源码。除指定的 workspace 输出外，不应修改 SGLang 或第三方源码仓。

## 3. 用户需要准备什么

### 3.1 KID v2 schema

输入必须是 KID Semantic Resolver 的最终 schema，例如：

```text
example_kernels/source_locate_golden/input/all_backends/decomposition.kid.schema.json
```

它必须满足：

- `schema_version` 为 `kernel-interface-decomposition/v2`；
- 每个 kernel 有唯一、安全的 `low_level_id`；
- 已确定 semantic `interface`；
- 有 `runtime_event.call_site.file/line`；
- `archetype` 是 KID capture 类别；
- `provider` 字段存在，值可以是字符串或 `null`；
- 尚未包含 `locate_candidates`、`source_locations` 或 `kernel_sources_dir`。

Source Locate 不会修改 rank、interface、provider、archetype、kernel、metrics、coverage 或 call site。

### 3.2 SGLang 源码根目录

准备完整的 SGLang source tree，并记录绝对路径：

```text
/absolute/path/to/sglang
```

若该目录中存在 `sgl-kernel/`，locate 会自动把它作为一个更具体的源码根加入搜索范围。

### 3.3 Third-party manifest

准备 `third_party_manifest.json`，列出允许 Agent 读取并写入正式 hits 的第三方源码仓。例如：

```json
{
  "schema_version": 1,
  "repos": [
    {
      "name": "flashinfer",
      "status": "ok",
      "local_path": "/absolute/path/to/flashinfer"
    },
    {
      "name": "deep_gemm",
      "status": "ok",
      "local_path": "/absolute/path/to/DeepGEMM"
    }
  ]
}
```

规则：

- 只有 `status=ok` 且 `local_path` 存在的仓库会进入搜索范围。
- `local_path` 应指向完整、可读的源码树，不是 wheel、`.so` 或编译缓存目录。
- 缺失仓库不会让 `locate` 整体失败，但相关层最终通常会是 `missed`。
- 若希望定位 PyTorch 等额外实现，必须先准备对应源码仓并加入 manifest。
- `provider` 不会自动授权一个仓库；正式 hit 必须实际落在上述允许 roots 中。

可以从本机示例开始修改：

```text
framework_engineer/source_location/example/third_party_manifest.json
```

### 3.4 可写 workspace

建议为每次 backend/source-locate 运行准备一个独立目录：

```text
<workspace>/
  decomposition.kid.json
  decomposition.locate_candidates.json
  source_locate_decisions.json
  decomposition.located.json
  ref/
    locate_agent_notes.md
```

输入、candidate 输出和 located 输出必须使用不同路径，避免污染 KID 原始产物。

## 4. 第一步：运行 locate CLI

在 `kernel_agent` 仓库根目录执行：

```bash
PYTHONPATH=. python3 -m framework_engineer.source_location.cli locate \
  --schema <workspace>/decomposition.kid.json \
  --manifest /absolute/path/to/third_party_manifest.json \
  --sglang-repo-root /absolute/path/to/sglang \
  --out <workspace>/decomposition.locate_candidates.json
```

Locate 会：

1. 校验 KID v2 的 source-location 必需字段；
2. 加载 SGLang、内嵌 sgl-kernel 和 manifest 源码根；
3. 从 call-site import、alias、relative import、re-export 和 qualified class method 定位 Python
   interface 候选；
4. 在 schema 副本中写入临时 `locate_candidates.interface_definition`。

单 target 候选状态可能是：

- `resolved`：一个候选；
- `ambiguous`：多个候选；
- `not_found`：没有候选。

这些只是 Agent 输入，不是最终四层状态。单个 `not_found` 不会阻断其他 targets。

核心产物：

```text
<workspace>/decomposition.locate_candidates.json
```

## 5. 第二步：启动 Source Locate Agent

在宿主 Agent 中创建一个能够访问本仓库和上述源码目录的任务，然后提交以下任务描述。将所有路径
替换成实际绝对路径：

```text
请阅读并严格执行：
  <kernel_agent>/framework_engineer/prompts/start_source_locate.md
  <kernel_agent>/framework_engineer/skills/source_locate.md

输入：
  candidate schema: <workspace>/decomposition.locate_candidates.json
  third-party manifest: /absolute/path/to/third_party_manifest.json
  sglang repo root: /absolute/path/to/sglang

输出：
  decisions: <workspace>/source_locate_decisions.json
  located schema: <workspace>/decomposition.located.json
  notes: <workspace>/ref/locate_agent_notes.md

逐个 low_level_id 阅读源码、完成四层定位并调用 finalize helper。
在生成 located schema 和 notes 后停止，不要调用 extract。
```

Agent 的内部执行顺序是：

1. 对每个 target 调用 `agent_helper inspect-target`；
2. 验证 locate 给出的 interface candidate；
3. 阅读 interface body，沿真实调用关系继续追踪；
4. 按需调用 `agent_helper search` 的 `literal/registration/loader/build` 模式；
5. 写 `source-locate-agent-decisions/v1` decisions；
6. 调用 `agent_helper finalize` 生成正式 schema 和 notes；
7. 在 extract 之前结束。

Agent 不应把 helper 的文本匹配直接当作结论。每个正式 hit 都必须有源码调用边证据，并在
decisions 中记录 `symbol/reason`。

## 6. Agent 阶段的产物

### `source_locate_decisions.json`

Agent 内部工作合同，包含：

- 每个 target 的调用链 summary；
- 每层的 `status/rationale`；
- 每个 hit 的 `file/def_line/symbol/reason`；
- `best_effort/missed` 的 gaps；
- 需要补仓或人工确认时的 `manual_followup`。

它用于 finalize、审计和 Golden evaluate，不是下游固定 schema。

### `decomposition.located.json`

正式 Agent 输出。它：

- 保持所有 KID-owned fields 不变；
- 删除临时 `locate_candidates`；
- 为每个 target 增加最小 `source_locations.layers`；
- 只保留 `{status, hits, repo_hint}`；
- 不包含 `kernel_sources_dir`。

最终层状态只有：

- `resolved`
- `best_effort`
- `missed`
- `not_applicable`

`interface_definition` 和 `kernel_impl` 缺源码时必须是 `missed`，不能是
`not_applicable`。

### `ref/locate_agent_notes.md`

由 finalize 根据 decisions 机械生成，包含：

- interface 与完整调用链概述；
- 每层判断理由；
- 每个 hit 的 symbol 和证据；
- 未完成的动态/模板/生成代码路径；
- 缺失仓库和人工 follow-up 建议。

## 7. 可选：运行 Golden evaluate

开发或回归测试时可以执行：

```bash
PYTHONPATH=. python3 -m framework_engineer.source_location.agent_helper evaluate \
  --actual <workspace>/decomposition.located.json \
  --decisions <workspace>/source_locate_decisions.json \
  --golden example_kernels/source_locate_golden/workspaces/all_backends/agent/located.schema.json \
  --manifest /absolute/path/to/third_party_manifest.json \
  --sglang-repo-root /absolute/path/to/sglang
```

评测要求 Golden 核心 hits 按调用顺序出现在实际结果中，允许增加 decisions 中有明确 reason 的
helper hits。

返回码：

- `0`：通过；
- `1`：输出 contract 合法，但不满足 Golden 核心链；
- `2`：输入、decisions、源码路径或行号非法。

## 8. 第三步：运行 extract CLI

Agent 成功结束后，由用户或外层工作流显式执行：

```bash
PYTHONPATH=. python3 -m framework_engineer.source_location.cli extract \
  --schema <workspace>/decomposition.located.json \
  --workspace-out <workspace>
```

Extract 会：

1. 在清理旧输出前预检四层结构、文件和行号；
2. 对 `resolved/best_effort` 复制完整源文件；
3. 对 `missed/not_applicable` 生成说明占位；
4. 用 Python AST 或 C/C++/CUDA brace scanner 计算 definition end line；
5. 把范围写入 `read_hints.txt`，不向 schema 添加 `end_line`；
6. 在 schema 中为每个 target 增加 `kernel_sources_dir`。

Extract 会原地更新传入的 located schema。若需要同时保留 extract 前后的 schema，应先复制一份：

```bash
cp <workspace>/decomposition.located.json \
   <workspace>/decomposition.extracted.json

PYTHONPATH=. python3 -m framework_engineer.source_location.cli extract \
  --schema <workspace>/decomposition.extracted.json \
  --workspace-out <workspace>
```

## 9. 最终产物

推荐最终目录：

```text
<workspace>/
  decomposition.kid.json                 # KID 原始输出
  decomposition.locate_candidates.json   # locate 临时候选
  source_locate_decisions.json           # Agent 证据与决议
  decomposition.located.json             # extract 前的正式四层结果
  decomposition.extracted.json           # 增加 kernel_sources_dir
  ref/
    locate_agent_notes.md
  kernel_sources/
    <low_level_id>/
      interface_definition.py
      py_cpp_binding/
      kernel_impl/
      kernel_header/
      read_hints.txt
```

各产物的消费者：

| 产物 | 主要用途 |
| --- | --- |
| candidate schema | Source Locate Agent 的 interface 起点 |
| decisions JSON | finalize、审计、问题排查和 Golden 评测 |
| located schema | extract 的正式输入 |
| locate notes | 人工查看证据、歧义和缺失仓库 |
| extracted schema | problem_translate/workspace 下游索引 |
| `kernel_sources/` | 脱离原始仓库后仍可阅读的源码物料 |
| `read_hints.txt` | 指示每份完整源文件应重点阅读的 definition 范围 |

## 10. 常见失败和处理

### Locate 返回 `not_found`

这是合法候选结果。Agent 会继续搜索；如果源码仓确实未提供，最终相关层标为 `missed`。

### Manifest repo 被跳过

检查该项是否为 `status=ok`，以及 `local_path` 是否存在并指向完整源码树。

### Finalize 拒绝文件路径

正式 hit 只能位于 SGLang、内嵌 sgl-kernel 或 manifest `status=ok` 的源码根。不要使用
site-packages、wheel 解包目录或编译缓存替代；应把正确源码仓加入 manifest。

### Finalize 报行号越界

源码版本可能和 Agent 调查时不同。确认仓库版本、重新打开定义并更新 decisions 中的 `def_line`。

### Agent 认为 KID interface 不正确

不要在 Source Locate 阶段改写 interface。把问题写入 notes，交给 KID Semantic Resolver 或人工
重新生成 KID schema。

### Extract 失败

Extract 会在清理已有 `kernel_sources/` 前完成预检。修正 located schema 的非法结构、路径或行号后
重试即可，已有成功输出不会因预检失败被删除。

## 11. 示例与测试

现有完整示例和 Golden：

```text
example_kernels/source_locate_golden/input/all_backends/decomposition.kid.schema.json
example_kernels/source_locate_golden/config/all_backends/third_party_manifest.json
example_kernels/source_locate_golden/workspaces/all_backends/locate/locate_candidates.schema.json
example_kernels/source_locate_golden/workspaces/all_backends/agent/located.schema.json
example_kernels/source_locate_golden/workspaces/all_backends/extract/decomposition.extracted.schema.json
example_kernels/source_locate_golden/workspaces/all_backends/extract/kernel_sources/
```

运行 Source Location 回归测试：

```bash
PYTHONPATH=. python3 -m unittest \
  framework_engineer.tests.test_source_location_agent_helper \
  framework_engineer.tests.test_source_location_locate \
  framework_engineer.tests.test_source_location_extract
```

这些测试不需要 GPU，覆盖两个公开 CLI、四个私有 helper、当前十 target Golden 和 extract 文件树。

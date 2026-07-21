# 剩余核心开发（remain core dev）

> 当前权威设计见 `kernel_agent_kadai/KID_and_locate_source_desgin_v2.md`。
> 本文只记录尚未完成的模块、输入输出和开发顺序。

## 流水线总览

```text
用户提供单-backend配置目录
  │
  ▼
KID Agent
  ├─ Runtime Capture CLI
  │    raw events + Nsight trace + high→execution stacks
  └─ Semantic Resolution
  │
  ▼
source_locate_golden/input/<case>/decomposition.kid.schema.json
                                       # semantic targets，无 source_locations
  │
  ▼
locate CLI                             # 只定位 Python interface candidates
  │
  ▼
source_locate_golden/workspaces/<case>/locate/locate_candidates.schema.json
  │
  ▼
source_locate Agent                    # 自主定位全部四层，写 decisions
  │
  ▼
finalize helper                        # 校验、剥离 reasoning、生成 notes
  │
  ▼
source_locate_golden/workspaces/<case>/agent/located.schema.json
                                       # 四层 source_locations 完成
  │
  ▼
extract CLI
  │
  ▼
source_locate_golden/workspaces/<case>/extract/
                                       # extracted schema + kernel_sources/
  │
  ▼
snapshot / problem_translate / task_pack
```

`to_fill_after_layer1.json` 只保留为旧分层设计的历史 fixture，不再是正式流水线节点。

---

## 1. KID Runtime Capture CLI

- **路径**：`framework_engineer/kernel_interface_decomposer/`
- **职责**：
  - 为用户指定的 `high_level_target` 打 NVTX range；
  - 在各类 execution common interface 处捕获执行事件；
  - 保存 high-level 到 common interface 的完整 Python 调用链；
  - 用 CUDA Runtime/Driver correlation id 关联 GPU kernel；
  - 支持单-backend 启动命令、多 invocation、warmup 排除和默认
    `unique_decomposition`（另保留 `all/last_n/single`）选择。
- **输入**：单-backend `kid-runtime-config/v3`：`cmd`、`test_cmd`、`target`、profiling/selection；
  `output_dir` 是一级产物根，backend-first 子目录由 KID 自动派生。
- **输出**：raw events 和 `kid-runtime-capture/v1`；Nsight SQLite 是 analyze 临时文件，默认
  成功删除、失败保留，golden/debug 可显式保留。多 backend 由上层串行执行多份配置。

### 已完成

- [x] 独立单文件 Nsight Systems PoC：
  `framework_engineer/kernel_interface_decomposer/nsys_poc.py`。
- [x] 验证 NVTX high/execution、CUDA correlation、SQLite 解析和 GPU duration 聚合。
- [x] PoC 已移除显式 low-level decorator：通过 PyTorch dispatcher 与 Triton launcher 自动捕获，
  并保存 high→execution 调用链。
- [x] 验证一个 execution capture 可关联两个 GPU kernel launch。
- [x] 本地 synthetic SQLite self-test 和远端 H20 实跑通过。
- [x] 自动环境 probe 与逐 case smoke/prewarm；H20/SM90 上 11 个必测 case 全部通过。
- [x] 固化七类 capture：`pytorch_dispatch`、`triton_launch`、`cute_dsl_launch`、
  `tilelang_launch`、`tvm_ffi_call`、`inductor_launch`、`python_binding`。
- [x] 保留 nested capture，并让 GPU kernel 独占归因给包含 launch API 的最底层 capture；本次
  trace 12 个 kernel 全部归因，coverage 为 100%。
- [x] 固化 `CAPTURE_MECHANISMS.md`、registry v2 和人工 semantic golden
  `example_kernels/nsys_poc_kid_golden/`。
- [x] 正式 CLI 已切换为 `capture/analyze`，不再调用 KID 内部 SourceResolver 或生成最终 semantic schema。
- [x] 正式 high-level instrumentation、完整 Python 调用链、七类 adapter、nested capture 和 Runtime/Driver correlation 已接入。
- [x] 实现默认 `unique_decomposition` 收敛与 `all/last_n/single` 可扩展采样；JSONL
  保留全量 invocation，Runtime schema 每种拆分保留末次代表。
- [x] 增加 repeat/softmax 双 invocation PoC、结构 golden、Runtime-only validator，以及
  golden/synthetic 无 GPU 回归测试。
- [x] service 模式改为 paused Nsight session：服务启动并通过 ready 后才开启 Nsight 和
  Runtime event 记录；轻量 GPU E2E 已验证 ready 前 3 次 target 调用不会进入 trace。
- [x] direct 模式统一为 paused session：同一 `test_cmd` 先关闭采集 warmup、再开启采集正式
  执行；Nsight trace 缩减为 `cuda,nvtx`，SQLite 默认仅失败时保留。

### TODO

1. 在真实 SGLang 服务命令上执行正式 `capture` 远端验收，确认启动、ready、stop 与多进程事件文件。
2. 根据真实版本漂移继续扩充 Triton/CuTe/TileLang/Inductor compatibility probe。
3. 新增 third-party Python binding 时扩展显式 export registry，并补对应远端 case。

---

## 2. KID Agent：Semantic Resolution 阶段

- **职责**：读取 high→execution 调用树和源码，选择真正的 semantic low-level target，并完成
  kernel 耗时聚合和热点排序。
- **输入**：Runtime Capture CLI 的 raw events/调用树、Nsight kernel 归因、相关源码、可选 UT/
  provider/repository 信息。
- **输出**：`source_locate_golden/input/<case>/decomposition.kid.schema.json` 或正式
  `decomposition_<backend>.schema.json`。

### Agent 必须完成

1. 在候选调用路径中选择 semantic frontier，不能固定取 high-level 下一层。
2. 识别并跳过透明 wrapper/orchestration interface。
3. 必要时把一个 mid-level 下的多个独立 low-level 拆开。
4. 必要时把共同构成一个完整 ABI 的多个 execution 聚合到一个 semantic target。
5. 填写 semantic `interface` 和 semantic `runtime_event.call_site`。
6. 对可选 `provider` 做兜底；无法确定允许为空。
7. 按关联 GPU kernel duration 之和计算 semantic target 耗时和 share。
8. 保持正式 schema 精简，不写 candidate、execution id、完整 stack、理由或 confidence。

### Schema 迁移

- `archetype`：改为 execution capture 类别，不再是旧 F0–F8 源码形态。
- 删除 `F2|F3` provisional/finalize 语义。
- `binding_provider` 改为可选 `provider` 自由字符串。
- `provider` 不参与 source locate 分派；不能识别可省略或为 `null`。
- `interface/call_site` 必须属于 semantic target，而不是 common execution interface。
- KID 不产 `source_locations` 和 `kernel_sources_dir`。

### TODO

1. 在真实 SGLang Runtime capture 上运行 Prompt Agent，继续积累 transparent wrapper、split
   与 fused semantic 的评测样例。
2. 持续更新 `example_kernels/source_locate_golden/input/<case>/decomposition.kid.schema.json`
   的最终字段语义，并接入上层串行编排。

### 已完成

- [x] 固化精简的 `kid-semantic-resolver-config/v3`、`kid-semantic-context/v1` 和
  `kid-semantic-decisions/v1` 内部合同。
- [x] 新增统一 KID Agent Prompt：接收单-backend配置目录，先调用 Runtime Capture CLI，再执行
  `prepare/decisions/finalize/validate`；Agent 只在 semantic 阶段作判断。
  决议，指标、代表 kernel、rank、coverage 和最终 archetype 确定性生成。
- [x] 实现 direct-owner exact-once、真实 stack-edge call site、confidence、provider 冲突、
  mixed-archetype notes 和最终禁止字段校验。
- [x] Golden 增加 decisions 和 context；现有 11-target final schema 可由 helper 精确复现。
- [x] 增加跨 invocation interface 并集、sample count、AST/path mapping 和 Prompt contract
  的 CPU-only 测试。

---

## 3. source_location CLI 工具包【已完成】

- **路径**：`framework_engineer/source_location/`
- **定位**：source_locate Agent 可直接调用的确定性工具包；代码能力已经完成，不再列为核心待开发模块。
- **入口**：
  - `python -m framework_engineer.source_location.cli locate`
  - `python -m framework_engineer.source_location.cli extract`
- **实现文件**：
  - `locator.py`：KID v2 schema 遍历、interface definition 候选定位、manifest/search-root
    处理和原子输出；
  - `extractor.py`：四层文件复制、range completion、`read_hints.txt`、占位和
    `kernel_sources_dir` 回填；
  - `contracts.py`：候选/Agent/抽取 contract；
  - `cli.py` / `__main__.py`：`locate`、`extract` 命令入口。

### 已完成能力

- [x] KID v2 flat schema 遍历和 KID-owned field 原样保留。
- [x] 基于 callsite/import、module alias、relative import、re-export、
  class method/overload 和 binary re-export 的 interface definition 定位。
- [x] locate candidate 的 `resolved/ambiguous/not_found` 与 Agent final 的
  `resolved/best_effort/missed/not_applicable` 状态。
- [x] locate report、缺失 manifest repo 处理、原子更新和重复运行保护。
- [x] extract 全文件复制、Python/C-family end-line 计算、binding/impl 多 hit 目录化。
- [x] `not_applicable/missed` 占位、严格预检和 `kernel_sources_dir` 回填。
- [x] 公开 CLI 只保留 `locate/extract`。

### 在最新设计中的使用方式

该目录不再被描述为 locate Layer 1/3，而是 source_locate Agent 的工具箱：

- `locate` 的 interface-definition 结果是 Agent 可采纳或修正的候选；
- `extract` 在 Agent 写完四层 `source_locations` 后负责机械抽取；
- `py_cpp_binding/kernel_impl/kernel_header` 的最终语义判断仍归自主 Agent。

现有实现只接受最新 KID v2 和最小四层 contract；旧
`archetype_code/binding_provider/needs_agent/source` 会被拒绝。

---

## 4. source_locate Agent【已完成】

- **类型**：一个自主 Agent，不再划分 locate Layer 1/2/3。
- **载体**：`framework_engineer/skills/source_locate.md` 方法论 +
  `framework_engineer/prompts/start_source_locate.md` 入口 Prompt；不内置 LLM runner。
- **职责**：读取一个 `source-locate-agent-config/v1`，自主编排 locate、源码阅读、finalize 和
  extract，最终负责：
  - `interface_definition`
  - `kernel_impl`
  - `py_cpp_binding`
  - `kernel_header`
- **输入**：一个配置文件，引用 KID schema、`third_party_manifest.json`、sglang root 和独立
  testcase workspace。
- **输出**：`source-locate-agent-decisions/v1`、写入完整 `source_locations` 的 schema、
  `ref/locate_agent_notes.md`、extracted schema 和 `kernel_sources/`。
- **私有 helper**：`prepare-run/inspect-target/search/finalize/evaluate/validate-run`；不加入两个
  公开 CLI。

### 设计约束

- source_locate 不重新选择 semantic target。
- 不按 `archetype/provider` 分派；两者为空时也必须能工作。
- `py_cpp_binding` 全部由 Agent负责，不建设 provider-specific CLI registry。
- `kernel_impl` 允许多 hit，按真实调用链顺序记录，覆盖跨仓和模板/device helper。
- 找不到时允许 `missed/best_effort`，证据和人工建议写入 notes，不扩张主 schema。
- 正式 hit 仅允许来自 SGLang/内嵌 sgl-kernel/manifest `status=ok` 的源码根。
- `interface_definition/kernel_impl` 缺源码时为 `missed`，不得标 `not_applicable`。

### 已完成能力

1. decisions 严格合同：每 target/每层 rationale，每 hit symbol/reason，missed/best-effort 缺口说明。
2. finalize：合法 repo、文件/行号、状态、KID projection 校验，自动 `repo_hint` 与 notes。
3. evaluate：Golden 核心 hits 按序匹配，允许有证据的额外 helper hits。
4. `prepare-run` 从单配置校验输入并派生固定 workspace；`validate-run` 校验完整
   locate→Agent→extract 产物。
5. 当前十 target Golden 可由单配置验证；source-location 相关 40 个 CPU 测试通过。
6. `locate/extract` 仍是两个独立公开 CLI，但由入口 Prompt 内部编排，不要求用户手工串联。

---

## 5. problem_translate Agent

- **定位**：source_locate/extract 的下游消费者。
- **职责**：针对 semantic target，结合 snapshot、UT/reference、四层源码和原仓库，生成
  PyTorch/基础 Python 等价实现与问题定级。
- **输入**：`source_locate_golden/workspaces/<case>/extract/decomposition.extracted.schema.json`、
  同目录 `kernel_sources/`、snapshot、UT、third-party manifest 和源码。
- **输出**：问题定级、reference implementation、后续 task_pack 所需物料。

### 约束

- translate 对象始终是 semantic target，不是单个 runtime component。
- source_locate 的多 `kernel_impl` hit 只是实现证据，不改变 snapshot/task_pack 的外层 ABI。
- 必须在 KID semantic resolution 和 source_locate 定位完成后启动。

---

## 6. 推荐开发顺序

1. 将 PoC 的 high boundary、raw stack event contract、nested capture tree 和 correlation/多 kernel
   聚合接入正式 KID。
2. 实现 Semantic Resolver Agent 与最小评测集。
3. 将 golden 中的 capture `archetype`、可选 `provider`、semantic callsite 迁移到正式 KID schema。
4. 将已完成的 source_locate Agent 接入上层 workspace 编排。
5. 实现 problem_translate Agent。
6. 远端 GPU 端到端跑通：KID → source_locate → extract → problem_translate。

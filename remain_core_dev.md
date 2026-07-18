# 剩余核心开发（remain core dev）

> 当前权威设计见 `kernel_agent_kadai/KID_and_locate_source_desgin_v2.md`。
> 本文只记录尚未完成的模块、输入输出和开发顺序。

## 流水线总览

```text
用户配置
  │
  ▼
KID Runtime Capture CLI
  │  raw events + Nsight trace + high→execution stacks
  ▼
KID Semantic Resolver Agent
  │
  ▼
to_fill_kid.json                       # semantic targets，无 source_locations
  │
  ▼
source_locate Agent                    # 自主定位全部四层，可调用 locator/validator CLI
  │
  ▼
to_fill_locate.json                    # 四层 source_locations 完成
  │
  ▼
extract CLI
  │
  ▼
to_fill_extract.json + kernel_sources/ # 回填 kernel_sources_dir
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
  - 支持单-backend 启动命令、多 invocation、warmup 排除和 `all/last_n/single` 采样。
- **输入**：单-backend `kid-runtime-config/v2`：`cmd`、`test_cmd`、`target`、profiling/selection。
- **输出**：raw events、Nsight SQLite 和 `kid-runtime-capture/v1`；多 backend 由上层串行执行多份配置。

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
- [x] 实现 `all/last_n/single` 可扩展采样、Runtime-only validator，以及 golden/synthetic 无 GPU 回归测试。

### TODO

1. 在真实 SGLang 服务命令上执行正式 `capture` 远端验收，确认启动、ready、stop 与多进程事件文件。
2. 根据真实版本漂移继续扩充 Triton/CuTe/TileLang/Inductor compatibility probe。
3. 新增 third-party Python binding 时扩展显式 export registry，并补对应远端 case。

---

## 2. KID Semantic Resolver Agent

- **职责**：读取 high→execution 调用树和源码，选择真正的 semantic low-level target，并完成
  kernel 耗时聚合和热点排序。
- **输入**：Runtime Capture CLI 的 raw events/调用树、Nsight kernel 归因、相关源码、可选 UT/
  provider/repository 信息。
- **输出**：`to_fill_kid.json` 或正式 `decomposition_<backend>.schema.json`。

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

1. 定义 Agent 输入的内部调用树 contract；该 contract 不进入最终 schema。
2. 实现 Semantic Resolver Agent/skill 和受约束的候选选择输出。
3. 实现 deterministic validator：Agent 选择必须来自已捕获调用树，callsite/文件必须有效。
4. 建立 transparent wrapper、split、fused semantic、direct builtin 等评测 case。
5. 更新 `example_kernels/to_fill_kid.json` 到新字段语义。

---

## 3. source_location CLI 工具包【已完成】

- **路径**：`framework_engineer/source_location/`
- **定位**：source_locate Agent 可直接调用的确定性工具包；代码能力已经完成，不再列为核心待开发模块。
- **入口**：
  - `python -m framework_engineer.source_location.cli locate`
  - `python -m framework_engineer.source_location.cli extract`
- **实现文件**：
  - `locator.py`：schema 遍历、`source_locations` 骨架、interface definition 候选定位、
    manifest/search-root 处理、report 和原子写回；
  - `extractor.py`：四层文件复制、range completion、`read_hints.txt`、占位和
    `kernel_sources_dir` 回填；
  - `contracts.py`：定位/抽取 contract；
  - `cli.py` / `__main__.py`：`locate`、`extract` 命令入口。

### 已完成能力

- [x] flat schema 与 nested invocation schema 遍历。
- [x] 基于 runtime implementation、callsite/import、module alias、relative import、re-export、
  class method/overload 和 binary re-export 的 interface definition 定位。
- [x] `resolved/ambiguous/not_found/not_applicable/missed` 状态和 `repo_hint`。
- [x] locate report、缺失 manifest repo 处理、原子更新和重复运行保护。
- [x] extract 全文件复制、Python/C-family end-line 计算、binding/impl 多 hit 目录化。
- [x] `not_applicable/missed` 占位、`--allow-empty` 和 `kernel_sources_dir` 回填。
- [x] locator + extractor 共 31 个 CPU 单测通过（2026-07-16 复核）。

### 在最新设计中的使用方式

该目录不再被描述为 locate Layer 1/3，而是 source_locate Agent 的工具箱：

- `locate` 的 interface-definition 结果是 Agent 可采纳或修正的候选；
- `extract` 在 Agent 写完四层 `source_locations` 后负责机械抽取；
- `py_cpp_binding/kernel_impl/kernel_header` 的最终语义判断仍归自主 Agent。

现有实现仍兼容旧 schema 中的 `archetype_code/binding_provider/needs_agent/source`。后续若按最新
设计移除这些字段，只属于 contract 迁移和命名清理，不表示 `source_location` 功能尚未完成。

---

## 4. source_locate Agent

- **类型**：一个自主 Agent，不再划分 locate Layer 1/2/3。
- **职责**：调用已完成的 `source_location` CLI，并自主阅读源码，最终负责：
  - `interface_definition`
  - `kernel_impl`
  - `py_cpp_binding`
  - `kernel_header`
- **输入**：KID schema、`third_party_manifest.json`/`missing_repos.md`、sglang/sgl-kernel/
  third-party 源码。
- **输出**：写入完整 `source_locations` 的 schema、`ref/locate_agent_notes.md`，并调用 extract
  生成 `kernel_sources/`。

### 设计约束

- source_locate 不重新选择 semantic target。
- 不按 `archetype/provider` 分派；两者为空时也必须能工作。
- `py_cpp_binding` 全部由 Agent负责，不建设 provider-specific CLI registry。
- `kernel_impl` 允许多 hit，按真实调用链顺序记录，覆盖跨仓和模板/device helper。
- 找不到时允许 `missed/best_effort`，证据和人工建议写入 notes，不扩张主 schema。

### 剩余工作

1. 将自主 Agent/skill 与已完成的 `source_location locate/extract` CLI 串起来。
2. 增加或复用 source_locations validator，供 Agent 提交结果前检查。
3. 按最新设计逐步清理旧 Layer 命名及 `needs_agent/source/archetype` contract；不影响现有 CLI 使用。
4. 更新 `to_fill_locate.json` golden，并将 `to_fill_after_layer1.json` 标为 historical。
5. 用 `example_kernels/all_backends_sglang.py` 做完整的无 GPU Agent 定位评测。

---

## 5. problem_translate Agent

- **定位**：source_locate/extract 的下游消费者。
- **职责**：针对 semantic target，结合 snapshot、UT/reference、四层源码和原仓库，生成
  PyTorch/基础 Python 等价实现与问题定级。
- **输入**：`to_fill_extract.json`、`kernel_sources/`、snapshot、UT、third-party manifest 和源码。
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
4. 实现 source_locate Agent；保留 interface locator/extract 为工具。
5. 简化 source_locations contract 并适配 extractor。
6. 跑 `all_backends_sglang.py` 无 GPU locate 评测。
7. 远端 GPU 端到端跑通：KID → source_locate → extract → problem_translate。

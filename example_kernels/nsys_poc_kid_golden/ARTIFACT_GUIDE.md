# KID Golden 产物填写指南

本指南定义两阶段 KID workspace 中每个文件的责任人、填写依据、格式稳定性、
下游消费者和失败处理。`<backend>` 在本 golden 中为 `nsys_poc`；正式运行时每份
配置也只对应一个 backend。

## 总体责任边界

| 区域 | 主要填写者 | 定位 |
|---|---|---|
| `config/<backend>/` | 用户或上层编排器 | 单-backend 用户输入，不是 KID 运行产物。 |
| `cli_log/<backend>/` | Runtime Capture CLI | 可复查的运行时事实、trace 和日志；不得包含 semantic oracle。 |
| `output/<backend>/` | Semantic Resolver finalizer | KID 对外发布的固定 schema，也是 `source_locate` 的唯一 KID 输入。 |
| `ref/<backend>/` | Helper、Agent、维护者 | Context、decisions 和 notes，仅供解析复现。 |

## 配置输入

### `config/<backend>/runtime_capture_config.json`

- **谁填写**：用户或负责串行调度 backend 的上层编排器。
- **何时填写**：Runtime Capture CLI 启动之前。
- **如何填写**：`backend_name` 必须与四个区域的子目录名一致；`target` 是用户明确
  指定的 high-level Python 接口定义；`cmd` 是可选常驻服务命令，没有服务时填
  `null`；`test_cmd` 必须是唯一的触发命令；`selection` 定义 high-level invocation
  的选择规则（默认按拆分去重并保留末次代表的 `unique_decomposition`，或 `all`、每 stage
  末尾 N 次的 `last_n`、末尾一次的 `single`）；
  `profiling` 定义 Nsight、CUDA Graph 和 trace retention；无服务时同一 `test_cmd`
  会先关闭采集执行一次 warmup，再开启采集执行一次正式测试，因此必须可重复且幂等；
  `output_dir` 指向 `cli_log/<backend>`。本 golden 为 parser 回归显式使用
  `trace_retention=always`，生产默认是 `on_failure`。
- **格式**：固定，`kid-runtime-config/v2`。
- **谁消费**：Runtime Capture CLI。
- **失败处理**：路径、命令或 target 无效时在 profiling 前失败；不能自动猜测另一
  个 backend，也不能把多组命令隐式追加到该配置。

### `config/<backend>/semantic_resolver_config.json`

- **谁填写**：用户或上层编排器；Agent 只读取，不修改输入合同。
- **何时填写**：Semantic Resolver Agent 启动之前。
- **如何填写**：引用同 backend 的 Runtime、context、decisions、最终 output 和 notes
  路径；`sglang_repo_root`、`source_roots` 指向本地源码；`third_party_manifest`
  只引用上游 `resolve-third-party` 的真实产物，本例为
  `kernel_agent/framework_engineer/source_location/example/third_party_manifest.json`，
  不复制进 KID；`runtime_to_local_path_mappings` 将容器、site-packages 路径映射到
  Agent 可读的本地源码。`analysis_source_overrides` 可在 trace 行号与当前源码漂移时，将规范
  发布路径指向冻结的分析快照；它不会改变最终 call site 路径。这些源码仅用于判断 semantic interface 和 provider，
  Semantic Resolver 不输出接口定义、binding 或 kernel implementation 的源码位置。
- **格式**：固定，`kid-semantic-resolver-config/v2`。
- **谁消费**：Semantic Resolver Agent。
- **失败处理**：Runtime、manifest 或显式 source override 不存在时停止；普通 source root
  缺失会记录在 context，若它阻止 semantic 判断则由 Agent 报告 blocker。

## Runtime Capture CLI 产物

### `cli_log/<backend>/environment_probe.json`

- **谁填写**：Runtime Capture CLI 的静态 probe 阶段；fixture 可附加 test 自身的 smoke 摘要。
- **填写依据**：实际远端 Python、GPU、CUDA、Nsight、包版本/API 和 adapter 可用性。
- **填写规则**：只记录环境事实和可选 workload smoke 事实；不得携带 semantic target、
  provider 或 expected archetype 等 PoC 答案。
- **格式**：固定，`kid-runtime-environment/v1`。
- **谁消费**：Runtime Capture CLI 的前置门禁、validator、问题排查人员。
- **失败处理**：Nsight、PyTorch CUDA 或基础环境门禁失败时不启动正式 profiling；
  无服务 workload 的第一次 `test_cmd` 是 warmup，失败体现在 `warmup.log` 和退出码。
  当 `cmd` 非空时，服务以暂停采集的 Nsight session 启动；CLI 在 `ready` 成功后才同时
  开启 Nsight collection 和 Runtime capture gate，因此启动期事件不属于本次产物。

### `cli_log/<backend>/capture_events/events_<pid>.jsonl`

- **谁填写**：安装在通用执行入口上的 capture wrapper。
- **填写依据**：每次 common-interface 进入时的进程/线程、capture parent、archetype、
  execution interface、provider/implementation hint 与 high→execution Python stack。
- **填写规则**：所有 capture 都保留，包括没有 kernel 的辅助调用；每行一个
  `execution_capture`。`call_site_to_next` 表示当前 frame 调用下一 frame 的源码边，
  不能改写为 semantic call site；不得写 `workload_case` 或 `semantic_target_hint`。
- **格式**：固定 JSONL 事件合同。
- **谁消费**：Runtime Capture 聚合器、Semantic Resolver Agent、validator。
- **失败处理**：ID 重复、parent 缺失、stack 为空或事件数与 NVTX 不一致时，该轮
  capture 无效，必须重新运行而不是丢弃异常事件。

### `cli_log/<backend>/trace/profile.sqlite`（golden/调试可选）

- **谁填写**：Runtime Capture CLI 调用 `nsys export --type=sqlite` 生成。
- **填写依据**：与上述 JSONL 同一次 profiling 的 NVTX、CUDA Runtime/Driver API
  和 GPU kernel activity。
- **填写规则**：生产默认在成功 analyze 和校验后删除，失败时保留；golden 和 parser
  调试通过 `trace_retention=always` 保留完整 SQLite，且不能用其他轮次或旧文件替换。
  本合同不保留 `.nsys-rep`，但原始 `nsys.log` 可以记录其临时路径。
- **格式**：Nsight Systems SQLite，不比较二进制字节，只验证关系和数值。
- **谁消费**：Runtime Capture 聚合器；保留时也供疑难回查和 validator 作原始数据交叉校验。
- **失败处理**：SQLite 损坏、correlation 匹配失败或与 JSON 时间不一致时整轮失效并
  重新 profiling。

### `cli_log/<backend>/runtime_capture.schema.json`

- **谁填写**：Runtime Capture CLI 的 SQLite/JSONL join 与聚合阶段。
- **填写依据**：high/execution NVTX、raw capture stack、CUDA launch API 与 GPU kernel
  correlation。
- **填写规则**：
  - `execution_captures` 只结构化与 kernel 归因相关的 capture；完整树仍在 JSONL；
  - `kernel_ids` 和 `direct_*` 只计最底层 owner，不能在父子 capture 重复；
  - `inclusive_*` 可包含后代 kernel，用于上下文分析，但不用于热点总和；
  - 每个 kernel 保存 correlation、原始名称、device/stream、GPU 时间戳、duration、
    launch API 与唯一 owner；
  - `coverage` 是 high-level 全部 kernel 中已归因 duration 的比例；
  - 只给出 `provider_hint`/`implementation_hint`，不提前决定 semantic interface。
- **格式**：固定，`kid-runtime-capture/v1`。
- **谁消费**：Semantic Resolver Agent 和 validator；不直接交给 `source_locate`。
- **失败处理**：coverage 不足、kernel 多 owner、direct/inclusive 聚合不守恒或 SQLite
  对不上时 CLI 失败；默认 `on_failure` 会保留 logs/SQLite 供定位。

### `cli_log/<backend>/logs/probe.log`

- **谁填写**：Runtime Capture CLI。
- **内容**：环境 smoke/prewarm 的原始 stdout/stderr。
- **格式/消费者**：自由文本，仅供开发者和 Agent 排错。
- **失败处理**：probe 失败时它是首要诊断证据，不能用空文件占位。

### `cli_log/<backend>/logs/nsys.log`

- **谁填写**：Runtime Capture CLI。
- **内容**：所有模式的 `nsys launch/start/stop`、`nsys export` 命令和原始输出；正式
  trace 只启用 `cuda,nvtx`，不启用 `osrt`。
- **格式/消费者**：自由文本，供开发者、validator 失败调查使用。
- **失败处理**：Nsight 非零退出、导出失败或 report 路径异常时保留日志并终止。

### `cli_log/<backend>/logs/warmup.log`

- **谁填写**：Runtime Capture CLI。
- **内容**：`cmd=null` 时第一次 `test_cmd` 的输出；服务模式记录 ready 前启动阶段已经完成。
- **格式/消费者**：自由文本，仅供排错，不进入 Runtime 或最终 schema。
- **失败处理**：direct warmup 非零退出时不执行 `nsys start`，并以该日志为首要证据。

### `cli_log/<backend>/logs/test.log`

- **谁填写**：被 profiling 的 test/worker 命令。
- **内容**：该次执行的 workload、设备、依赖版本、checksum 与 adapter 状态摘要。
- **格式/消费者**：自由文本，供 Runtime CLI 和开发者判断测试是否真正完成。
- **失败处理**：命令非零退出或日志缺失时不得继续发布 Runtime schema。

### `cli_log/<backend>/logs/summary.log`

- **谁填写**：Runtime Capture CLI 聚合结束阶段。
- **内容**：high-level 总耗时、coverage、raw/materialized capture 数、kernel 数及直接
  owner 热点列表；不得输出 semantic target 答案。
- **格式/消费者**：稳定的人类可读文本，不作为机器接口。
- **失败处理**：与 Runtime JSON 汇总不一致时以 SQLite 重新解析结果为准，并修复 CLI。

## Semantic Resolver Agent 产物

### `ref/<backend>/semantic_resolver_context.json`

- **谁填写**：`semantic_resolver_tools prepare`，Agent 不手工编辑。
- **填写依据**：Runtime 中所有被选 invocation 的 direct kernel owner、ancestor capture、完整
  stack edge、路径映射、源码/AST call expression 和仓库线索。
- **格式**：固定，`kid-semantic-context/v1`；不得包含 `low_level_id`、最终 semantic interface
  或 normalized kernel name 等答案。
- **谁消费**：Semantic Resolver Agent。
- **失败处理**：Runtime hash 不一致、owner/parent 不完整或分析源码与 stack 行号不匹配时重新 prepare，
  不能沿用旧 context。

### `ref/<backend>/semantic_resolver_decisions.json`

- **谁填写**：Semantic Resolver Agent。
- **填写依据**：Context 的真实 stack edge、源码与 provider/repository 证据。
- **格式**：固定，`kid-semantic-decisions/v1`。每个 target 只含 ID、semantic interface、可选
  provider、normalized name、`high|medium` confidence 和 member 引用；每个 direct owner
  必须恰好分配一次，同一 interface 跨 invocation 合并。
- **谁消费**：deterministic finalizer 和 validator，不交给 `source_locate`。
- **失败处理**：非法 call site、重复/遗漏 owner、低置信度或 provider 冲突都会阻止发布。

### `output/<backend>/decomposition.schema.json`

- **谁填写**：`semantic_resolver_tools finalize`；Agent 只填写 decisions 与 notes。
- **填写依据**：通过验证的 decisions 与 Runtime kernel/capture 事实。
- **填写规则**：
  - `interface` 是适合后续输入输出 dump 和 kernel 优化的 semantic Python 接口；
  - `runtime_event.call_site` 必须来自某条 runtime stack edge 映射后的本地文件/行号；
  - `archetype` 直接继承实际拥有 kernel 的最底层 capture mechanism；
  - `provider` 表示实现源码仓库，无法可靠确定时允许 `null`；
  - 一个 semantic target 下的多个 kernel 聚合为一个 `duration_us`，并选择其中最热
    kernel 的精确 Nsight 名称作为 representative；
  - 相同 interface 跨 stage/call site/invocation 合并；`sample_count` 是出现该 target 的
    invocation 数，share 的分母是全部代表 invocation 的 GPU 总时间；
  - `low_level_id` 是稳定、唯一、只含小写字母/数字/下划线的标识；
  - 候选列表、stack、execution capture id 和自由文本判断不得塞入最终 schema；
  - 不得输出 `implementation`、`source_files` 或 `symbols`。接口定义、Python/C++
    binding 和 kernel implementation 的源码定位全部由后续 `source_locate` 填写。
- **格式**：固定，`kernel-interface-decomposition/v2`。
- **谁消费**：`source_locate`、后续 problem translate 和任务打包流程。
- **失败处理**：无法可靠消歧时 Agent 在 notes 说明阻塞点并请求人工判断；不得用 execution
  interface 冒充 semantic interface。finalizer/validator 不通过时不得手工修正聚合字段。

### `ref/<backend>/kid_semantic_resolver_notes.md`

- **谁填写**：Semantic Resolver Agent。
- **填写依据**：候选 stack、源码阅读、provider 证据、wrapper 消歧、多-kernel 合并和
  confidence 判断。
- **格式**：自由 Markdown，不定义机器 schema。
- **谁消费**：开发者、评审者和人工介入者；`source_locate` 不消费。
- **失败处理**：至少保留可复现决策的关键证据；文件缺失或为空视为 Agent 阶段不完整。

## 维护与验证

`README.md` 和本文件由 KID 维护者更新，不由 CLI 或 Agent 自动覆盖。任何合同字段、
目录责任或消费者发生变化时，必须同步更新文档和
`framework_engineer.kernel_interface_decomposer.artifact_validator`。validator 使用
Python 标准库校验配置/目录、禁止字段、capture tree、聚合守恒、SQLite correlation、
最终 schema、源码路径和跨阶段一致性；退出码非零即表示 golden 不可作为开发基准。

# Nsight PoC KID 两阶段 Golden

这个目录是正式 KID 开发的完整输入/输出基准，数据来自 2026-07-20 使用当前
`nsys_poc.py` 在 NVIDIA H20（SM90）上执行的一次 11-case Nsight Systems PoC。该次执行
捕获 36 个原始 execution capture、13 个与 GPU kernel 相关的结构化 capture、
12 个 GPU kernel，GPU duration sum 为 48.097 us，归因 coverage 为 100%。

## 目录结构

```text
nsys_poc_kid_golden/
├── config/nsys_poc/
│   ├── runtime_capture_config.json
│   └── semantic_resolver_config.json
├── nsys_poc/
│   ├── cli_log/
│   │   ├── environment_probe.json
│   │   ├── runtime_capture.schema.json
│   │   ├── capture_events/events_2274295.jsonl
│   │   ├── trace/profile.sqlite
│   │   └── logs/{probe,nsys,test,summary}.log
│   ├── output/decomposition.schema.json
│   └── ref/
│       ├── semantic_resolver_context.json
│       ├── semantic_resolver_decisions.json
│       └── kid_semantic_resolver_notes.md
├── ARTIFACT_GUIDE.md
└── README.md
```

`config/` 是用户或上层编排器提供的单-backend 输入；`nsys_poc/cli_log` 是 Runtime
Capture CLI 的证据和规范化结果；`nsys_poc/output` 是 KID 对外发布的最终产物；
`nsys_poc/ref` 是 Semantic Resolver 的内部分析证据和自由格式记录。

## KID Agent 数据流

KID 的正式入口是 `framework_engineer/prompts/start_kid.md`，输入为
`config/nsys_poc/` 目录。Agent 依次执行：

1. Runtime Capture CLI 读取 `runtime_capture_config.json`，执行静态环境探测和
   Nsight profiling；无服务时先关闭采集运行一次 `test_cmd` warmup，再开启采集运行正式
   `test_cmd`，CLI 输出完整 Python 调用栈、execution
   capture 树、CUDA correlation 和 GPU kernel activity。
2. Semantic Resolver helper 的 `prepare` 把 direct kernel owner、调用栈边、源码片段
   和仓库线索整理为 context。Agent 只填写 decisions 与 notes；`finalize` 再从 Runtime
   事实确定性计算 representative kernel、耗时、share、rank、coverage 和最终 archetype。
3. `source_locate` 只消费 `nsys_poc/output/decomposition.schema.json`，不读取
Runtime Capture 的 JSONL、SQLite 或 Agent notes。KID 最终产物只交付 semantic
   interface、调用位置和运行时归因；接口定义、binding 与 kernel implementation
   的源码位置由 `source_locate` 在后续阶段补充。

KID 每份配置只处理一个 `cmd`/`test_cmd`。PoC 没有常驻服务，因此
`cmd=null`，同一 `test_cmd` 必须可重复执行：第一次预热，第二次才进入 Nsight/KID
采集窗口。多个 backend 由上层串行执行多份配置，
不会合并到一个 KID 配置中。

Semantic Resolver 的标准命令为：

```bash
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  prepare config/nsys_poc/semantic_resolver_config.json
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  finalize config/nsys_poc/semantic_resolver_config.json
python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
  validate config/nsys_poc/semantic_resolver_config.json
```

本 golden 的 Runtime target、raw stack、semantic context 和 call site 均直接对应当前
`framework_engineer/kernel_interface_decomposer/nsys_poc.py`，不使用 source override 或历史
snapshot。源码行号变化时必须重新生成整套 trace/context/final，不能只修改配置行号。

生产默认在成功 analyze 后删除 `profile.sqlite`；本 golden 为 parser 回归而在配置中显式
使用 `trace_retention=always`，因此保留这份可复查证据。`.nsys-rep` 是导出 SQLite 前的临时
中间文件，不属于 CLI 交付合同，也不保存在 golden 中。GPU 耗时均为关联
kernel activity duration 之和，用于热点排序，不等同于多 stream 端到端 wall time。

正式默认使用 `unique_decomposition`：相同 execution 拆分只保留最后一次代表，不同拆分
分别保留；同时兼容 `all`、`last_n` 和 `single`。本 golden 显式使用 `single`，
等价于每个 stage 选择最后一次 invocation；全部未选调用始终保留在 JSONL 中，SQLite
仅在本 golden 或失败排障等 retention 场景中保留。

## 验证

在仓库根目录运行：

```bash
PYTHONPATH=kernel_agent python3 -m \
  framework_engineer.kernel_interface_decomposer.artifact_validator \
  kernel_agent/example_kernels/nsys_poc_kid_golden
```

validator 不依赖第三方 Python 包，会同时检查目录合同、两阶段字段、capture
归因和聚合、SQLite 原始记录、最终 semantic 结果及外部路径引用。每个文件的
生产者、填写方法和失败处理见 [ARTIFACT_GUIDE.md](ARTIFACT_GUIDE.md)。

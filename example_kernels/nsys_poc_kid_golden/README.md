# Nsight PoC KID 两阶段 Golden

这个目录是正式 KID 开发的完整输入/输出基准，数据来自 2026-07-17 在
NVIDIA H20（SM90）上重新执行的一次 11-case Nsight Systems PoC。该次执行
捕获 36 个原始 execution capture、13 个与 GPU kernel 相关的结构化 capture、
12 个 GPU kernel，GPU duration sum 为 48.096 us，归因 coverage 为 100%。

## 目录结构

```text
nsys_poc_kid_golden/
├── config/nsys_poc/
│   ├── runtime_capture_config.json
│   └── semantic_resolver_config.json
├── cli_log/nsys_poc/
│   ├── environment_probe.json
│   ├── runtime_capture.schema.json
│   ├── capture_events/events_2012382.jsonl
│   ├── trace/profile.sqlite
│   └── logs/{probe,nsys,test,summary}.log
├── output/nsys_poc/decomposition.schema.json
├── ref/nsys_poc/kid_semantic_resolver_notes.md
├── ARTIFACT_GUIDE.md
└── README.md
```

`config/` 是用户或上层编排器提供的单-backend 输入；`cli_log/` 是 Runtime
Capture CLI 的证据和规范化结果；`output/` 是 KID 对外发布的最终产物；
`ref/` 是 Semantic Resolver Agent 的自由格式分析记录。四个目录中的
`nsys_poc` 必须一一对应。

## 两阶段数据流

1. Runtime Capture CLI 读取 `runtime_capture_config.json`，执行静态环境探测和
   Nsight profiling；`test_cmd` 自行在 high-level 外完成 warmup，CLI 输出完整 Python 调用栈、execution
   capture 树、CUDA correlation 和 GPU kernel activity。
2. Semantic Resolver Agent 读取 `semantic_resolver_config.json`、Runtime
   Capture 结果、源码仓库和外部 third-party manifest，从调用栈中识别语义
   接口、call site 和 provider，生成 `decomposition.schema.json` 与分析 notes。
3. `source_locate` 只消费 `output/nsys_poc/decomposition.schema.json`，不读取
   Runtime Capture 的 JSONL、SQLite 或 Agent notes。KID 最终产物只交付 semantic
   interface、调用位置和运行时归因；接口定义、binding 与 kernel implementation
   的源码位置由 `source_locate` 在后续阶段补充。

KID 每份配置只处理一个 `cmd`/`test_cmd`。PoC 没有常驻服务，因此
`cmd=null` 并直接 profiling `test_cmd`。多个 backend 由上层串行执行多份配置，
不会合并到一个 KID 配置中。

`profile.sqlite` 是可复查的 Nsight 证据；`.nsys-rep` 是导出 SQLite 前的临时
中间文件，不属于 CLI 交付合同，也不保存在 golden 中。GPU 耗时均为关联
kernel activity duration 之和，用于热点排序，不等同于多 stream 端到端 wall time。

正式采样支持 `all`、`last_n` 和 `single`。本 golden 的 `single` 等价于每个
stage 选择最后一次 invocation；全部未选调用仍保留在 JSONL/SQLite 中。

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

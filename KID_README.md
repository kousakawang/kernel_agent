# KID 使用指南

KID（Kernel Interface Decomposition）由 KID Agent 统一执行：Runtime Capture CLI
收集 execution-level 事实，Semantic Resolver 再从调用链判断 semantic low-level target。
最终 `kernel-interface-decomposition/v3` 交给 `source_locate`。

## 1. 配置与目录

一次任务只处理一个 backend。准备两个同目录配置：

```text
<workspace>/config/<backend>/
├── runtime_capture_config.json
└── semantic_resolver_config.json
```

Runtime 的 `output_dir` 是所有 backend 共用的一级产物根。KID 自动派生：

```text
<output_dir>/<backend_name>/
├── cli_log/                 # Runtime schema、JSONL、日志、可选 SQLite
├── ref/                     # context、Agent decisions 与 notes
└── output/
    └── decomposition.schema.json
```

多个 backend 可以共享同一 `output_dir`，由上层串行执行多份配置，彼此不会覆盖。
完整实例见 [`example_kernels/nsys_poc_kid_golden`](example_kernels/nsys_poc_kid_golden)。

## 2. Runtime 配置（v3）

完整 direct 示例；可选字段在实际配置中允许省略，以下显式写出默认/空值：

```json
{
  "schema_version": "kid-runtime-config/v3",
  "backend_name": "flashinfer",
  "workdir": "/sgl-workspace/infra_agent/kernel_agent",
  "output_dir": "/sgl-workspace/infra_agent/kernel_agent/kid_output",
  "target": {
    "file": "/usr/local/lib/python3.12/dist-packages/flashinfer/gdn_prefill.py",
    "line": 120,
    "qualified_name": "gdn_prefill"
  },
  "cmd": null,
  "test_cmd": "python3 test_gdn.py",
  "ready": null,
  "stop": null,
  "env": {},
  "selection": {
    "skip_invocations": 0,
    "sample_count_per_stage": 1,
    "sampling": "unique_decomposition",
    "aggregation": "single"
  },
  "profiling": {
    "nsys_bin": "nsys",
    "max_runtime_sec": 1800,
    "disable_cuda_graph": true,
    "min_capture_coverage": 1.0,
    "trace_retention": "on_failure"
  }
}
```

- `workdir` 是命令执行目录；`output_dir` 是一级输出根，不能再填写到 backend 或
  `cli_log`。
- `target.file/line/qualified_name` 唯一标识 high-level Python 定义；pip 安装目录中的
  Python 函数同样支持。纯 C/pybind 接口需要可观测的 Python wrapper。
  KID 默认在目标模块 import 完成后、返回调用方前直接 wrap 该函数，不修改源码；目标是
  直接执行的 `__main__`、无法推导模块名或无法安全替换属性时，自动回退到轻量化的
  `sys.setprofile` 捕获。普通 import-patch 路径不会检查每一次 Python 函数调用。
- `cmd=null` 表示 direct 模式：同一 `test_cmd` 先在关闭采集时执行一次 warmup，再正式
  profile 一次，因此命令必须可重复且幂等。
- `env` 注入 server/test；值必须是字符串且不能为 `null`。
- `sampling` 支持 `unique_decomposition`（默认）、`all`、`last_n`、`single`。
  `skip_invocations` 先跳过最早 N 次；`last_n` 按 Runtime 自动识别的 stage 各取末尾 N 次。
  stage 会出现在结果中，但 v3 禁止配置 `selection.stages`。
- `trace_retention` 为 `on_failure`（默认）、`always` 或 `never`。成功运行默认删除 SQLite；
  golden/parser 开发才使用 `always`。

非空环境变量示例：

```json
"env": {
  "CUDA_VISIBLE_DEVICES": "0",
  "PYTHONPATH": "/sgl-workspace/sglang/python"
}
```

### Service 的 ready/stop

HTTP ready：

```json
"cmd": "python3 -m sglang.launch_server --port 30000 ...",
"ready": {
  "type": "http",
  "url": "http://127.0.0.1:30000/health",
  "timeout_sec": 300
},
"stop": null
```

`ready=null` 表示启动后不等待；`stop=null` 表示退出时发送 `SIGINT`，最多等待 30 秒。
固定等待和自定义停止可写：

```json
"ready": {"type": "sleep", "seconds": 20},
"stop": {"signal": "SIGTERM", "grace_sec": 60}
```

也可用 `{"type":"none"}` 显式不等待。HTTP ready 接受 200–499 状态码。有服务时 KID
在 ready 之后才执行 `nsys start` 和 `test_cmd`。

## 3. Semantic 配置（v3）

Semantic 配置不重复填写 Runtime/output 路径；helper 自动读取同目录
`runtime_capture_config.json`：

```json
{
  "schema_version": "kid-semantic-resolver-config/v3",
  "backend_name": "flashinfer",
  "source_context": {
    "third_party_manifest": "/workspace/third_party_manifest.json",
    "runtime_to_local_path_mappings": []
  }
}
```

`third_party_manifest` 是 `resolve-third-party` 的产物，提供 `sglang_repo_root` 和所有
`repos[].local_path`；KID 只引用，不复制。Runtime 与 Resolver 共享文件系统时 mapping
填 `[]` 或省略。远端 capture、本地解析时填写前缀映射；最长匹配优先，并同时作用于
Runtime artifact 路径和 Python stack/source 路径：

```json
"runtime_to_local_path_mappings": [
  {
    "runtime_prefix": "/mnt/infra_agent",
    "local_prefix": "/Users/example/infra_agent"
  },
  {
    "runtime_prefix": "/usr/local/lib/python3.12/dist-packages/flashinfer",
    "local_prefix": "/Users/example/flashinfer/flashinfer"
  }
]
```

v3 明确拒绝旧的 `runtime_capture`、`source_roots`、`analysis_source_overrides`、
`context_output`、`decisions_output`、`notes_output` 和 `output` 字段。

## 4. 启动 KID Agent

在 Codex 中提交：

```text
请阅读 kernel_agent/framework_engineer/prompts/start_kid.md，
执行 KID，配置目录是：<workspace>/config/<backend>
```

Agent 执行：

```text
capture → prepare → 阅读 context/源码 → 写 decisions/notes → finalize → validate
```

Runtime 必须使用有 CUDA/Nsight 的 Python；本工作区通常用远端 runner `python`。
Semantic helper 可用能访问 Runtime JSON 与本地源码的 `python3`。只有用户明确要求复用且
Runtime validator 通过时，Agent 才跳过 capture。

## 5. 手工排错与验收

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

Runtime CLI 会把阶段进度和长任务 heartbeat 输出到 `stderr`，最终机器可读摘要仍单独写到
`stdout`。默认每 15 秒报告一次长任务状态；需要调整时可设置：

```bash
KID_PROGRESS_INTERVAL_SEC=30 python -m \
  framework_engineer.kernel_interface_decomposer capture \
  <config-dir>/runtime_capture_config.json
```

只有 Runtime 自校验、direct owner exact-once 分配和 Semantic `validate` 全部通过才算完成。
`source_locate` 只读取 `<output_dir>/<backend>/output/decomposition.schema.json`。产物责任边界见
[`ARTIFACT_GUIDE.md`](example_kernels/nsys_poc_kid_golden/ARTIFACT_GUIDE.md)，对接合同见
[`KID_to_source_locate_handoff.md`](kernel_agent_kadai/KID_to_source_locate_handoff.md)。

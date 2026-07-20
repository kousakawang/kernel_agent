# Framework Engineer Phase 1.2 Tests

这些测试用于验证 config 驱动的 snapshot task-pack 生成系统。

CI / 本地测试不使用真实用户配置；它通过 toy target 覆盖 CLI、snapshot、selector、
harness 和 batch launcher 的代码路径。

真实 Framework Engineer Agent 工作时应读取用户配置，并优先按细粒度 CLI 分步执行。
`run-phase1` 是 batch/smoke/稳定复跑入口，不是 agent 的唯一工作方式。

## 本地快速测试

在仓库根目录运行：

```bash
python3 -m unittest discover framework_engineer/tests
python3 -m compileall -q framework_engineer
python3 -m framework_engineer.cli --help
```

CPU toy 测试不需要 GPU；完整 Golden 测试会使用当前开发工作区中与 `kernel_agent` 同级的
SGLang/third-party 源码树。测试覆盖：

- `validate-config` 能校验单目标/多目标配置。
- `run-phase1` 能为多个 target 生成独立 task pack。
- forward boundary 能给 target call 标记 `forward_id`。
- snapshot capture 使用 forward-windowed shape group、bounded samples 和 hit count。
- 自动 mutation diff 能发现原地修改输入。
- `generate-harness` 后初始 correctness smoke pass。
- `validate-task-pack` 检查必需文件、selected snapshots 和 correctness smoke。
- source-location 的 `locate/extract` 两个公开 CLI，以及 Agent 私有
  `inspect-target/search/finalize/evaluate` helper。

当前 source_locate Agent 的方法论和入口分别是：

```text
framework_engineer/skills/source_locate.md
framework_engineer/prompts/start_source_locate.md
```

真实十 target Golden 测试会在本机已有 SGLang/third-party 源码根上验证 decisions finalize 能
精确生成 `example_kernels/source_locate_golden/workspaces/all_backends/agent/located.schema.json`，
并验证 Golden 核心调用链 evaluator。

## GPU 服务器主路径

复制并编辑配置：

```bash
cp framework_engineer/configs/phase1_targets.example.py /tmp/phase1_targets.py
```

至少填写：

```python
task_group_id = "your_task_group"
output_root = "/tmp/your_task_packs"

service_cmd = "python -m sglang.launch_server ..."
workload_cmd = "python /path/to/workload.py ..."

forward_boundary_file = "/path/to/model_or_runner.py"
forward_boundary_line = 123

targets = [
    {"task_id": "target_1", "target_file": "/path/to/file.py", "target_line": 100},
    {"task_id": "target_2", "target_file": "/path/to/file.py", "target_line": 200},
]
```

Agent 模式建议先验证配置，然后按 `prompts/start_phase1_validation.md` 中的细粒度 CLI
逐步执行、检查每一步产物：

```bash
python3 -m framework_engineer.cli validate-config --config /tmp/phase1_targets.py
```

如果配置、服务和 target 已经稳定，或只想做 batch smoke，可以一键运行：

```bash
python3 -m framework_engineer.cli run-phase1 --config /tmp/phase1_targets.py
```

成功后输出：

```text
<output_root>/multi_target_report.json
<output_root>/multi_target_report.md
<output_root>/<target_task_id>/
```

每个 `<target_task_id>/` 都是一个独立 task pack，可单独交给 Kernel Engineer。

## 真实 SGLang 单步 Debug 测试

如果 `run-phase1` 中某个步骤失败，可以用旧的逐条 CLI unittest 复现该步骤：

```bash
cp framework_engineer/tests/real_sglang_phase1_config.example.py \
  /tmp/real_sglang_phase1_config.py

KA_REAL_SGLANG_CONFIG=/tmp/real_sglang_phase1_config.py \
python3 -m unittest framework_engineer.tests.test_real_sglang_phase1
```

这个测试允许通过 `cli_tests` 字典选择要运行的 subcommand。它不是主入口；
它的用途是调试 `probe-target-calls`、`capture-snapshots` 或 `validate-task-pack`
这类单步失败。

## Task Pack 验证

单独验证某个 task pack：

```bash
python3 -m framework_engineer.cli validate-task-pack \
  --task-pack <task_pack> \
  --skip-env-check \
  --run-correctness
```

有 CUDA / Nsight / Triton / CuTe DSL 环境时，可先写入环境 manifest：

```bash
python3 -m framework_engineer.cli probe-env --task-pack <task_pack>
python3 -m framework_engineer.cli validate-task-pack \
  --task-pack <task_pack> \
  --run-correctness \
  --run-benchmark
```

`validate-task-pack` 会输出结构化 checks：必需文件、selected snapshot 完整性、
环境一致性、correctness smoke 和可选 benchmark smoke。

## 运行边界

- Framework Engineer 只校验用户配置和运行链路，不修服务脚本、workload、数据集或框架环境。
- `selected snapshots` 是唯一 replay 来源，`shape_list.json` 只是摘要索引。
- 用户不需要声明 mutable 输入；capture 会保存 pre/post inputs 并自动检测 mutation。
- 多目标 Phase 1.2 只是 launcher：每个 target 生成一个独立 task pack，不做 fusion、merge 或 cross-target benchmark。

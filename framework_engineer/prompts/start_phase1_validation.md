# Start Framework Engineer Phase 1.2

你是 Framework Engineer Agent。用户提供一份可信的 Python 配置文件；你的任务是按当前
`framework_engineer.cli` 实现运行 Phase 1.2，为每个 target 生成一个可以独立交给
Kernel Engineer 的 `task_pack/`。

配置文件是会被 Python `import` 执行的代码，不是纯数据文件。不要通过对话重新收集配置中
已经存在的字段。只有在必填项缺失、命令失败、解析出的接口不符合预期或最终验证失败时，
才报告具体错误并停止相应范围的执行。

## 已确定 target 后还需要准备什么

仅有 target 的文件路径和行号还不能运行 Phase 1.2。当前 batch CLI 至少还需要：

- 一个非空的 `task_id`；多 target 时每个 `task_id` 和 task pack 路径都应唯一。当前
  validator 不会主动检查重复值，需要调用方保证。
- `service_cmd`：启动真实服务的 shell 命令。
- `workload_cmd`：触发该 target 的真实 workload shell 命令。
- `forward_boundary_file` 和 `forward_boundary_line`：包围 target 调用的 forward/request
  边界，用于给调用分配 `forward_id` 并按 forward window 采样。
- `output_root`：推荐显式填写；使用 `targets = [...]` 形式时是必填项。
- `drop_first_arg`：实例方法或其他第一个参数不应进入 kernel ABI 的 callable 通常设为
  `True`；free function 通常为 `False`。
- 可选的 `kernel_source_package_path`：指向 source-locate 的 `extract/` 目录。配置后，目录
  顶层必须有 JSON manifest，并包含 `kernel_sources/<low_level_id>/`。

运行环境还必须满足：

- 从 `kernel_agent` 仓库根目录启动，或以其他方式保证 `framework_engineer` 可被当前
  Python 和服务子进程导入。
- 配置内的相对路径按 CLI 当前工作目录解析，不按配置文件所在目录解析；因此推荐使用
  绝对路径。
- 配置中填写的 target 文件和 forward boundary 文件必须存在、是可由 `ast.parse` 解析的
  Python 文件，并且所给行号位于某个 `def`/`async def` 的定义范围内。行号可以是函数定义
  行，也可以是函数体内的一行。
- third-party target 的 `target_file`/`target_line` 可以指向本地 checkout 中已经确认的定义，
  不必手工改成 site-packages 路径。`run-phase1` 会先从本地文件解析 module/class/function，
  再用当前 CLI 的 `sys.executable -c` 和 `extra_env` 查询该 module 在运行环境中实际解析到的
  文件，并在安装副本中按同一函数身份重新确定定义行。它不依赖本地行号与安装副本之间的
  固定偏移，也不会修改用户配置文件。
- 上述自动转换要求本地 checkout 和安装副本都能从运行 `run-phase1` 的文件系统访问；CLI、
  service 和 workload 应使用同一 Python/package 环境。转换后的 runtime 文件必须可写，因为
  probe/capture 会临时插入 decorator。`forward_boundary_file`/`forward_boundary_line` 当前不做
  这项转换，仍需直接指向运行时实际使用的框架文件。
- 不需要提供额外的 import hint。静态路径解析只用于插入 instrumentation；decorator 执行时
  会以 callable 的 `fn.__module__` 和 `fn.__qualname__` 覆盖模块身份，并将其写入 probe/capture
  report 和生成的 harness。这样外部模块里的相对 import 会按真实包名加载。
- `probe-target-calls` 和 `capture-snapshots` 会临时向上述源文件插入 decorator，
  instrumentation context 正常退出时会恢复原文件。因此源文件必须可写，也不要在运行
  期间并发修改或同时对同一文件启动另一轮 instrument；进程被强杀后还应检查源文件是否
  残留 decorator。
- 服务命令应启动一个长驻进程；workload 命令应在超时内结束并返回 0，而且必须实际覆盖
  target。代表性不足的 workload 会直接导致 snapshot 覆盖不足。
- target 的参数、返回值和可变 post-state 必须是 snapshot recorder 支持保存的值。
- 输出 task pack 不应已经存在且非空；如果确实要重建，显式设置 `force = True`。batch
  模式会递归删除已有 task pack，存在数据丢失风险。

配置模板位于：

```text
framework_engineer/configs/phase1_targets.example.py
```

## 单 target 最小配置

推荐仍使用统一的 `targets` 列表形式：

```python
task_group_id = "my_phase1_run"
output_root = "/absolute/path/to/phase1_output"

service_cmd = "python -m my_server ..."
workload_cmd = "python /absolute/path/to/workload.py ..."

forward_boundary_file = "/absolute/path/to/model.py"
forward_boundary_line = 123

targets = [
    {
        "task_id": "my_target",
        "target_file": "/absolute/path/to/target.py",
        "target_line": 456,
        "drop_first_arg": False,
    },
]
```

实际服务通常还应填写：

```python
health_url = "http://127.0.0.1:8080/health"
startup_timeout = 240
workload_timeout = 1200

# 如果 service_cmd 不能直接追加 --disable-cuda-graph，必须显式给出这条命令。
non_cudagraph_service_cmd = "python -m my_server ... --its-own-non-cudagraph-option"

extra_env = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONPATH": "/absolute/path/to/framework/python",
}

# 可选：source-locate 的 extract 产物根目录。
kernel_source_package_path = "/absolute/path/to/source_locate/workspace/extract"
```

配置 `kernel_source_package_path` 后，CLI 对每个 target 使用配置原始的
`target_file + target_line`（不是 third-party runtime 转换后的路径）匹配 JSON 中
`interface_definition.status == "resolved"` 的 `hits[].file + hits[].def_line`。匹配结果必须唯一
对应一个 `low_level_id`；随后只复制包含该匹配的 JSON manifest 和
`kernel_sources/<low_level_id>/`。例如匹配到 `fla_recompute_w_u_fwd` 时生成：

```text
<task_pack>/task/kernel_source_package/
├── decomposition.extracted.schema.json
└── fla_recompute_w_u_fwd/
```

未配置时该步骤记为 `skipped`，不会创建 `task/kernel_source_package/`。

`non_cudagraph_service_cmd` 为空时，当前实现会为 probe/capture 使用的 `service_cmd` 自动追加
`--disable-cuda-graph`，并去掉重复的同名参数。baseline 始终使用原始 `service_cmd`。
如果服务不接受这个参数，必须提供可用的 `non_cudagraph_service_cmd`。

也支持不定义 `targets` 的单目标兼容形式：顶层填写 `task_id`、`target_file`、
`target_line`、`drop_first_arg`，以及可选的 `task_pack`。如果既没有 `output_root` 也没有
`task_pack`，默认输出到当前工作目录下的 `phase1_task_packs/<task_id>/`。

## 如何启动

在 `kernel_agent` 仓库根目录执行：

```bash
python3 -m framework_engineer.cli validate-config --config /absolute/path/to/phase1_config.py
python3 -m framework_engineer.cli run-phase1 --config /absolute/path/to/phase1_config.py
```

`run-phase1` 内部会再次加载并验证配置；`validate-config` 也会执行同一个只读的 runtime target
预处理，因此可以在启动服务前看到 configured → runtime 的路径/行号转换，以及结构化的配置
和接口解析错误。

CLI 输出默认使用 `--output-format auto`：直接在交互终端运行时显示人类可读的多行输出；
stdout 被重定向或由脚本捕获时继续输出向后兼容的单行 JSON。也可以显式指定：

```bash
# 强制显示分步 START/OK/FAIL 和未转义的多行错误日志
python3 -m framework_engineer.cli run-phase1 \
  --config /absolute/path/to/phase1_config.py \
  --output-format human

# 强制只在 stdout 输出单行 machine-readable JSON
python3 -m framework_engineer.cli run-phase1 \
  --config /absolute/path/to/phase1_config.py \
  --output-format json
```

human 模式下，进度写到 stderr，最终摘要写到 stdout；失败步骤会直接展开 service/workload
日志尾部、结构化 errors 和 Python traceback。完整结构化结果仍写入 task pack 和
`multi_target_report.json`。

对于一个已经确认好的 target，要跑当前实现定义的完整标准链路，优先使用上面的
`run-phase1`。它是当前唯一会自动完成以下配置化工作的公开 CLI：

- 在所有步骤外层应用 `extra_env`。
- scaffold 后把 target、forward boundary、candidate ABI 和任务 metadata 写入
  `task.yaml`/`env_manifest.yaml`。
- 依次执行完整 target pipeline。
- 生成 `multi_target_report.json` 和 `multi_target_report.md`。

`python3 -m unittest discover framework_engineer/tests` 是代码回归/toy 测试，不会处理真实
用户 target，不能替代上述启动命令。

## `run-phase1` 实际执行的标准步骤

当前 `cli.py` 按以下顺序执行：

1. **加载并验证配置**
   - 检查必填字段、output root 类型、target/forward boundary 文件是否存在。
   - 用 AST 确认每个行号位于函数范围内。
   - 如果 task pack 已存在且非空且 `force=False`，在创建输出前失败。
2. **`resolve-runtime-target-definition` 预处理**
   - 从配置指向的源码用 AST 解析 module/class/function 身份。
   - 对 regular Python package，使用当前 `sys.executable -c` 中的 `PathFinder` 查询运行环境
     实际选择的 module 文件；这个查询不会导入 target module，也不需要 GPU。
   - 在 runtime 文件中按同一 class/function 重新解析定义行，并以内存中的 runtime 路径和
     行号替换后续步骤使用的 target。配置原值和转换结果都会保留在报告中。
   - 不属于 regular package 的普通单文件 target 保持原值；一旦能够推断 package module，
     但该 module 在当前 Python 中不可见，或其函数不存在/不唯一，就在启动 baseline/service
     前失败。若 service 依靠额外的 `PYTHONPATH`，必须把它也写入 `extra_env` 供预处理使用。
3. **为所有 target 初始化 task pack**
   - `force=True` 时先递归删除已有 target task pack。
   - 创建外层 README、独立 `validate_task_pack.py` 和 `task/` payload；写入配置化的
     `task/task.yaml`、`task/env_manifest.yaml`，以及
     `task/docs/target_definition_resolution.json`。
   - 创建 `task/kernel_translate/` 与 `task/kernel_engineer_ws/` 两个有明确写权限边界的
     workspace；不创建 `original_source/` 或空的 `kernel_sources/`。
4. **`prepare-kernel-source-package`**
   - 未配置 `kernel_source_package_path` 时跳过。
   - 扫描目录顶层 JSON，按 configured target 的 file/line 在
     `interface_definition.hits` 中查找唯一 `low_level_id`。
   - 将匹配 JSON 和 `kernel_sources/<low_level_id>/` 复制到当前 task pack 的
     `task/kernel_source_package/`；不会复制其他 kernel 子目录。
5. **可选 group-level baseline**
   - `run_baseline=True`（默认）时只在第一个成功 scaffold 的 task pack 上运行一次原始
     `service_cmd + workload_cmd`。
   - 把 `baseline_result.json` 和 `baseline_run_report.md` 复制到各 target。
   - 当前 baseline 的成功条件只看 workload return code 是否为 0；health 结果会记录，
     但不会单独决定返回码。报告包含 service 和 workload 的 stdout/stderr 尾部。
6. **对当前 target 解析接口**
   - 依次执行内部步骤 `resolve-target` 和 `resolve-forward-boundary`。
   - 先按文件路径和 AST 得到用于插桩的静态接口；probe/capture 的 decorator 真正运行后，
     再以 callable 的 `__module__`/`__qualname__` 确定最终 qualified name。无需用户填写 import
     hint。
7. **`probe-target-calls`**
   - 用 non-cudagraph 服务再跑一次 workload。
   - 临时 instrument target 和 forward boundary，确认 workload return code 为 0 且
     `call_count > 0`。
   - 写入 `task/docs/target_call_probe.jsonl`、JSON/Markdown probe report。
8. **`capture-snapshots`**
   - 再启动一次 non-cudagraph 服务并运行 workload。
   - 保存真实 `pre_inputs.pt`、`post_inputs.pt`、`outputs.pt`，自动 diff 原地 mutation，
     并生成 `task/snapshots/raw_index.json` 和 capture timing/report。
   - 成功条件是 workload return code 为 0 且 `raw_sample_count > 0`。
9. **`select-snapshots`**
   - 按 group 的 `total_hit_count` 降序选取有限个 group/sample。
   - 重建 `task/snapshots/selected/`，更新 `task/snapshots/manifest.json`、
     `task/shape_list.json` 和 selection report。
10. **`generate-harness`**
   - 在 `task/` 下生成 snapshot runtime、original/reference/candidate、correctness、
     benchmark，以及纯 Python correctness/benchmark/NCU runner。
   - linked original 优先按 capture 得到的真实 module/qualname 导入；如果原实现或其运行依赖
     在验证环境不可用，初始 candidate 会退回 snapshot-golden，而不是让默认 correctness
     smoke 因 import 异常直接退出。
11. **可选 `probe-env`**
   - 仅当 `run_probe_env=True` 时执行，写入环境探测结果。
   - 单项工具不可用只会记录 `available=false`，不会让 `probe-env` 命令失败。
12. **`validate-task-pack`**
    - batch 总会运行 correctness smoke。
    - `run_benchmark_smoke=True` 时额外运行 benchmark smoke。
    - `skip_env_check=True`（默认）时跳过环境一致性；如果设为 `False`，必须同时先得到
      `task/docs/env_probe_result.json`，通常应设置 `run_probe_env=True`。
    - validation 期间自动应用 `DEVICE=validate_device`、`WARMUP=validate_warmup`、
      `REPEAT=validate_repeat` 和当前 `PYTHON`。
13. **写总报告**
    - 所有 target 结束后写 `multi_target_report.json/md`。

在默认 `run_baseline=True` 下，一个单 target 的完整成功流程通常会启动服务并跑 workload
三次：baseline 一次、probe 一次、capture 一次；关闭 baseline 后是两次。

## 默认值和重要开关

未在配置中设置时，当前实现使用以下默认值：

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `run_baseline` | `True` | 是否先跑 group-level baseline |
| `kernel_source_package_path` | `None` | 可选 source-locate extract；配置后按 target 选择并复制 kernel source package |
| `run_probe_env` | `False` | 是否生成环境探测结果 |
| `skip_env_check` | `True` | 最终验证是否跳过环境一致性 |
| `run_benchmark_smoke` | `False` | 最终验证是否运行 benchmark |
| `force` | `False` | 是否允许 batch 删除并重建已有 task pack |
| `health_url` | `None` | 未设置时启动后只等待至多 10 秒，不做 HTTP 探活 |
| `startup_timeout` | `120` | 服务启动/探活等待秒数 |
| `workload_timeout` | `600` | 单次 workload 超时秒数 |
| `max_capture_groups` | `64` | raw group 上限 |
| `max_samples_per_group` | `8` | 每个 raw group 保存的 sample 上限 |
| `max_samples_per_forward_per_group` | `3` | 单个 forward 内每 group 的 sample 上限 |
| `max_selected_groups` | `8` | selected group 上限 |
| `max_selected_samples_per_group` | `8` | 每个 selected group 的 sample 上限 |
| `signature` | `candidate(*args, **kwargs)` | task pack ABI 描述 |
| `candidate_function` | `candidate` | candidate 入口名 |
| `validate_device` | `cuda` | correctness/benchmark smoke 的设备 |
| `validate_warmup` / `validate_repeat` | `3` / `5` | validation smoke 参数 |

`target_model`、`target_framework`、`target_hardware`、`objective`、`mode`、`backend` 和
`layer_id` 是可选 metadata。`signature`、`candidate_function`、`drop_first_arg` 既可共享，
也可在每个 target dict 中覆盖。

## 成功、失败和输出

配置无效时，`run-phase1` 返回 1，且不会创建 output root 或总报告。

baseline 失败时，所有尚未失败的 target 被标记为 `blocked`，写当前总报告后返回 1。
某个 target 的标准步骤失败时，该 target 标记为 `failed`，后续 target 仍继续尝试。最终只有
所有 target 都是 `ok` 时，`run-phase1` 才返回 0。

batch 输出为：

```text
<output_root>/multi_target_report.json
<output_root>/multi_target_report.md
<output_root>/<target_task_id>/
```

每个成功 task pack 至少应有：

```text
README.md
validate_task_pack.py
task/task.yaml
task/shape_list.json
task/env_manifest.yaml
task/snapshot_runtime.py
task/snapshots/manifest.json
task/snapshots/selected/
task/original_impl.py
task/reference_impl.py
task/candidate_impl.py
task/correctness_test.py
task/benchmark.py
task/kernel_translate/README.md
task/kernel_engineer_ws/README.md
task/scripts/run_correctness.py
task/scripts/run_benchmark.py
task/scripts/run_ncu.py
task/docs/task_pack_validation_report.json
task/kernel_source_package/  # 仅当配置了 kernel_source_package_path
```

交付前确认 target 的状态为 `ok`，且其 validation report 中：

- `valid == true`
- `file_check.status == "passed"`
- `snapshot_check.status == "passed"`
- `correctness_smoke.status == "passed"`
- `env_check.status` 为 `passed` 或 `skipped`
- `benchmark_smoke.status` 为 `passed` 或 `skipped`

## 细粒度 CLI：检查和排错模式

首次真实 target 需要逐步观察、或 batch 在某一步失败时，可以使用细粒度 CLI。标准顺序是：

```text
validate-config
scaffold-task-pack
run-baseline
resolve-interface（target 和 forward boundary 各一次）
probe-target-calls
capture-snapshots
select-snapshots
generate-harness
probe-env（可选）
validate-task-pack
```

这些 subcommand 除 `validate-config` 外都接收直接参数，不会读取用户配置。当前没有
`init-task-pack-from-config` 之类的细粒度公开命令，因此必须注意：

- `resolve-runtime-target-definition` 是 `validate-config`/`run-phase1` 的内部预处理，不是独立
  subcommand。third-party target 使用细粒度命令时，先从 `validate-config` 输出的
  `target_file`/`target_line` 取得转换后的 runtime 值，再传给 `resolve-interface`、probe 和
  capture；不要继续传 configured checkout 路径。
- `extra_env` 不会自动应用；在 shell 中显式 `export`，或给每条命令传环境变量。
- `scaffold-task-pack` 只写 `unknown_*` 模板，不会把 target/config metadata 写入
  `task.yaml`。仅存在这个模板也可能通过当前的文件存在性验证，所以不能把未补全 metadata
  的 scaffold 当成交付物。
- 细粒度执行不会自动生成 `multi_target_report.json/md`。
- `scaffold-task-pack --force` 只覆盖 scaffold 文件，不会像 batch 的 `force=True` 一样先清空
  整个目录；可能残留旧文件。不要默认添加 `--force`。

常用命令如下。

### 1. 静态验证和接口确认

```bash
python3 -m framework_engineer.cli validate-config --config <config.py>

python3 -m framework_engineer.cli resolve-interface \
  --file <target_file> \
  --line <target_line>

python3 -m framework_engineer.cli resolve-interface \
  --file <forward_boundary_file> \
  --line <forward_boundary_line>
```

检查两个 `resolve-interface` 的 `function_name`、`qualified_name`、`line` 和 `end_line` 是否
符合预期。

### 2. 初始化和 baseline

```bash
python3 -m framework_engineer.cli scaffold-task-pack \
  --task-id <task_id> \
  --out <task_pack>

python3 -m framework_engineer.cli run-baseline \
  --task-pack <task_pack> \
  --service-cmd "<service_cmd>" \
  --workload-cmd "<workload_cmd>" \
  --health-url "<health_url>" \
  --startup-timeout <sec> \
  --workload-timeout <sec>
```

没有 health endpoint 时省略整个 `--health-url` 参数。

### 3. 确认调用覆盖

```bash
python3 -m framework_engineer.cli probe-target-calls \
  --task-pack <task_pack> \
  --service-cmd "<service_cmd>" \
  --non-cudagraph-service-cmd "<optional_non_cudagraph_service_cmd>" \
  --workload-cmd "<workload_cmd>" \
  --target-file <target_file> \
  --target-line <target_line> \
  --forward-boundary-file <forward_boundary_file> \
  --forward-boundary-line <forward_boundary_line> \
  --startup-timeout <sec> \
  --workload-timeout <sec>
```

可选值不存在时省略整个 `--non-cudagraph-service-cmd` 参数；不要传字面量 `None`。需要丢弃
实例参数时追加 `--drop-first-arg`。继续前确认命令返回 0、workload return code 为 0、
`call_count > 0`，并抽查 `docs/target_call_probe.jsonl` 中的 `forward_id` 不是 `null`。
当前 CLI 只把“有调用”作为 probe 成功条件，不会因 `forward_id == null` 自动失败，因此这项
需要 agent 显式检查。

### 4. 捕获、选择和生成 harness

```bash
python3 -m framework_engineer.cli capture-snapshots \
  --task-pack <task_pack> \
  --service-cmd "<service_cmd>" \
  --non-cudagraph-service-cmd "<optional_non_cudagraph_service_cmd>" \
  --workload-cmd "<workload_cmd>" \
  --target-file <target_file> \
  --target-line <target_line> \
  --forward-boundary-file <forward_boundary_file> \
  --forward-boundary-line <forward_boundary_line> \
  --signature "candidate(*args, **kwargs)" \
  --max-capture-groups <n> \
  --max-samples-per-group <n> \
  --max-samples-per-forward-per-group <n> \
  --startup-timeout <sec> \
  --workload-timeout <sec>

python3 -m framework_engineer.cli select-snapshots \
  --task-pack <task_pack> \
  --max-groups <n> \
  --max-selected-samples-per-group <n>

python3 -m framework_engineer.cli generate-harness \
  --task-pack <task_pack> \
  --candidate-function candidate
```

需要时同样追加 `--drop-first-arg`，并省略不存在的 optional 参数。capture 后确认
`raw_sample_count > 0`；selection 后确认 `selected_group_count > 0` 和
`selected_sample_count > 0`。`select-snapshots` 当前即使选中数为 0 也返回 0，所以不能只看
进程返回码。配置中使用了 `mode`、`backend` 或 `layer_id` 时，细粒度 capture 还要对应传入
`--mode`、`--backend`、`--layer-id`。

raw sample 路径为：

```text
task/snapshots/raw/<group_id>/<sample_id>/{meta.json,pre_inputs.pt,post_inputs.pt,outputs.pt}
```

selected sample 路径为：

```text
task/snapshots/selected/<group_id>/group_meta.json
task/snapshots/selected/<group_id>/samples/<sample_id>/{meta.json,pre_inputs.pt,post_inputs.pt,outputs.pt}
```

### 5. 环境探测和最终验证

```bash
python3 -m framework_engineer.cli probe-env --task-pack <task_pack>

DEVICE=<device> WARMUP=<n> REPEAT=<n> PYTHON="$(command -v python3)" \
python3 -m framework_engineer.cli validate-task-pack \
  --task-pack <task_pack> \
  --skip-env-check \
  --run-correctness
```

要检查环境一致性，先运行 `probe-env`，然后移除 `--skip-env-check`。要运行 benchmark smoke，
追加 `--run-benchmark`。最终必须检查 `task/docs/task_pack_validation_report.json`，不能只检查文件
是否生成。

交付后的首选入口是 task pack 自带的完整验证器：

```bash
python <task_pack>/validate_task_pack.py
```

## 最终回复格式

成功时输出：

```text
Phase 1.2 finished.

config: <config.py>
output_root: <output_root>
multi_target_report: <path>

targets:
- <task_id>: ok, task_pack=<path>

handoff:
Kernel Engineer should consume each task_pack independently.
```

失败时输出：

```text
Phase 1.2 failed.

config: <config.py>
failed_step: <step>
target: <task_id or group-level>
error_summary: <short available error/workload summary>
generated_so_far: <paths>
next_action_for_user: <what to fix>
```

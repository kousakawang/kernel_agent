# Phase 1 Snapshot Design

本文定义 Phase 1 中 `snapshot` 的标准形态。它是 Framework Engineer 和 Kernel Engineer 之间最重要的数据合同之一：Framework Engineer 负责从真实框架调用中录制 snapshot，Kernel Engineer 只基于这些 snapshot 做 correctness、benchmark 和优化。

## 1. 设计目标

Snapshot 需要解决三个问题：

1. 真实性：输入来自真实 SGLang workload，而不是 Kernel Engineer 随机编造。
2. 可重放：脱离 SGLang 服务后，仍能在 task pack 中重建 kernel-level 输入。
3. 可验证：既能比较输出，也能比较被原地更新的输入状态。

Phase 1 推荐从 `shape trace` 升级为 `snapshot replay`。shape 仍然重要，但它只是 snapshot 的索引和收敛依据，不再是 UT/benchmark 的唯一输入来源。

## 2. Snapshot 文件布局

推荐在 task pack 中使用如下目录：

```text
task_pack/
  snapshots/
    manifest.json
    raw/
      call_000001/
        meta.json
        pre_inputs.pt
        post_inputs.pt
        outputs.pt
      call_000002/
        ...
    selected/
      case_0001/
        meta.json
        pre_inputs.pt
        post_inputs.pt
        outputs.pt
      case_0002/
        ...
```

含义：

- `raw/` 保存采集到的原始调用样本。
- `selected/` 保存最终进入 UT/benchmark 的核心 case。
- `manifest.json` 保存全局索引、schema 版本、目标接口、选择策略和 hash 规则。

Kernel Engineer 默认只读取 `selected/`，不依赖完整 `raw/`。

## 3. 逻辑类设计

这些类是数据合同设计，不要求第一版一定用 Python class 实现；CLI 可以先用 JSON/PT 文件落盘。但工具和文档都应该按这些概念组织。

### SnapshotTensorMeta

描述一个 tensor 的结构信息：

```text
SnapshotTensorMeta:
  path: "args.q" | "kwargs.ssm_states" | "outputs.0"
  dtype: "bfloat16"
  shape: [total_tokens, num_heads, head_dim]
  stride: [...]
  device_type: "cuda"
  device_index: 0
  layout: "strided"
  is_contiguous: true | false
  requires_grad: false
  numel: 123456
  storage_offset: 0
```

注意：

- `device_index` 只记录采集位置，replay 时不应强绑定同一个 GPU id。
- `stride` 和 `storage_offset` 必须保存，因为 kernel 性能和正确性可能依赖 layout。
- 非 contiguous tensor 需要在 replay 时尽量恢复原 layout；无法恢复时必须在 validation report 中报错，而不是静默转 contiguous。

### SnapshotValueRef

描述一个被保存的数据对象：

```text
SnapshotValueRef:
  path: "args.q"
  kind: "tensor" | "primitive" | "list" | "tuple" | "dict" | "none"
  tensor_meta: SnapshotTensorMeta | null
  file: "pre_inputs.pt"
  object_key: "args.q"
```

Tensor 数据存储建议：

- 使用 `torch.save` 保存 CPU copy。
- 保存时使用 `tensor.detach()`，禁止保留 autograd graph。
- CPU copy 本身会同步 GPU 数据；采集工具仍建议在接口调用前后显式 `torch.cuda.synchronize()`，让时间和数据边界更清楚。

### SnapshotCase

一个可 replay 的接口调用样本：

```text
SnapshotCase:
  schema_version: "phase1.snapshot.v1"
  task_id: "qwen35_gdn_extend_core_h20_20260621"
  case_id: "case_0001"
  raw_call_ids: ["call_000013", "call_000219", ...]
  target:
    qualified_name: "sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.extend"
    logical_name: "gdn_extend_core_v1"
    mode: "extend"
    layer_id: 12
    backend: "triton"
  interface:
    signature: "candidate_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc)"
    args_tree: ...
    kwargs_tree: ...
    output_tree: ...
  files:
    pre_inputs: "pre_inputs.pt"
    post_inputs: "post_inputs.pt"
    outputs: "outputs.pt"
  mutation:
    mutable_arg_paths: ["kwargs.ssm_states"]
    compare_mutations: true
  hashes:
    shape_hash: "..."
    semantic_hash: "..."
    value_hash: "..."
    case_key: "..."
  selection:
    call_count: 37
    priority: "required"
    reason: "top_frequency"
  tolerance:
    atol: 0.02
    rtol: 0.02
```

### SnapshotStore

负责落盘和读取：

- `save_raw_call(snapshot_case)`
- `load_raw_call(call_id)`
- `save_selected_case(snapshot_case)`
- `load_selected_case(case_id, device)`
- `list_cases(priority=None)`

第一版可以是普通 Python 模块，不需要独立服务。

### SnapshotRecorder

负责在插桩装饰器里捕获调用：

1. 在接口调用前保存 `pre_inputs`。
2. 调用原始接口，得到 `outputs`。
3. 在接口调用后保存 `post_inputs`。
4. 生成 `meta.json`。
5. 根据 hash 判断是否保存完整数据，避免同类 case 无限增长。

### SnapshotSelector

负责从 `raw/` 收敛到 `selected/`：

- 按 `semantic_hash` 分组。
- 记录每组调用频率。
- 保留 top frequency case。
- 保留最大 token、最大 batch、非整齐 shape、边界 shape。
- 输出 `selected/` 和 `manifest.json`。

### SnapshotHarnessBuilder

负责从 selected snapshots 生成 UT 和 benchmark：

- 生成 `reference_impl.py`。
- 生成 `candidate_impl.py`。
- 生成 `correctness_test.py`。
- 生成 `benchmark.py`。
- 生成 `scripts/run_correctness.sh`、`scripts/run_benchmark.sh`、`scripts/run_ncu.sh`。

## 4. Hash 设计

Snapshot 至少需要三类 hash。不要只用 shape hash，否则会把语义不同但 shape 相同的 case 合并掉。

### shape_hash

用于粗粒度归类：

```text
shape_hash = hash(
  target.logical_name,
  interface tree structure,
  tensor dtype,
  tensor shape,
  tensor stride,
  tensor storage_offset,
  tensor is_contiguous,
  primitive argument type/value when value affects dispatch
)
```

不建议包含：

- 完整 tensor 数值。
- GPU device index。
- Python object id 或 pointer。

### semantic_hash

用于决定是否是同一个优化 case。它包含 shape hash，并加入少量接口语义摘要。

对 Qwen3.5/GDN linear attention，建议加入：

- `mode`: extend/decode/packed_decode。
- `layer_id` 是否参与分组：默认可记录，但不一定必须分组；如果不同 layer 的参数 layout 或 state shape 不同，则纳入。
- `query_start_loc` 的 diff 列表或 diff histogram。
- `cache_indices` 的 min/max、是否含 `-1`、unique count、是否连续。
- `ssm_states` 的 slot 数、state shape、stride。
- `q/k/v/g/beta` 的 dtype/layout。
- backend 名称。

```text
semantic_hash = hash(shape_hash, semantic_features)
```

### value_hash

用于数据完整性和完全重复样本去重：

```text
value_hash = hash(sampled_or_full_tensor_bytes, primitive_values)
```

规则：

- 小 tensor 可以 full hash。
- 大 tensor 可以分块采样 hash，例如 head/middle/tail chunk。
- `value_hash` 不应用来合并优化 case，只用于确认保存的数据是否被破坏，以及去掉完全重复的 raw call。

### case_key

最终 selected case 的稳定 key：

```text
case_key = short_hash(schema_version, target.logical_name, semantic_hash)
case_id = "case_" + ordinal + "_" + case_key[:8]
```

## 5. 原地更新输入的 UT 设计

很多 kernel 不是纯函数，例如会更新 `ssm_states`。因此 correctness 不能只比较返回值。

标准 correctness 流程：

1. 从 snapshot 加载 `pre_inputs`。
2. 为 reference 和 candidate 各 clone 一份输入。
3. 运行 reference。
4. 运行 candidate。
5. 比较 reference output 与 candidate output。
6. 对 `mutable_arg_paths` 中的输入，比较 reference post state 与 candidate post state。
7. 可选：reference output/post state 与 snapshot 捕获的 `outputs`/`post_inputs` 做 self-check。

伪代码：

```python
case = load_snapshot_case(case_id, device="cuda")

ref_inputs = case.clone_pre_inputs()
cand_inputs = case.clone_pre_inputs()

ref_outputs = reference_impl.run(ref_inputs)
cand_outputs = candidate_impl.run(cand_inputs)

assert_tree_close(cand_outputs, ref_outputs, tolerance)

for path in case.mutable_arg_paths:
    assert_close(get_path(cand_inputs, path), get_path(ref_inputs, path), tolerance)
```

为什么不直接只比较 captured golden：

- 直接比较 snapshot golden 可以让 UT 脱离原始框架代码，适合作为 fallback。
- 但 reference replay 可以同时提供 baseline benchmark，并且能发现 snapshot capture 与当前 reference 实现不一致的问题。

Phase 1 推荐默认同时支持两种模式：

- `reference-replay`：优先模式，运行当前默认实现作为 golden。
- `snapshot-golden`：fallback 模式，只和 snapshot 捕获的 outputs/post state 比较。

## 6. Benchmark 设计

Benchmark 必须避免 mutable state 污染每轮计时。

标准流程：

1. 加载 selected snapshot。
2. 将 `pre_inputs` 移动到目标 GPU。
3. 每次 timed run 前，将 mutable inputs 恢复到 pre state。
4. reset 操作不计入 kernel 时间。
5. timed region 只包含 candidate/reference 调用和必要同步。

伪代码：

```python
runner = SnapshotBenchmarkRunner(case, target="candidate")
runner.prepare(device="cuda")

for _ in range(warmup):
    run_inputs = runner.reset_inputs_outside_timer()
    candidate_impl.run(run_inputs)

torch.cuda.synchronize()

for _ in range(repeat):
    run_inputs = runner.reset_inputs_outside_timer()
    torch.cuda.synchronize()
    start.record()
    candidate_impl.run(run_inputs)
    end.record()
    torch.cuda.synchronize()
    record_elapsed()
```

注意：

- 如果每轮 clone 大 tensor 成本很高，应预分配 working buffers，用 `copy_` 恢复 mutable state。
- 对无 mutation 的 case，可以复用输入。
- reference 和 candidate 必须使用同样的 reset 策略。
- benchmark 输出必须记录 warmup、repeat、device、dtype、case_id、mutable reset 策略。

## 7. 从原接口生成 Harness 的标准

`SnapshotHarnessBuilder` 生成的 task pack 需要遵循以下规则：

### reference_impl.py

优先封装原始默认实现：

```python
def reference_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc):
    return original_extend(q, k, v, g, beta, ssm_states=ssm_states, cache_indices=cache_indices, query_start_loc=query_start_loc)
```

如果原始实现不能在 task pack 中独立 import，则生成 snapshot-golden reference：

```python
def reference_extend_from_snapshot(case):
    return case.outputs, case.post_mutable_inputs
```

### candidate_impl.py

初始版本建议直接调用 reference，而不是 return dummy：

```python
def candidate_extend(...):
    return reference_impl.reference_extend(...)
```

这样 task pack 交付给 Kernel Engineer 时，`run_correctness.sh` 初始就是 pass。Kernel Engineer 替换 candidate 时，才开始引入优化风险。

### correctness_test.py

必须支持：

- `--case-id` 跑单个 case。
- 默认跑所有 `required` case。
- 比较 outputs。
- 比较 mutable inputs。
- 输出机器可读 JSONL 或至少稳定文本结果。

### benchmark.py

必须支持：

- `--target reference|candidate|both`
- `--case-id`
- `--warmup`
- `--repeat`
- CUDA event timing 优先；CPU timer 作为 fallback。
- 输出 JSONL。

## 8. 与 7 个工具的关系

Snapshot 设计直接服务于这些工具：

| 工具 | 使用 Snapshot 的方式 |
| --- | --- |
| `run_baseline` | 只记录 e2e baseline，不写 snapshot |
| `probe_target_calls` | 只记录目标接口是否被调用和调用次数 |
| `capture_snapshots` | 使用 `SnapshotRecorder` 写入 `snapshots/raw/` |
| `select_snapshots` | 使用 `SnapshotSelector` 生成 `snapshots/selected/` |
| `generate_harness_and_smoke` | 使用 `SnapshotHarnessBuilder` 生成 UT/benchmark，并跑 smoke test |
| `probe_env` | 不直接使用 snapshot，但结果写入同一个 task pack |
| `validate_task_pack` | 校验 snapshot manifest、selected case、UT/benchmark/env 是否一致 |

## 9. 实现难度判断

第一版最难的是 `capture_snapshots`，不是 `generate_harness`。

主要难点：

- Python 装饰器插桩要能安全回滚源码。
- 大 tensor 保存要控制体积。
- 非 contiguous tensor replay 不能静默变成 contiguous。
- mutable input 的 pre/post 捕获和比较容易漏。
- 不同接口的 args/kwargs tree 需要统一序列化。
- snapshot 采集不能明显改变用户 workload 行为。

建议第一版限制：

- 只支持 Python 层可见的 tensor/primitive/list/tuple/dict。
- 不支持任意复杂 Python 对象直接 snapshot；必须由 Framework Engineer 降维到 tensor/scalar ABI。
- 每个 semantic hash 默认最多保存 1 个完整 selected case，raw 只记录计数和少量 meta。
- 单个 snapshot 超过大小阈值时需要显式标记，并由用户或 Framework Engineer 决定是否保留。

这组限制不会破坏 Phase 1 的目标，反而能防止工具一开始承担过宽的范围。

# Skill: Snapshot Harness Construction

Snapshot harness 用来把真实框架接口调用变成可独立验证的 kernel 任务。Phase 1 不再默认从 shape 随机生成输入；UT 和 benchmark 必须 replay selected snapshots。

## 输入

- `task/snapshots/manifest.json`
- `task/snapshots/selected/<group_id>/group_meta.json`
- `task/snapshots/selected/<group_id>/samples/<sample_id>/meta.json`
- `task/snapshots/selected/<group_id>/samples/<sample_id>/pre_inputs.pt`
- `task/snapshots/selected/<group_id>/samples/<sample_id>/post_inputs.pt`
- `task/snapshots/selected/<group_id>/samples/<sample_id>/outputs.pt`
- 目标候选接口签名默认为：

```python
candidate(*args, **kwargs)
```

## Correctness 规则

1. 从 selected snapshot 加载 `pre_inputs`。
2. 为 reference 和 candidate 各 clone 一份输入。
3. 运行 reference 或使用 snapshot-golden fallback。
4. 运行 candidate。
5. 比较 outputs。
6. 根据 sample meta 中自动检测出的 mutation paths，比较运行后的 post-state。

Framework Engineer 不要求用户手工声明哪些输入会被原地修改。capture 会保存
`pre_inputs.pt` 和 `post_inputs.pt`，并递归 diff 二者，自动生成
`mutation.mutable_arg_paths`。tensor 通过 metadata + value hash 判断变化；
primitive 直接比较；list/tuple/dict 递归比较；结构变化或不可比较类型会记录
warning。correctness 只按 sample meta 中的自动检测结果比较 candidate post-state。

如果 reference 不能脱离 SGLang 独立 import，允许使用 snapshot-golden fallback。此时 reference 返回 captured outputs，并把 mutable inputs 更新到 captured post-state。

## Reference / Original 实现

`generate-harness` 必须生成两类 reference：

- `original_impl.py`：尝试通过原框架环境 linked import 并调用 capture 时的原始 target，用作 benchmark baseline。
- `reference_impl.snapshot_reference(...)`：只返回 captured outputs，用作 snapshot-golden correctness fallback。

`reference_impl.reference(...)` 默认调用 `original_impl.original(...)`。如果原始 target
是 instance method 且 task pack 无法重建 framework-owned `self`，或当前环境缺少 SGLang/vLLM
等框架依赖，benchmark 的 reference 分支会标记 unavailable。此时 task pack 仍然有效，
Kernel Engineer 可以阅读 `kernel_source_package/` 与 `kernel_translate/`，并使用
candidate-only benchmark。

## Benchmark 规则

- reference 和 candidate 使用同一批 selected snapshots。
- 每轮 timed run 前从 snapshot pre-state 重新 clone 输入；reset/clone/copy 不计入 timed region。
- reset/clone/copy 不计入 timed region。
- CUDA event timing 优先；CPU timer 只作为 fallback。
- 输出 JSONL，包含 group_id、sample_id、target、warmup、repeat、median_us、mean_us、min_us、max_us，并输出 group summary。
- `--target reference` 要求 linked original 可执行；`--target both` 会尽量运行 reference，
  reference 不可用时记录错误并继续运行 candidate；`--target candidate` 是正式支持的
  candidate-only 路径。

## Candidate 初始状态

`candidate_impl.py` 初始版本应该优先调用 `original_impl.original(...)`，让初始 benchmark
得到真实 baseline；如果原始 target 不可用，则 fallback 到 `snapshot_reference(...)`，用于
让初始 correctness smoke pass。Kernel Engineer 接手后只替换 candidate 实现，正式性能结果
必须来自真实 candidate kernel。

## 交付给 Kernel Agent 的内容

- `README.md`
- `validate_task_pack.py`
- `task/task.yaml`
- `task/shape_list.json`
- `task/snapshot_runtime.py`
- `task/snapshots/manifest.json`
- `task/snapshots/selected/<group_id>/samples/<sample_id>`
- `task/kernel_source_package/`（配置时）
- `task/kernel_translate/`
- `task/kernel_engineer_ws/`
- `task/original_impl.py`
- `task/reference_impl.py`
- `task/candidate_impl.py`
- `task/correctness_test.py`
- `task/benchmark.py`
- `task/scripts/run_correctness.py`
- `task/scripts/run_benchmark.py`
- `task/scripts/run_ncu.py`
- `task/env_manifest.yaml`

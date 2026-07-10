# Skill: Qwen3.5 Linear Attention Snapshot Task Pack

此 skill 指导 Framework Engineer 为 Qwen3.5 linear attention core kernel 生成 Phase 1 snapshot task pack。

## 适用范围

首期默认处理 GDN linear attention 的 core kernel：

- `TritonGDNKernel.extend`
- 可选：`TritonGDNKernel.decode`
- 可选：`TritonGDNKernel.packed_decode`

不要一开始把完整 `BailingMoELinearAttention.forward` 当成单个 kernel 任务。完整模块包含 projection、norm、rope、conv、gating、core recurrent/chunk kernel、state tracking、output projection，应该拆成多个 task。

## Snapshot 优先原则

以下输入必须来自真实 workload snapshot，不能由 Kernel Engineer 随机编：

- `query_start_loc`
- `cache_indices`
- `ssm_states`
- `g` / `beta`
- `q` / `k` / `v` 的数值、layout、stride、dtype
- prefill/decode mode
- state 是否需要被更新

`shape_list.json` 只是 selected snapshots 的摘要。UT 和 benchmark 必须从 `snapshots/selected/` replay 输入。

Phase 1 snapshot 按 forward window 内的 shape group 组织：`hit_count`
用于选择高频 group；每个 group 下保留多个真实 samples，供 Kernel Engineer
观察同 shape 下的真实输入分布。Framework Engineer 不需要判断哪些 tensor
value 会影响 kernel 控制流。

## 推荐接口边界

优先插在 core kernel wrapper 调用前：

```text
GDNAttnBackend.forward_extend
  -> self.kernel_dispatcher.extend(...)
```

如果用户给的是 module forward 级目标，先阅读代码拆解到一个或多个 tensor/scalar ABI 接口；如果用户已经给了接口列表，直接验证这些接口是否在 workload 中被调用。

## CLI 流程

推荐使用：

```bash
python -m framework_engineer.cli scaffold-task-pack ...
python -m framework_engineer.cli run-baseline ...
python -m framework_engineer.cli probe-target-calls ...
python -m framework_engineer.cli capture-snapshots ...
python -m framework_engineer.cli select-snapshots ...
python -m framework_engineer.cli generate-harness ...
python -m framework_engineer.cli probe-env ...
python -m framework_engineer.cli validate-task-pack ...
```

## Task Pack 必需文件

```text
task_pack/
  README.md
  task.yaml
  shape_list.json
  env_manifest.yaml
  snapshot_runtime.py
  original_source/
    manifest.json
    <copied_target_source>
  snapshots/
    manifest.json
    selected/
  original_impl.py
  reference_impl.py
  candidate_impl.py
  correctness_test.py
  benchmark.py
  scripts/run_correctness.sh
  scripts/run_benchmark.sh
  scripts/run_ncu.sh
```

## 验收标准

Framework Engineer 完成后，Kernel Engineer 应该能在不启动 SGLang 服务的情况下运行：

```bash
bash scripts/run_correctness.sh
bash scripts/run_benchmark.sh
```

`run_correctness.sh` 默认使用 snapshot-golden mode，必须能运行。`run_benchmark.sh`
默认会尝试 linked original reference；如果当前环境无法导入原框架或目标是无法重建
`self` 的 instance method，可以使用：

```bash
TARGET=candidate bash scripts/run_benchmark.sh
```

此时 `original_source/` 仍然作为 kernel engineer 阅读原始实现的参考材料。

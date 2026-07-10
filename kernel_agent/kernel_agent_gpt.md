# Kernel Agent 工程设计（GPT）

本文基于 `kernel_agent_new_design.md` 的问题、当前 `kernel_engineer/` active 工程、示例 `task_pack/`，以及 `backup/reference_repos/ref/` 下的参考仓库，给出一版可以落地的 Kernel Agent 工程设计。

核心结论：

- Kernel Agent 不应该把所有硬件、语言、样例、规则一次性塞进上下文。工程上应采用 **L0 系统 prompt、L1 skill、L2 知识库按需检索** 的分层加载。
- `task_pack/` 是唯一任务输入，`kernel_engineer/` 是可复用工作流。两者必须隔离，避免 agent 直接依赖框架源码或临时上下文。
- 优化流程必须是证据驱动的状态机：验收任务、建立 baseline、定目标、选择实现路径、实现、correctness、benchmark、profile、tuning、交付或请求框架变更。
- 开源实现建议优先作为 “上界探针 + 候选源” 接入，但不应跳过同一套 correctness/benchmark 验证，也不能绕过 `env_manifest.yaml` 的依赖约束。

---

## 1. 现状与输入契约

当前 active 工程在顶层：

```text
kernel_engineer/
  prompts/kernel_engineer.md
  skills/
  templates/
```

`kernel_agent/` 目录当前是归档和设计区，`README.md` 已说明 active entry point 是 `kernel_engineer/prompts/kernel_engineer.md` 和 `kernel_engineer/skills/task_pack_optimization_protocol.md`。

示例任务包位于：

```text
kernel_agent/backup/examples/qwen35_gdn_extend_task_pack/
```

这个 task pack 已经提供 Kernel Agent 所需的关键契约：

```text
task.yaml                       # ABI、目标、禁止修改项、允许实现路径
env_manifest.yaml               # Triton / CuTe DSL / CUDA extension / NCU 可用性
shape_list.json                 # selected snapshots 的摘要索引
snapshots/                      # replay 真值
snapshot_runtime.py             # snapshot replay runtime
original_impl.py                # baseline 实现
reference_impl.py               # correctness reference
candidate_impl.py               # Kernel Agent 允许修改的候选入口
correctness_test.py             # 不可改 correctness harness
benchmark.py                    # 不可改 benchmark harness
scripts/run_correctness.sh
scripts/run_benchmark.sh
scripts/run_ncu.sh
docs/                           # baseline、env probe、capture、selection 报告
```

因此 Kernel Agent 的设计重点不是重新定义任务格式，而是把 `kernel_engineer/` 补成一个可重复运行、可审计、上下文可控的工程。

---

## 2. 参考仓库调研结论

### 2.1 KernelAgent

本地参考路径：

```text
kernel_agent/backup/reference_repos/ref/KernelAgent/
```

主要特点：

- 生成与优化分成两条 pipeline：Fuser/Kernel generation 和 hardware-guided optimization。
- 优化 pipeline 是 `profile -> roofline -> bottleneck diagnosis -> LLM generate -> verify -> benchmark` 的闭环。
- prompt 用 Jinja 模板拼装，只把当前 kernel、GPU spec、roofline、bottleneck、recent attempts、RAG context 注入当前轮。
- 知识库不是全文注入，而是按 “瓶颈类型 -> 优化技术 -> 代码样例” 组织，再通过 RAG 取少量相关内容。
- 搜索策略有 greedy 和 beam search。beam search 维护 top-N kernel，并对多个瓶颈方向并行探索。
- 保留 runtime best 和 SOL best 两条记录，避免把不同 candidate 的性能和 profiler 指标混在一起。

对本工程的启发：

- prompt 必须模板化、阶段化，避免把知识库常驻进系统 prompt。
- profiling 结果要结构化，最好有 `parse_ncu.py` 这类工具把原始 NCU 输出压缩成瓶颈 JSON。
- iteration history、reflexion、candidate lineage 要落盘，后续轮次只读摘要。

### 2.2 AutoKernel

本地参考路径：

```text
kernel_agent/backup/reference_repos/ref/autokernel/
```

主要特点：

- 用一份很厚的 `program.md` 作为 agent 的 “研究组织代码”。
- 固定工具链：`profile.py -> extract.py -> bench.py -> verify.py`。
- agent 一次主要改一个 `kernel.py`，跑固定 benchmark，提升则 keep，退化则 revert。
- benchmark 自带多阶段 correctness、性能、roofline、结果 TSV。
- orchestrator 用 Amdahl law 决定优化哪个 kernel 更值得。

对本工程的启发：

- 固定评测脚本比自由发挥更重要。Kernel Agent 必须只信 `task_pack` 内的 correctness/benchmark。
- 每轮结果必须落盘成表，停机时选择 best-so-far，而不是默认交付最后一次改动。
- 如果有端到端占比，目标应按 Amdahl law 加权，而不是只看 micro benchmark speedup。

### 2.3 kernel-design-agents

本地参考路径：

```text
kernel_agent/backup/reference_repos/ref/kernel-design-agents/docs/agent-flow.md
```

主要特点：

- 可复用工作流和任务工作区分离。
- 每个任务必须有 task contract：目标、输入输出、正确性要求、约束、验证命令、评价命令、promotion criteria。
- 候选是否晋级必须有 evidence records。

对本工程的启发：

- `kernel_engineer/` 只放工作流、prompt、skill、知识与工具。
- `task_pack/` 只放任务数据、不可改 harness、允许修改的 candidate、运行证据。
- promotion rule 必须显式写进 skill，而不是靠 agent 自觉。

### 2.4 开源参考实现的问题定义和比较对象

几个参考实现的共同点是：**即使优化对象是一个已有 kernel，correctness oracle 仍然尽量回到 PyTorch/reference 语义；性能 baseline 则分成 PyTorch baseline、已有 kernel baseline、best-so-far 三类**。不能把 “正确性对谁比” 和 “性能对谁比” 混为一谈。

| 参考实现 / 模式 | 问题如何描述 | 原始参考实现 | 候选实现 | 准确度比较对象 | 性能比较对象 |
|---|---|---|---|---|---|
| KernelAgent generation / Fuser | KernelBench problem 或 PyTorch `Model.forward()`；也支持自然语言 problem description | PyTorch `Model` / 原始 PyTorch forward | 生成的 Triton `kernel_function`，或 composed Triton program | PyTorch reference output，测试中 `torch.allclose`，示例 tolerance 常为 `1e-2` | 主要看生成 kernel 是否通过；优化阶段才系统比较 PyTorch baseline/current kernel |
| KernelAgent hardware-guided optimization | 一个已验证 Triton kernel + `problem.py` + `test.py` + GPU spec | `problem.py` 里的 PyTorch `Model`，测试文件负责生成 reference output | 每轮 LLM 生成的新 Triton kernel | `test.py` 中 PyTorch model 输出 vs `kernel_function` 输出 | 当前 best Triton kernel、PyTorch eager baseline、NCU SOL/roofline |
| AutoKernel 通用模型模式 | 任意 PyTorch model；先用 profiler 找瓶颈，再抽取成标准 kernel 类型 | `reference.py` 中 PyTorch-only oracle，或原始 PyTorch model | `kernel.py` 中单个 `kernel_fn`，Triton 或 CUDA C++ | `bench.py` 的 5 段 correctness：smoke、shape sweep、numerical stability、determinism、edge cases，均对 PyTorch reference | `speedup_vs_pytorch`、latency、TFLOPS/GB/s、pct_peak；最后 `verify.py` 看端到端 speedup |
| AutoKernel KernelBench 模式 | KernelBench 的 `reference.py`，包含 `Model`、`get_inputs()`、`get_init_inputs()` | KernelBench 原始 `Model` | `kernel.py` 中 `ModelNew` | `ModelNew(*inputs)` vs `Model(*inputs)`，默认多 trial，`atol=rtol=1e-2` | CUDA event 计时：`reference_time_ms / kernel_time_ms`；批量评分用 `fast_p` |
| kernel-design-agents | Task contract：目标、输入输出、正确性要求、验证命令、评价命令、promotion criteria | 由任务工作区指定，不强制是 PyTorch | 任意候选实现 | 由 validation command 决定 | 由 evaluation command / promotion criteria 决定 |
| 本工程 task pack | `task.yaml` + selected snapshots + ABI；Framework Engineer 已捕获真实调用 | `reference_impl.py`、snapshot golden、`original_impl.py` | `candidate_impl.py` / `kernel_sources/` | `correctness_test.py`：snapshot-golden 为主，可 reference-replay；还会比对 mutable inputs | `benchmark.py`：reference/original median vs candidate median；最终看 required hot case speedup 和 stability |

更具体地说：

- **用 PyTorch 作为 reference**：KernelBench、KernelAgent generation、AutoKernel 标准 kernel 都主要这么做。优点是语义清楚、容易生成随机输入；缺点是可能和真实框架调用、真实 layout、inplace/mutable 语义不完全一致。
- **用已有算子实现作为 baseline**：KernelAgent optimization 和本工程 task pack 更接近这个模式。已有实现可以是 Triton/CUDA/框架 wrapper，用于性能对比；但 correctness 最好仍然对 PyTorch reference 或 snapshot golden。
- **用 snapshot golden 作为 reference**：这是本工程比通用 benchmark 更适合业务 kernel 的地方。它能覆盖真实输入树、stride、cache/state、mutable input；代价是问题更窄，泛化能力依赖 snapshot selection。
- **性能比较对象会随阶段变化**：接包时比原始实现，迭代时比 best-so-far，技术探针时比开源实现，最终交付时比 task target。不要用某一轮的 PyTorch eager speedup 替代最终 task-pack speedup。

对当前设计的直接约束：

1. `correctness_test.py` 是唯一 correctness gate，不能因为某个开源 kernel 自带 test 通过就认为可交付。
2. `original_impl.py` / `reference_impl.py` / snapshot golden 的角色要分清：前者多用于性能 baseline，后两者用于语义。
3. 开源实现接入后要变成 task pack 下的一个 candidate，用同一套 harness 重新评测。
4. 报告中必须同时写清楚 “correctness compared against X” 和 “performance compared against Y”。

---

## 3. 总体架构

### 3.1 工程边界

Kernel Agent 分为两层：

```text
kernel_engineer/       # 可复用 agent 工程
task_pack/             # 单个任务实例，由 Framework Engineer 生成
```

`kernel_engineer/` 可以演进、扩展、沉淀知识；`task_pack/` 是当前任务的事实来源。Kernel Agent 只通过 task pack 工作，不直接假设 SGLang 或其他框架内部状态。

### 3.2 推荐目录结构

当前已有 `prompts/`、`skills/`、`templates/`。建议补齐为：

```text
kernel_engineer/
  agent.yaml
  prompts/
    kernel_engineer.md
  skills/
    task_triage.md
    task_pack_optimization_protocol.md
    kernel_optimization_loop.md
    triton_cuda_codegen.md
    nvidia_ncu_analysis.md
    framework_feedback.md
    perf_target_and_roofline.md
    open_source_probe.md
    tuning_and_search.md
  knowledge/
    INDEX.md
    hardware/
      nvidia_h20.md
      nvidia_h100.md
      _template.md
    languages/
      triton.md
      cutedsl.md
      cuda_extension.md
      cutlass.md
    playbooks/
      memory_bound.md
      compute_bound.md
      underutilized.md
      launch_bound.md
      low_occupancy.md
      numerical_stability.md
    samples/
      triton/
      cutedsl/
      cuda/
  tools/
    summarize_benchmark.py
    parse_ncu.py
    estimate_roofline.py
    extract_hot_cases.py
    pick_best_iteration.py
    validate_iteration_log.py
  schemas/
    task_pack.schema.json
    iteration_record.schema.json
    benchmark_summary.schema.json
    ncu_summary.schema.json
  templates/
    task_acceptance_review.md
    iteration_log.md
    benchmark_report.md
    kernel_constraints.md
    kernel_delivery_package.md
    framework_change_request.yaml
```

各目录职责：

| 目录 | 职责 | 常驻上下文 |
|---|---|---|
| `prompts/` | 定义 agent 身份、边界、总流程、加载规则 | 是 |
| `skills/` | 阶段性操作手册 | 否，进入阶段时按名读取 |
| `knowledge/` | 硬件、语言、优化 playbook、代码样例 | 否，只通过 `INDEX.md` 或检索命中读取 |
| `tools/` | 把 benchmark/NCU/日志转换为短 JSON 或表格 | 否，由 skill 调用 |
| `schemas/` | 约束中间产物格式，减少日志不可解析问题 | 否 |
| `templates/` | 固定交付物格式 | 否，交付阶段读取 |

### 3.3 `agent.yaml`

建议增加一个轻量配置文件，让不同工程环境不靠 prompt 硬编码：

```yaml
name: kernel_engineer
version: 0.1
default_task_root: task_pack
target_hardware:
  default: nvidia_h20
context_policy:
  l0_prompt_max_lines: 160
  max_knowledge_files_per_round: 3
  max_tool_output_chars: 12000
evaluation:
  min_speedup_if_unspecified: 1.10
  final_rerun_tolerance_pct: 5.0
  keep_threshold_pct: 1.0
  plateau_rounds: 3
  plateau_min_gain_pct: 3.0
allowed_output_paths:
  - candidate_impl.py
  - kernel_sources/
  - docs/
```

这个文件不替代 `task.yaml`。它只定义 agent 默认策略；一旦 `task.yaml` 给出更具体约束，以 `task.yaml` 为准。

---

## 4. 上下文治理设计

### 4.1 为什么不能把知识全塞进去

GPU kernel 优化的知识包含硬件 spec、Triton/CuTe/CUDA 文档、NCU metric 解释、开源 kernel 样例、历史失败经验。全部输入模型会导致三个问题：

- 关键 task contract 被淹没，agent 更容易忽略 ABI 和 forbidden changes。
- 上下文成本高，且不同阶段需要的知识完全不同。
- 旧知识可能污染当前轮判断，例如把 H100/Hopper 的某些策略误套到 H20 的具体 shape。

参考仓库的共同做法不是全文注入，而是按阶段和证据取用：

- KernelAgent 用 prompt template + RAG context。
- AutoKernel 让工具输出落盘，agent 读摘要或 grep。
- kernel-design-agents 强调任务 workspace 自带 evidence。

### 4.2 三层加载模型

建议固定为：

| 层级 | 内容 | 进入上下文时机 | 体量约束 |
|---|---|---|---|
| L0 | 系统 prompt、职责边界、总状态机、加载规则 | 常驻 | 少于 160 行 |
| L1 | 当前阶段 skill | 进入阶段时读取 | 单文件单职责 |
| L2 | 知识库和代码样例 | profiler/语言/瓶颈命中后读取 | 每轮最多 3 个文件 |

`knowledge/INDEX.md` 是 L2 的入口。示例：

```text
目标硬件是 H20
  -> hardware/nvidia_h20.md

选择 Triton
  -> languages/triton.md

NCU: DRAM SOL 高，SM SOL 低
  -> playbooks/memory_bound.md

NCU: compute SOL 和 memory SOL 都低
  -> playbooks/underutilized.md

小 shape 延迟主要是 launch overhead
  -> playbooks/launch_bound.md

正确性失败集中在 bf16/fp32 累积误差
  -> playbooks/numerical_stability.md
```

### 4.3 工具输出策略

所有重日志必须先落盘，再由工具压缩：

```text
task_pack/docs/raw/
  benchmark_iter_003.jsonl
  ncu_iter_003_case_group0_sample1.txt

task_pack/docs/summaries/
  benchmark_iter_003.summary.json
  ncu_iter_003_case_group0_sample1.summary.json
```

agent 后续只读取 summary，必要时再打开 raw 的局部片段。

这样可以避免 NCU 原始输出、benchmark 全量日志、编译报错把当前上下文撑爆。

---

## 5. Skill 设计

### 5.1 现有 skill 保留并补强

当前已有：

```text
task_triage.md
task_pack_optimization_protocol.md
kernel_optimization_loop.md
triton_cuda_codegen.md
nvidia_ncu_analysis.md
framework_feedback.md
```

建议保持这些文件的边界，但做两类补强：

| skill | 补强点 |
|---|---|
| `task_pack_optimization_protocol.md` | 明确 task pack schema、允许修改路径、每轮证据记录格式 |
| `kernel_optimization_loop.md` | 加入 keep/reject、best-so-far、plateau、final rerun gate |
| `triton_cuda_codegen.md` | 加入语言选择矩阵和开源探针 |
| `nvidia_ncu_analysis.md` | 从指南升级为工具驱动流程，配合 `parse_ncu.py` |
| `framework_feedback.md` | 明确何时必须停下来输出 `FrameworkChangeRequest` |

### 5.2 新增 `perf_target_and_roofline.md`

职责：

- 从 `task.yaml.success_criteria` 读取目标。
- 如果用户没有给目标，派生默认目标。
- 读取 baseline，计算每个 hot case 的目标 latency。
- 用 roofline 或 NCU SOL 判断目标是否物理合理。

默认规则：

- correctness 永远是门控。
- 如果 task 未指定 performance target，默认 required hot case 至少 `1.10x` speedup。
- 如果存在端到端占比，用 Amdahl law 把 micro speedup 转成端到端收益，优先优化收益最大的 case。
- 如果 baseline 已接近 `90%` 以上有效 roofline，目标改为 “保持 correctness + 提供物理上限解释 + 尝试低风险小优化”。

### 5.3 新增 `open_source_probe.md`

职责：

- 判断是否值得接入开源实现。
- 检查 license、依赖、ABI、硬件适配。
- 把开源实现用同一套 `correctness_test.py` 和 `benchmark.py` 评估。
- 把结果作为候选、上界或不可用证据。

输出记录：

```yaml
source_name: fla_or_other
source_path_or_url: ...
license: ...
dependency_status: allowed_or_blocked
abi_mapping: direct_or_adapter_needed
correctness: pass_or_fail
median_speedup_by_case: ...
decision: adopt | use_as_upper_bound | reject
reason: ...
```

### 5.4 新增 `tuning_and_search.md`

职责：

- 定义结构调优与参数搜索的顺序。
- 管理 candidate lineage。
- 明确 keep/reject 和停止条件。

推荐规则：

- 每轮只改一个主要因素。
- 提升超过 `1%` 且 final rerun 不劣化，才可以 keep。
- 同分取更简单实现，避免为噪声引入复杂分支。
- 连续 3 轮有效提升小于 3%，必须重新 profile 并换方向。
- 结构未稳定前不做大规模 autotune；结构稳定后再扫 block/tile/warps/stages。

---

## 6. 知识库设计

### 6.1 `hardware/`

每个硬件文件只放会影响决策的内容：

```text
hardware/nvidia_h20.md
  - compute capability
  - SM 数
  - HBM 容量和带宽
  - L2 / shared memory / registers 关键限制
  - tensor core 支持 dtype
  - NCU 常用 metric 映射
  - 对 Triton/CuTe/CUDA 的注意事项
```

不要把整份官方文档复制进来。只保留 agent 需要用来判断 bottleneck、tile、occupancy、roofline 的事实。

### 6.2 `languages/`

每个语言文件回答三件事：

- 什么场景优先用这个语言。
- 常见正确性坑和性能坑。
- 最小可用代码范式。

语言选择建议：

| 场景 | 优先路径 |
|---|---|
| elementwise、简单 reduction、layout transform、快速试错 | Triton |
| 需要更细 tile、MMA、shared memory pipeline、向量化控制 | CuTe DSL |
| 需要内联 PTX、复杂同步、非 JIT 集成、Triton/CuTe 已证明达不到目标 | CUDA extension |
| GEMM-like 且 ABI 能稳定映射到库范式 | CUTLASS |

### 6.3 `playbooks/`

playbook 按瓶颈组织，而不是按语言组织。一个 `memory_bound.md` 可以同时指向 Triton/CuTe/CUDA 样例。

推荐模板：

```text
# Memory Bound Playbook

## 判定信号
...

## 常见根因
...

## 优先尝试
1. 减少 global memory round trip
2. 改善 coalescing
3. 提高 L2 reuse
4. 融合 epilogue

## 不建议优先尝试
...

## 相关样例
- samples/triton/...
- samples/cutedsl/...
```

这样 agent 先根据 profiler 选瓶颈，再根据实现路径读语言细节。

---

## 7. Tools 设计

### 7.1 `summarize_benchmark.py`

输入：`benchmark.py` 的 JSONL 输出。

输出：

```json
{
  "record_type": "benchmark_summary",
  "iteration": 3,
  "all_required_passed": true,
  "cases": [
    {
      "group_id": "g0",
      "sample_id": "s0",
      "reference_median_us": 120.0,
      "candidate_median_us": 90.0,
      "speedup_median": 1.333,
      "meets_target": true
    }
  ],
  "worst_required_speedup": 1.12,
  "geomean_required_speedup": 1.24
}
```

注意：最终判定必须看每个 required hot case，不能只看平均值。

### 7.2 `parse_ncu.py`

输入：NCU raw 输出或 csv。

输出：

```json
{
  "record_type": "ncu_summary",
  "target_kernel": "candidate_kernel",
  "compute_sol_pct": 42.1,
  "memory_sol_pct": 78.4,
  "achieved_occupancy_pct": 51.0,
  "dram_throughput_pct": 76.0,
  "l2_hit_rate_pct": 62.5,
  "top_stalls": [
    {"name": "long_scoreboard", "pct": 35.2}
  ],
  "bottleneck": "memory_bound",
  "confidence": "medium",
  "next_playbook": "knowledge/playbooks/memory_bound.md"
}
```

关键要求：

- 过滤 PyTorch 或 runtime 内部 kernel，只保留 candidate target。
- 输出 metric 的原始 key，便于追溯。
- 指明数据质量问题，例如 metric 缺失、多个 kernel 混杂。

### 7.3 `estimate_roofline.py`

输入：shape、dtype、读写字节、FLOPs 估计、硬件 spec。

输出：

```json
{
  "arithmetic_intensity": 1.8,
  "predicted_bound": "memory",
  "best_possible_us": 42.0,
  "baseline_pct_of_estimated_peak": 67.0,
  "target_1p10_feasible": true,
  "notes": ["estimate ignores cache reuse"]
}
```

它用于没有 NCU 或 NCU 成本太高时做快速目标 sanity check。

### 7.4 `pick_best_iteration.py`

输入：`docs/iteration_log.md` 或结构化 `iterations.jsonl`。

输出：

- best iteration ID。
- 对应 candidate 路径。
- 是否满足 final gate。
- 如果最新不是最优，给出回滚建议。

停机时必须运行或等价执行这个逻辑，避免把最后一次失败尝试交付出去。

---

## 8. Agent 工作流

### 8.1 状态机

```text
S0 Load Task
  -> S1 Triage
  -> S2 Baseline And Target
  -> S3 Implementation Path Selection
  -> S4 Candidate Implementation
  -> S5 Correctness Gate
  -> S6 Benchmark Gate
  -> S7 Profile And Diagnose
  -> S8 Tune Or Search
  -> S9 Final Rerun
  -> S10 Delivery Or FrameworkChangeRequest
```

每个状态都必须有输入、动作、输出和失败转移。

### 8.2 S0 Load Task

读取：

- `README.md`
- `task.yaml`
- `env_manifest.yaml`
- `shape_list.json`
- `snapshots/manifest.json`
- `candidate_impl.py`
- `reference_impl.py`
- `correctness_test.py`
- `benchmark.py`
- `docs/*baseline*`

输出：

- 任务摘要。
- ABI 摘要。
- forbidden changes。
- required hot cases。

### 8.3 S1 Triage

验收条件：

- `task.yaml` 字段足够定义 ABI 和 success criteria。
- `correctness_test.py` 能独立验证 candidate。
- `benchmark.py` 能比较 reference/original 和 candidate。
- `env_manifest.yaml` 明确可用工具链。
- 至少有一个 required hot case。
- 禁止修改项明确。

失败转移：

- 输出 `task_acceptance_review.md`。
- 不修 snapshot、benchmark、tolerance、timing rules。

### 8.4 S2 Baseline And Target

动作：

1. 跑 `bash scripts/run_correctness.sh`。
2. 跑 `bash scripts/run_benchmark.sh`。
3. 用 `summarize_benchmark.py` 汇总。
4. 从 `task.yaml.success_criteria` 读取目标。
5. 若目标缺失，派生默认目标。
6. 若可能，用 `estimate_roofline.py` 或 NCU 给出目标 feasibility。

目标派生规则：

| 情况 | 目标 |
|---|---|
| task.yaml 写明 `hot required cases >= 1.10x` | 直接采用 |
| 未写 micro target | 默认 required hot case `>= 1.10x` |
| 有端到端占比 | 用 Amdahl law 排优先级，micro target 仍需逐 case 记录 |
| baseline 已近物理上限 | 允许以 “上限解释 + 小幅提升” 作为成功条件 |

### 8.5 S3 Implementation Path Selection

选择依据：

- `task.yaml.allowed_implementation_paths`
- `env_manifest.yaml`
- ABI 是否适合某语言
- hot shape 是否稳定
- correctness 风险
- 开源实现是否存在并可 vendor

决策矩阵：

| 条件 | 决策 |
|---|---|
| Triton 可用且 op 适合 block/tile 或 fusion | 先写 Triton prototype |
| CuTe DSL 可用且需要更细 tile/MMA/pipeline 控制 | CuTe DSL prototype |
| Triton/CuTe 都可用但不确定 | 先 Triton 拿 correctness，再用 profiling 决定是否转 CuTe |
| JIT 已证明达不到目标，nvcc/headers 可用 | 升级 CUDA extension |
| GEMM-like、layout 稳定、库可用 | 评估 CUTLASS |
| 开源实现可映射 ABI | 先做 open source probe |

开源实现处理：

- 正确、快、依赖合规，则可以采用。
- 正确但集成成本高，则作为性能上界。
- 不正确或 ABI 不匹配，则记录 reject reason。

具体技术选型的优劣：

| 技术路径 | 适合场景 | 优点 | 主要风险 | 应比较的对象 |
|---|---|---|---|---|
| PyTorch reference / eager | 定义语义、快速建立 oracle、生成随机输入 | 最清楚、最稳定、最容易维护 | 不代表真实业务 layout/state；性能 baseline 可能太弱 | correctness 对 PyTorch；性能只作为最低 baseline |
| 已有框架算子 / original kernel | 已有 Triton/CUDA/C++ 实现，需要局部提速 | 最贴近当前生产语义和 ABI；性能 baseline 有意义 | 代码可能复杂，难独立复现；隐藏框架侧假设 | correctness 仍对 reference/snapshot；性能对 original |
| Triton | elementwise、reduction、layout transform、轻中等复杂 fused op | 迭代快、Python 内联方便、autotune 成本低 | 对复杂 pipeline、极致寄存器/共享内存控制有限；JIT warmup 要处理 | 对 original/best-so-far；profile 看 SOL/occupancy |
| CuTe DSL | 需要精细 tile、MMA、shared memory pipeline、向量化控制 | 比 Triton 更接近底层 tile 表达，适合追极限 | 学习和调试成本高；环境可用不等于 task 可编译运行 | 先对 Triton/best-so-far，再对 original target |
| CUDA extension | 需要 warp primitive、inline PTX、复杂同步、非 JIT 集成 | 控制力最强，可做 Triton 难表达的细节 | 编译/调试慢，ABI 和部署风险高，容易过拟合 shape | 必须证明 JIT 路径受限后再对 original/best 比 |
| CUTLASS | GEMM-like、conv-like、固定 layout/tile、可映射库接口 | 复用成熟高性能模板，tensor core 路径强 | ABI 映射成本高；非标准 epilogue/state 更新不一定适合 | 对 cuBLAS/CUTLASS baseline、original、best-so-far |
| 开源专用 kernel | FLA、xFormers、FlashAttention、社区 Triton/CUDA 实现等已有近似算子 | 能快速拿到强 candidate 或性能上界 | license、依赖、shape/layout/语义不匹配；可能不支持 mutable state | 先跑 task correctness，再作为 candidate 或 upper bound |
| PyTorch internal / torch.compile | 小改即可调用更优内部路径，或作为 fallback | 工程风险低，适合先建立可用 candidate | 可能不满足 “kernel 优化” 目标；速度上限不稳定 | 可作为 fallback，对 original 和 custom kernel 比 |

选型顺序建议是：**先用 PyTorch/reference 锁语义，再用 original/已有算子锁业务 baseline；实现上先 Triton/CuTe 这种 JIT 快速试错，只有在 profiler 证明受限后升级 CUDA/CUTLASS；开源实现优先作为 probe，但必须经过 task-pack harness 重评测**。

### 8.6 S4 Candidate Implementation

原则：

- 先写最小正确实现，再做性能。
- 保持 `candidate(*args, **kwargs)` ABI 不变。
- 如需额外代码，放入 `kernel_sources/`。
- 不修改 snapshots、runtime、reference、benchmark、tolerance。
- 对 dtype、shape、layout、contiguous 假设写入 `kernel_constraints.md` 草稿。

### 8.7 S5 Correctness Gate

动作：

- 跑 `bash scripts/run_correctness.sh`。
- 必要时分别跑 required hot case。
- 正确性失败时，不进入 benchmark 优化。

正确性失败分类：

| 类型 | 处理 |
|---|---|
| ABI 调用失败 | 修 wrapper 和参数解析 |
| dtype/shape 不匹配 | 修输出结构和 dtype |
| 数值误差 | 检查累积精度、mask、边界、order |
| mutable input 不一致 | 对照 snapshot `post_inputs` 修 inplace 语义 |
| 只有某些 shape 失败 | 记录 shape，先做有限 specialization 或 fallback |

### 8.8 S6 Benchmark Gate

动作：

- correctness 通过后跑 benchmark。
- 每轮把原始输出落盘。
- 用 `summarize_benchmark.py` 生成摘要。

判定：

- required hot case 不得出现明显回退。
- 最终交付必须跑完整 `target=both` 或等价 reference/candidate 对比。
- 日常快速迭代可以用 candidate-only smoke，但不能作为最终结论。

### 8.9 S7 Profile And Diagnose

动作：

- 对最慢或未达标 hot case 跑 `scripts/run_ncu.sh`。
- 用 `parse_ncu.py` 转换为短 JSON。
- 根据 `next_playbook` 读取知识库。

诊断映射：

| 信号 | 初步瓶颈 |
|---|---|
| memory SOL 高、compute SOL 低 | memory bound |
| compute SOL 高、memory SOL 低 | compute bound |
| 二者都低 | underutilized |
| occupancy 低 | register/smem/block 配置问题 |
| long scoreboard 高 | memory latency 或依赖链 |
| barrier 高 | 同步或 tile 组织问题 |
| 小 shape latency 不随 workload 缩放 | launch bound |

### 8.10 S8 Tune Or Search

搜索顺序：

1. 修 correctness 和边界。
2. 做结构优化：减少读写、改变 tile、融合、消除临时 tensor。
3. 做 config tuning：block size、num warps、num stages、vector width。
4. 做有限 shape specialization。
5. 若 JIT 路径受限，升级语言或输出 FrameworkChangeRequest。

每轮记录：

```json
{
  "iteration": 4,
  "parent": 2,
  "change": "increase BLOCK_M from 64 to 128",
  "hypothesis": "improve memory coalescing and reduce launch count",
  "correctness": "PASS",
  "benchmark_summary": "...",
  "profile_summary": "...",
  "decision": "KEEP",
  "reason": "worst required speedup improved from 1.08x to 1.14x"
}
```

### 8.11 S9 Final Rerun

最终交付前必须：

- 回到 best-so-far candidate。
- 重新跑 correctness。
- 重新跑完整 benchmark。
- 检查 final rerun 是否在 best latency 的 5% 内。
- 如果 benchmark 波动大，增加重复次数或报告不稳定。

### 8.12 S10 Delivery Or FrameworkChangeRequest

成功交付：

- `candidate_impl.py`
- `kernel_sources/`
- `docs/benchmark_report.md`
- `docs/kernel_constraints.md`
- `docs/kernel_delivery_package.md`
- `docs/iteration_log.md`

需要框架配合：

- `docs/framework_change_request.yaml`
- 附带 benchmark/profile 证据。
- 明确如果框架不改，kernel 侧能做到的 best-so-far。

---

## 9. 评价体系

评价分四层，必须按顺序执行。

| 层级 | 指标 | 作用 |
|---|---|---|
| Correctness | snapshot-golden、reference-replay、mutable input 对比 | 门控 |
| Performance | per-case median latency、speedup、worst required speedup | 主评价 |
| Efficiency | NCU SOL、roofline、occupancy、stall | 判断空间和方向 |
| Robustness | final rerun、波动、shape 覆盖、fallback | 交付风险 |

### 9.1 Correctness

必须使用 task pack 的 `correctness_test.py`。不得修改：

- `snapshots/`
- `snapshot_runtime.py`
- `reference_impl.py`
- `correctness_test.py`
- tolerance
- selected sample

mutable inputs 必须按 snapshot 语义验证。示例 harness 已通过 `mutable_arg_paths` 比对 candidate 执行后的输入和 expected post inputs。

### 9.2 Performance

性能指标以 per-case median 为主：

```text
speedup_median = reference_median_us / candidate_median_us
```

最终报告必须至少包含：

- 每个 required case 的 reference median。
- 每个 required case 的 candidate median。
- 每个 required case 的 speedup。
- worst required speedup。
- geomean required speedup。
- 是否满足 task target。

不能只报平均值，因为平均值可能掩盖单个 hot case 的退化。

### 9.3 Efficiency

Efficiency 不直接替代 performance，但用于判断方向和停止：

- memory SOL 接近上限，优先减少字节或改善 reuse。
- compute SOL 接近上限，优先提升 math pipe 或 tensor core 使用。
- 二者都低，说明不是简单带宽或算力上限，优先看 occupancy、stall、launch。
- 已接近 roofline 时，不能要求 agent 无限追求 10% 提升。

### 9.4 Robustness

交付前需要记录：

- 支持的 dtype。
- 支持的 shape 范围。
- 支持的 layout/stride。
- 不支持条件和 fallback。
- 编译缓存或 JIT warmup 影响。
- benchmark 稳定性。

---

## 10. 停止标准

命中任一条件即可停止，但必须在 `iteration_log` 和交付物中写清楚证据。

| 停止类型 | 条件 | 输出 |
|---|---|---|
| 达标停止 | correctness 通过，所有 required hot case 达到目标，final rerun 稳定 | KernelDeliveryPackage |
| 物理上限停止 | NCU/roofline 证明已接近上限，继续优化收益很低 | 瓶颈解释 + best candidate |
| 平台期停止 | 连续 3 个有效迭代提升小于 3%，且尝试过至少 3 类方向 | best candidate + remaining risks |
| 阻塞停止 | 需要 layout、metadata、workspace、padding、融合边界等框架配合 | FrameworkChangeRequest |
| 预算停止 | 达到时间、轮次、计算预算 | best-so-far + 下一步建议 |
| 正确性阻塞 | 明确无法在现有 ABI 下满足语义 | TaskAcceptanceReview 或 change request |

重要规则：

- 停机时交付 best-so-far，不交付 latest。
- 未达标也要交付可审计证据，不允许静默丢弃失败尝试。
- 如果最终依赖 fallback，必须说明 fallback 命中条件和性能影响。

---

## 11. 产物格式

建议把 `docs/iteration_log.md` 改为同时支持人读和机器解析。可以保留 Markdown，但每轮附一个 JSON block。

示例：

````markdown
## Iteration 003

Hypothesis: BLOCK_M=128 may improve memory coalescing for hot shape group_0.

```json
{
  "iteration": 3,
  "parent": 2,
  "candidate_path": "candidate_impl.py",
  "change_type": "triton_config",
  "correctness": "PASS",
  "worst_required_speedup": 1.14,
  "decision": "KEEP"
}
```
````

交付物关系：

```text
benchmark_report.md
  证明性能

kernel_constraints.md
  说明支持范围和 fallback

kernel_delivery_package.md
  汇总实现、使用方式、结果、风险

framework_change_request.yaml
  只有需要框架配合时生成
```

---

## 12. 与当前 `kernel_engineer/` 的差距

当前已有：

- lean system prompt。
- task pack protocol。
- triage skill。
- optimization loop skill。
- Triton/CuTe/CUDA codegen strategy。
- NCU analysis 指南。
- framework feedback 指南。
- 交付模板。

建议新增或补强：

| 优先级 | 工作项 | 原因 |
|---|---|---|
| P0 | 在 prompt 中加入三层加载模型和工具输出落盘策略 | 解决输入太多的问题 |
| P0 | 增加 `perf_target_and_roofline.md` | 解决用户未给目标时如何定目标 |
| P0 | 扩展 `kernel_optimization_loop.md` 的 keep/reject/final gate | 保证不会交付 latest 而非 best |
| P1 | 增加 `knowledge/INDEX.md` 和 H20/Triton/NCU 最小知识库 | 让 agent 有按需知识入口 |
| P1 | 增加 `summarize_benchmark.py`、`parse_ncu.py` | 压缩日志，提升可审计性 |
| P1 | 增加 `open_source_probe.md` | 明确开源实现的接入规则 |
| P2 | 增加 schemas 和结构化 iterations.jsonl | 方便自动选择 best 和回放 |
| P2 | 增加简单 orchestrator CLI | 把手工流程产品化 |

---

## 13. 推荐落地路线

### Phase 0: 文档和协议收敛

目标：不写复杂框架，只让人工/agent 按固定协议工作。

动作：

- 更新 `prompts/kernel_engineer.md`。
- 补 `perf_target_and_roofline.md`、`open_source_probe.md`、`tuning_and_search.md`。
- 更新 `kernel_optimization_loop.md` 的停止和交付规则。
- 建 `knowledge/INDEX.md`，只放最小路由。

### Phase 1: 工具化

目标：把大日志变成小证据。

动作：

- 实现 `summarize_benchmark.py`。
- 实现 `parse_ncu.py`。
- 实现 `estimate_roofline.py`。
- 实现 `pick_best_iteration.py`。

### Phase 2: 半自动编排

目标：把高频固定步骤从 prompt 迁移到 CLI。

动作：

- `kernel_engineer run-triage <task_pack>`。
- `kernel_engineer run-benchmark <task_pack> --iter N`。
- `kernel_engineer run-profile <task_pack> --case ...`。
- `kernel_engineer pick-best <task_pack>`。

### Phase 3: 搜索增强

目标：引入 beam/greedy 搜索和轻量 RAG，但不破坏 task pack 约束。

动作：

- `iterations.jsonl` 做 program database。
- top-K candidate 管理。
- 根据 profiler 瓶颈选择 playbook。
- 可选 embedding 检索 `knowledge/`。

---

## 14. 一页回答

### 目录结构怎么设计

使用 `kernel_engineer/` 作为 agent 工程根目录，分为 `prompts/`、`skills/`、`knowledge/`、`tools/`、`schemas/`、`templates/`。系统 prompt 只放角色和总流程；skills 放阶段工作法；knowledge 放硬件、语言、playbook 和样例；tools 把 benchmark/NCU 原始输出压缩成摘要。

### 输入太多怎么办

采用三层加载：L0 prompt 常驻，L1 skill 阶段加载，L2 knowledge 按 `INDEX.md` 或检索命中加载。工具输出必须落盘再摘要，agent 默认只读 summary。

### 工作流程是什么

状态机为：

```text
Load Task -> Triage -> Baseline/Target -> Path Selection -> Implement
-> Correctness -> Benchmark -> Profile -> Tune/Search -> Final Rerun -> Delivery
```

失败时输出 `task_acceptance_review.md` 或 `framework_change_request.yaml`，不修改不可改 harness。

### 没给优化目标怎么办

先测 baseline。优先使用 `task.yaml.success_criteria`；缺失时默认 required hot case 至少 `1.10x`，并用 roofline/NCU 判断是否物理可行。有端到端占比时用 Amdahl law 排优先级。

### 多语言如何选择

由 `task.yaml.allowed_implementation_paths` 和 `env_manifest.yaml` 门控。Triton/CuTe DSL 是首选 JIT 路径；有 profile 证据表明 JIT 受限时，再升级 CUDA extension/CUTLASS。

### 是否优先接入开源实现

建议优先做 open source probe，但定位是 “上界探针 + 候选源”。开源实现必须通过同一套 correctness/benchmark，依赖必须合规，ABI 必须可映射。正确且最快才采用，否则作为性能上界或参考样例。

### 实现后如何 tuning

一次一改，correctness 过后 benchmark，再 profile 热 case。根据 NCU/roofline 读取对应 playbook。结构优化先于大规模 autotune；提升超过噪声阈值才 keep；停机时选 best-so-far。

### 如何评价优化后 kernel

Correctness 是门控；performance 看每个 required hot case 的 median speedup；efficiency 看 NCU SOL/roofline；robustness 看 final rerun、shape 覆盖、dtype/layout 约束和 fallback。

### 停止标准是什么

达标、接近物理上限、平台期、需要框架配合、预算耗尽、正确性阻塞都可以停。任何停止必须带证据，且交付 best-so-far 而不是 latest。

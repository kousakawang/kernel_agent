# Kernel Agent 工程设计（ops48）

> 本文回答 `kernel_agent_new_design.md` 中的两组问题：
> 1. agent 工程的**目录结构**（系统 prompt / skill / 知识库），重点解决"输入内容太多"的上下文负担问题。
> 2. agent 的**工作流程**：没有目标时如何定目标、多语言/开源实现如何选择、如何 tuning、评价体系、停止标准。
>
> 设计以**当前已有资产**为基线，不另起炉灶。文中"Kernel Agent"指当前仓库根目录下的 active 工程 `kernel_engineer/`，它的唯一输入是 Framework Engineer 产出的独立 `task_pack/`。

---

## 0. 设计前提：已经有什么

当前已经落地、本设计直接复用、不重造的部分：

- **交接契约 `task_pack/`** 已经定型（参考 `kernel_agent/backup/examples/qwen35_gdn_extend_task_pack/`）。它自带：
  - `task.yaml`：ABI、目标、`success_criteria`、`allowed_implementation_paths`、禁止修改项。
  - `snapshots/`（replay 真值）、`shape_list.json`（摘要索引）。
  - `original_impl.py`（性能 baseline）、`reference_impl.py`（correctness 真值）、`candidate_impl.py`（待替换入口）。
  - `correctness_test.py` / `benchmark.py`（**不可改**的评测口径）+ `scripts/run_*.sh`。
  - `env_manifest.yaml`（Triton / CuTe DSL / CUDA extension / NCU 的 `available` 与 `usable_for_task`）。
- **Kernel Agent 骨架** 已存在于 `kernel_engineer/`：
  - `prompts/kernel_engineer.md`（系统 prompt）
  - `skills/`：`task_triage`、`task_pack_optimization_protocol`、`kernel_optimization_loop`、`triton_cuda_codegen`、`nvidia_ncu_analysis`、`framework_feedback`
  - `templates/`：`task_acceptance_review`、`benchmark_report`、`kernel_constraints`、`kernel_delivery_package`、`framework_change_request`、`iteration_log`

**本设计的增量**集中在三处（详见第 7 节）：补 `knowledge/`（按需知识库）、补 `tools/`（可执行的 profiling/日志解析脚本）、把系统 prompt 收敛成显式的**三层加载模型**。

---

## 1. 参考仓库调研（简述）

调研了 `backup/reference_repos/ref/` 下四个仓库，分两种范式：

| 仓库 | 范式 | 一句话 |
|---|---|---|
| **AKO4ALL** | prompt/skill 驱动 | 单个 `SKILL.md` 驱动 "profile→改→bench→记录→commit" 闭环，知识全靠按需读文件 |
| **kernel-design-agents (KDA)** | 工作流参考 | 核心思想就是"可复用工作流"与"任务工作区"彻底分离，知识/profiler 作为外部 skill 链接进来 |
| **KernelAgent (Meta)** | 代码编排 | Python 编排 + Jinja 模板拼 prompt，知识库用 **RAG 分层检索**（瓶颈类型→技术→代码样例）按需注入 |
| **autokernel** | 代码编排 | 一份大 `program.md` playbook + 固定 Python 工具链；强调 **bench 输出落盘再 grep**，不淹没上下文 |

### 四个仓库对五个关键问题的做法

| 问题 | AKO4ALL | KernelAgent | autokernel | KDA |
|---|---|---|---|---|
| **没目标时定什么** | 仅"比 reference 快"，无上限 | 比 best/eager **≥10%** | profile + **Amdahl 排序**，人确认，隐含每核 2× | 必须人填 Task Contract |
| **语言选择 / 开源优先** | 鼓励换语言（Triton→CUDA），不查开源 | 仅 Triton | Triton+CUDA 双后端同接口，不查开源 | 由 contract 的"允许方案"决定 |
| **tuning 循环** | 一次一改→记录→commit（强约束"记录必须紧跟 bench"）；`--no-ref` 快排序 | NCU→roofline→LLM 诊断→生成→校验→bench；**beam search + Reflexion** | 一次一改→先 commit 再 bench→KEEP/REVERT，1% 算提升、同分取简单 | 一次一个候选，记录父子关系与证据 |
| **评价体系** | KernelBench：correct 门控 speedup，反作弊 | correctness 门控 + CUDA event + **NCU SOL roofline**，best-by-runtime/best-by-SOL 双轨 | **5 段 correctness** + 解析式 roofline + `fast_p` | 完全委托给 task 的 validate/eval 命令 |
| **停止标准** | 三触发：用户上限 / 证明物理下限 / 穷尽≥3 方向；停时 checkout 到最优 iter | roofline：≥95% SOL 或 5 轮收敛 <0.1% | `_should_move_on`：5 连续 revert / ≥90% peak / 120min / ≥2× 任一 | promotion gate：满足 contract 才晋级 |

### 给我们的启示（直接采纳）

1. **上下文管理 = 分层 + 按需 + 落盘**：lean 系统 prompt（AKO4ALL）+ 知识按需检索（KernelAgent）+ 工具输出落盘再 grep（autokernel）+ 工作流与任务区分离（KDA）。这四点正好回答"输入是不是太多了"——见第 2 节。
2. **目标要可度量且被物理上限约束**：先测 baseline，再用 roofline 给目标设天花板（KernelAgent + autokernel）。
3. **没有仓库做"开源优先"**，但这正是我们可以差异化的点——见 4.3。
4. **停止标准必须带证据、且回滚到最优 iter**，绝不"悄悄停"（AKO4ALL）。

---

## 2. 目录结构设计

### 2.1 核心原则：三层上下文加载模型（回答"输入太多"）

把所有内容按"何时进入模型上下文"分三层，**默认只有 L0 在上下文里**：

| 层 | 内容 | 何时加载 | 体量约束 |
|---|---|---|---|
| **L0 常驻** | 系统 prompt：角色、边界、闭环骨架、"去哪找更多" | 永远 | 单文件，目标 < 150 行 |
| **L1 按名加载** | skills：到某个阶段才读对应 skill 文件 | agent 进入该阶段时 `Read` 一次 | 每个 skill 单一职责、自包含 |
| **L2 按需检索** | 知识库：硬件 spec、语言文档、参考 kernel | 命中某瓶颈/某语言时，经 INDEX 定位后只读相关片段 | 切成小文件，**绝不整库进上下文** |

再叠加两条工程纪律（来自 autokernel / KDA）：

- **工具输出落盘再 grep**：`benchmark.py` / `run_ncu.sh` 的原始输出一律重定向到 `task_pack/docs/` 下的文件，agent 只读取/grep 需要的字段（median、speedup、瓶颈 section），不让原始日志淹没上下文。
- **任务区与工作流分离**：`kernel_engineer/`（可复用工作流，本设计的对象）与 `task_pack/`（任务数据）物理隔离，互不可见，唯一接口是 task_pack。agent 不去读 SGLang 框架代码（那是 Framework Engineer 的事）。

> 一句话：**系统 prompt 只告诉 agent "怎么找"，不把知识塞进去；知识在 L2 以小文件 + 索引存在，命中才读。**

### 2.2 目录树

```text
kernel_engineer/                      # = Kernel Agent 工程（可复用工作流，与 task_pack 隔离）
  prompts/
    kernel_engineer.md                # [L0] 系统 prompt：角色/边界/闭环骨架/加载指引（lean）
  skills/                             # [L1] 单一职责、按名加载
    task_triage.md                    #   接包验收：能不能开工
    kernel_optimization_loop.md       #   内循环骨架
    triton_cuda_codegen.md            #   实现路径选择（含开源探针，见 4.3）
    nvidia_ncu_analysis.md            #   profiling 诊断指南（配合 tools/parse_ncu.py）
    framework_feedback.md             #   何时/如何发 FrameworkChangeRequest
    perf_target_and_roofline.md       #   [新增] 定目标 + roofline 天花板
    tuning_and_autotune.md            #   [新增] 结构调优 + config autotune 方法
  knowledge/                          # [L2] 按需检索，绝不整库进上下文
    INDEX.md                          #   唯一"先读"的入口：把症状→该读哪个文件
    hardware/
      h20.md                          #   H20 spec：TFLOPS/HBM 带宽/SM 数/smem/L2/时钟
      _template.md                    #   新硬件按模板补
    languages/
      triton/cheatsheet.md            #   Triton 常用范式 + 约束（autotune/TMA/对齐）
      cutedsl/cheatsheet.md           #   CuTe DSL tile/MMA/pipeline 范式
      cuda/cheatsheet.md              #   CUDA/CUTLASS 范式
    playbooks/                        #   瓶颈→技术→代码样例（仿 KernelAgent 三层）
      memory_bound.md
      compute_bound.md
      low_occupancy.md
      small_shape_launch_bound.md
    samples/                          #   可直接借鉴的最小参考 kernel 片段
      *.py
  tools/                              # [新增] 可执行脚本：profiling / 日志解析
    parse_ncu.py                      #   解析 ncu csv/rep → 瓶颈类型 + 关键指标 JSON
    summarize_bench.py                #   解析 benchmark.py 输出 → 每 case speedup 表
    roofline.py                       #   解析式 roofline（无 NCU 时算天花板）
    pick_best_iter.py                 #   从 iteration_log 选最优 candidate 回滚
  templates/                          # 交付物模板（已存在）
    task_acceptance_review.md  benchmark_report.md  kernel_constraints.md
    kernel_delivery_package.md  framework_change_request.yaml  iteration_log.md
```

`task_pack/`（任务区，Kernel Agent 只读其内容、只写允许修改项）：

```text
task_pack/
  task.yaml  shape_list.json  env_manifest.yaml
  snapshots/  snapshot_runtime.py
  original_impl.py  reference_impl.py  candidate_impl.py   # ← 只允许改 candidate
  correctness_test.py  benchmark.py  scripts/run_*.sh      # ← 不可改的评测口径
  kernel_sources/                                          # ← agent 写自定义 kernel
  docs/                                                    # ← agent 写报告/落盘日志
    iteration_log.md  benchmark_report.md  ncu_*.json ...
```

### 2.3 系统 prompt（L0）怎么写

`kernel_engineer.md` 保持 lean，只放四件事（现有版本已接近，按此收敛）：

1. **角色与首期环境**：在给定 task_pack 下实现/优化高性能算子；首期 H20 + Triton/CuTe DSL/CUDA + Nsight Compute。
2. **职责边界**：负责 / 不负责清单（沿用现有；不猜模型语义、不改框架绕过 spec、不为数字破坏 correctness、不藏失败 shape）。
3. **闭环骨架（一段话）** + **三层加载指引**：明确"进入 triage 读 `skills/task_triage.md`，选实现路径读 `skills/triton_cuda_codegen.md`，遇瓶颈先读 `knowledge/INDEX.md` 再按指引读对应 playbook"。
4. **工程纪律**：先 correctness 再性能；工具输出落盘再 grep；一次一改并立即记录 `iteration_log`；NCU 指标必须与代码改动建立因果。

### 2.4 知识库（L2）怎么做到"按需"

这是回答"输入太多"的关键。采用 **INDEX 路由 + 小文件 + 瓶颈分层**：

- `knowledge/INDEX.md` 是唯一"先读"的文件，本质是一张**症状→文件**的路由表，例如：

  ```text
  DRAM 高 / SM 低         -> playbooks/memory_bound.md
  occupancy 低           -> playbooks/low_occupancy.md
  小 shape、launch 占比高  -> playbooks/small_shape_launch_bound.md
  目标硬件 = H20          -> hardware/h20.md
  写 Triton              -> languages/triton/cheatsheet.md
  ```

- 每个 `playbooks/*.md` 内部再分三层（仿 KernelAgent 的 bottleneck→technique→code-sample），但以**纯文件**形式存在：先给该瓶颈的判定标准，再给 2-3 个技术，每个技术指向 `samples/` 里的最小代码片段。agent 一次只读命中的那一个 playbook + 最多 1-2 个 sample。
- **演进路径**：文件 + INDEX 已能覆盖首期。当 `playbooks/` 与 `samples/` 增多到人工路由困难时，再引入 embedding 检索（KernelAgent 的 `RAG_based_prescriber` 模式），接口不变——agent 仍然"问 INDEX、读片段"。

这样无论知识库长到多大，**进入上下文的永远只是命中的少数小文件**。

### 2.5 tools/（可执行脚本）

skills 是"说明"，tools 是"可执行件"，两者配套。首期四个脚本即可，全部输出结构化 JSON/表，配合"落盘再 grep"：

- `parse_ncu.py`：跑/解析 NCU，按一组精选 metric（SOL、MemoryWorkload、SchedulerStats、occupancy、warp stall）输出瓶颈分类 + 关键指标 JSON，过滤掉非目标 kernel。
- `summarize_bench.py`：把 `benchmark.py` 的逐行 JSON 汇总成"每 hot case：baseline/candidate median + speedup + 是否达标"的表。
- `roofline.py`：无 NCU 时，从 shape 推算算术强度与 ridge point，给出"达到峰值百分比 + 是否带宽/launch bound"的解析估计（autokernel 的兜底做法）。
- `pick_best_iter.py`：停机时从 `iteration_log.md` 选出**最优而非最新**的 candidate 用于交付/回滚。

---

## 3. 工作流程设计

### 3.1 总览

```text
接包验收(triage) ──fail──> task_acceptance_review.md (NEEDS_MORE_INFO / REJECT)
      │ pass
      ▼
建立 baseline + 定目标 ──> 锁定 target & roofline 天花板
      │
      ▼
选实现路径(含开源探针) ──> Triton / CuTe DSL / (escalate) CUDA / CUTLASS
      │
      ▼
┌─────────── 优化内循环（一次一改）───────────┐
│ 实现/改 → correctness → benchmark → profile  │
│        → 分析(查 knowledge) → 记录 iteration  │
└──────────────┬───────────────┬──────────────┘
        命中停止标准         需框架配合
            │                   │
            ▼                   ▼
   pick_best_iter +        framework_change_request.yaml
   KernelDeliveryPackage
```

### 3.2 （a）没有目标时如何定目标

**原则：目标永远先来自 `task.yaml.success_criteria`；缺失时按规则自动派生，并用 roofline 设天花板。** 对应新增 skill `perf_target_and_roofline.md`。

1. **先建 baseline，再谈目标**：跑 `bash scripts/run_benchmark.sh` 拿到 baseline（优先 `original_impl`，不可用时用 `TARGET=candidate` 的真实候选）逐 hot case 的 median。没有 baseline 不允许设目标。
2. **目标取值优先级**：
   - a. `task.yaml` 写了 `performance`（如本例"hot case ≥1.10×，否则解释硬件/launch bound"）→ 直接用。
   - b. 没写 → 默认 **"required hot case 比 baseline 快 ≥10%，且 correctness 不破、stability 达标；或用 roofline 证明已近物理下限"**（KernelAgent + AKO4ALL 合成）。
3. **用 roofline 给目标封顶**：用 `tools/roofline.py`（或 NCU SOL）算该 op 在 H20 上的理论上限。若 baseline 已达 ~90% SOL，则目标退化为"维持正确性 + 小幅收尾"，并在交付里直接给出"已近上限"的结论，避免无意义空转。
4. **多 case 的目标聚合**：必须**所有 required hot case** 各自达标；逐 case 报告，不许用平均掩盖个别劣化。

> 这样即使用户/框架没给数字，agent 也有一个**可度量、被物理约束**的目标，不会无休止优化。

### 3.3 （b）多语言如何选择 & 是否优先接入开源实现

#### 语言选择

由 `env_manifest.yaml` 的 `available && usable_for_task` 与 `task.yaml.allowed_implementation_paths` 共同**门控**（沿用现有 `triton_cuda_codegen.md`）：

- **首选 JIT**：Triton 或 CuTe DSL —— 最快拿到可改可测的候选。按瓶颈选：偏访存/elementwise/fusion/layout → Triton；需要细粒度 tile/向量化/smem/pipeline/MMA → CuTe DSL。不因"某现成 SGLang CuTe backend 要 SM100"就拒绝在 H20 自写 CuTe。
- **升级路径**：仅当 JIT 路径经 **profile/benchmark 证据**证明触顶，且 `env_manifest` 证明 nvcc/头文件/库链路可用，才上 CUDA extension / CUTLASS。
- **反"偷懒"**（AKO4ALL 教训）：明确禁止"停在 PyTorch 只调 config 就交差"。换语言/重写是合法且被鼓励的手段。

#### 是否优先接入开源实现？——**是，但作为"上界探针 + 候选源"，不是默认交付**

这是用户明确问的点。建议把"开源探针"作为**选路径阶段的一个可选前置步骤**（写进 `triton_cuda_codegen.md`）：

- **何时做**：该 op 存在成熟开源 kernel（如 FLA / triton 社区实现），且其依赖满足 `env_manifest`（**不得新增依赖**——开源代码须 vendoring 进 `task_pack/kernel_sources/`），且能映射到 task 的 ABI。
- **怎么用**：让它跑过同一套 `correctness_test.py`，并用 `benchmark.py` 测它的性能。两种结果：
  - 开源实现**正确 + 最快 + 可集成** → 直接采纳/适配为 candidate（站在巨人肩上）。
  - 否则 → 把它测到的性能当作**经验性能天花板/目标**，用自研实现去追，并在 `iteration_log` 记录差距。
- **为什么不是无脑直接用**：licensing、ABI 不完全匹配、依赖策略、以及它可能并非为当前 hot shape/H20 调优。所以开源实现的价值首先是**廉价地拿到一个强 baseline 和性能上界**，降低"目标定多高"的不确定性。

> 直觉：开源探针让你**先知道天花板在哪**，再决定自己写还是直接用，避免闭门造车。

### 3.4 （c）实现完后如何 tuning

沿用 `kernel_optimization_loop.md` 的骨架，叠加来自调研的护栏，分两个层次（写进新增 `tuning_and_autotune.md`）：

**内循环（一次一改）**：实现/改 → correctness → benchmark → profile（热 case 跑 NCU）→ 查 `knowledge/` 分析 → 记录 → 下一轮。

护栏：
- **一次只动一个独立因素**，便于归因。
- **记录必须紧跟 bench**（AKO4ALL）：每次 benchmark 之后，立即把结果写入 `docs/iteration_log.md`（candidate ID、改动点、correctness、benchmark 摘要、profiler 摘要、保留/放弃原因）。
- **KEEP/REVERT 规则**（autokernel）：相对当前 best，提升 ≥ ~1%（超过噪声）才 KEEP；劣化或持平则 REVERT；**同分取更简单实现**。始终维护"当前最优 candidate"。
- **signal vs verdict**（AKO4ALL 省时）：日常迭代可用 `TARGET=candidate` 只测候选自身做快速排序；只在"宣布达标/交付"前用 `TARGET=both` 跑完整 reference 对比作为最终判定。
- **stall 重评估**：连续 3 轮 <3% 提升时，先暂停重新 profile / 查 knowledge / 必要时 web 检索，找到新方向再继续，而不是硬磨同一方向。

**调优两个层次**：
1. **结构/算法**：由 NCU 瓶颈类型驱动（DRAM 高→减读写/融合/coalescing；occupancy 低→降寄存器/smem；long scoreboard→增并行/缩依赖链；小 shape→persistent/fusion/或发 FrameworkChangeRequest 做 batching）。
2. **config autotune**：结构固定后，对 hot shape 扫 block/tile/`num_warps`/`num_stages` 等；做**有限 specialization**，不为所有 shape 写复杂分支。

何时跳出去发 `FrameworkChangeRequest`：需要 contiguous/特定 stride、metadata 预计算、persistent workspace、prefill/decode 拆核、权重/cache 重排、padding/alignment 才能稳定提速时（沿用 `framework_feedback.md`）。

---

## 4. 评价体系

评价**完全基于 task_pack 内不可改的口径**，分四个维度，外加反作弊。对应 `benchmark_report.md` 里的一张 scorecard。

| 维度 | 口径 | 工具 | 判定 |
|---|---|---|---|
| **正确性（门控）** | `correctness_test.py`：snapshot-golden 为主、reference-replay 为辅，固定 tolerance，含 mutable-arg 比对（输入也保存并比对，见下） | `scripts/run_correctness.sh` | **二值门控**：不过则性能无意义 |
| **性能** | 逐 hot case：`speedup = baseline_median / candidate_median`，CUDA event 计时、warmup+repeat、**每次调用 fresh inputs** | `benchmark.py` + `tools/summarize_bench.py` | 所有 required hot case 各自达标 |
| **效率/上限** | 达到 SOL 的百分比（有 NCU）或解析式 roofline 百分比（无 NCU） | `tools/parse_ncu.py` / `tools/roofline.py` | 判断"还有没有头部空间" |
| **稳定性** | 最终重测在 best 测量值 **5% 以内**（task.yaml stability） | `benchmark.py` 重跑 | 抖动过大不算达标 |

**关于 mutable inputs**：采纳 `advices.md` 的结论——不依赖人工标注哪些参数会被原地修改；correctness 口径已是"保存并比对前后输入"（`correctness_test.py` 对 `mutable_arg_paths` 做 `assert_tree_close`），最保险。

**反作弊**（沿用 + 强化）：
- 每次调用 fresh inputs（harness 已保证），禁止依赖单一 seed / 输入分布。
- 禁止改 golden / 放松 tolerance / 改计时规则。
- snapshots、`snapshot_runtime.py`、`benchmark.py` 不可改。
- 对异常 speedup（如 >10×）要求在 iteration_log 给出物理解释，否则视为可疑。

---

## 5. 停止标准

**复合标准（autokernel 多触发为骨架，叠加 AKO4ALL 的"带证据 + 回滚最优"）。命中任一即停：**

1. **达标（成功停）**：所有 required hot case 达到目标 speedup，correctness 通过，stability 达标。
2. **物理下限（成功停）**：roofline/NCU 证明已近上限（≈ ≥90% SOL，或明确 memory-bandwidth / launch-overhead bound），**且在 iteration_log 引用了证据**。
3. **平台期（收尾停）**：最近 3 个有效迭代提升 <3%，**且**重新 profile 无新方向，**且**已尝试 ≥3 类不同优化方向仍无收益。
4. **被阻塞**：必须有 `FrameworkChangeRequest` 才能继续 → 输出请求并挂起。
5. **预算耗尽**：到达迭代/时间上限 → 带 best-so-far 交回。

**停机动作**（关键，来自 AKO4ALL）：
- 用 `tools/pick_best_iter.py` 回滚到**最优 iteration 而非最新**。
- 达标 → 产出 `kernel_delivery_package.md` + `benchmark_report.md` + `kernel_constraints.md`，注明支持/不支持范围与 fallback。
- 未达标 → 仍产出交付物，但在其中给出**瓶颈解释**（如已近带宽上限、e2e 占比不足、需框架配合）。
- **绝不悄悄停**：任何停止都必须在 `iteration_log.md` 留下原因与证据。

> 默认就有 1/2/3 三个"自然收敛"触发；4/5 是"交回人/框架"的安全阀。这样 agent 既不会无限优化，也不会在还有明显空间时过早放弃。

---

## 6. 与现状的差异 & 落地清单

相对当前 `kernel_engineer/`，本设计的**增量**（按优先级）：

1. **新增 `kernel_engineer/knowledge/`**（L2 按需知识库）：`INDEX.md` + `hardware/h20.md` + `languages/*/cheatsheet.md` + `playbooks/*.md` + `samples/`。**这是回答"输入太多"的核心落点**——知识从此不进系统 prompt，命中才读。
2. **新增 `kernel_engineer/tools/`**（可执行脚本）：`parse_ncu.py` / `summarize_bench.py` / `roofline.py` / `pick_best_iter.py`。现有 `nvidia_ncu_analysis.md` 只是"指南"，缺配套可执行件。
3. **新增两个 skill**：`perf_target_and_roofline.md`（4.2）、`tuning_and_autotune.md`（4.4 的护栏 + autotune）。
4. **收敛系统 prompt** `prompts/kernel_engineer.md`：补"三层加载指引"和"工具输出落盘再 grep"两条纪律，保持 lean。
5. **扩 `triton_cuda_codegen.md`**：加入 3.3 的"开源探针"前置步骤（vendoring + 同口径评测 + 当上界）。
6. **强化停止/评价表述**：把第 4、5 节的 scorecard 与复合停止标准写进 `kernel_optimization_loop.md` / `task_pack_optimization_protocol.md`，替换目前较粗的版本。

**不改的**：task_pack 契约、`correctness_test.py` / `benchmark.py` 口径、framework/kernel 双目录隔离、交付物模板集合。

---

## 7. 一页速记

- **上下文不爆 = 三层（L0 prompt / L1 skill / L2 知识）+ 落盘 grep + 任务区隔离**；知识用 INDEX 路由按需读。
- **定目标**：先测 baseline，目标取自 task.yaml，缺则默认 ≥10%，并用 roofline 封顶。
- **选语言**：env + task.yaml 门控，JIT 优先（Triton/CuTe DSL），证据驱动升级 CUDA/CUTLASS。
- **开源实现**：作为"上界探针 + 候选源"优先探一次，正确且最快且可集成则直接用，否则当目标追。
- **tuning**：一次一改 → 即记 → KEEP/REVERT（≥1%、同分取简单）→ stall 重评估；结构调优 + config autotune 两层。
- **评价**：correctness 门控 → 逐 case speedup → roofline 效率 → stability，配反作弊。
- **停止**：达标 / 近物理下限 / 平台期 / 需框架配合 / 预算耗尽，任一即停；停时回滚到最优 iter 并带证据交付。

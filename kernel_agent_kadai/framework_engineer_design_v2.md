# Framework Engineer 升级设计 V2

> 本文是对 [framework\_engineer\_design\_review.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent_kadai/framework_engineer_design_review.md) 中提出问题的最终收敛方案。
> 目标：把 framework\_engineer 从"单 target 走通"升级为"支持模块级和 low\_level 两种入口、按参考资料丰富度分级、多路径评分排序的 task\_pack 生成流水线"。

***

## 1. 整体产品方案

按用户能提供的输入形态，framework\_engineer 支持两种 target 类型：**high\_level（模块）** 和 **low\_level（具体 kernel）**。二者共享同一套 low\_level 优化主链路，区别仅在 high\_level 需要额外的**多路径分解 + 路径排序**前置流程。

### 1.0 与 deploy-agent 的职责边界（前置说明）

framework\_engineer / kernel-agent 假定拿到的 **sglang 服务启动命令已经是最优的**，不承担后端探索、启动参数调优、分布式调优等责任。这些属于另一个产品 **deploy-agent** 的范畴。

实际使用时的分工：

- **deploy-agent**：调优模型部署参数（后端选择、分布式配置、调度参数），产出可用的启动命令
- **kernel-agent**（本文档）：基于确定好的启动命令做算子层面的优化

用户准备阶段的工作：**先手动确认对应硬件 + 模型上哪几条后端路径可用**，把可用的启动命令 + 测试命令记录下来交给 framework\_engineer。

### 1.1 High\_level target：多路径分解 + 排序

用户给出的是一个模型模块（例如 `Qwen3_5GatedDeltaNet.forward`），希望优化模块内部实现。

**核心思路**：sglang 对同一 layer 通常有多条后端实现路径（如 GDN 有 triton / flashinfer / cutedsl；GQA 有 flashinfer / fa3 / triton；MLA 有 flashinfer / flashmla / fa3 / cutlass\_mla）。不同路径的算子划分粒度不同，**通用算子库的路径（flashinfer / fla / flash\_attn）接口定义更清晰，天然带 UT / 文档**。

framework\_engineer 会对用户在 config 中列出的所有 backend 路径分别跑一次 KID 分解，为每条路径生成一个 **workspace**（内含 K 个 low\_level target 的 task\_pack）。所有 task\_pack 都跑完 problem\_translate 定级后，最后一步 `summarize-workspaces` 按公式对 workspace 排序：

**评分公式**：

```
workspace_score = Σ_i (level_i.weight × kernel_time_cost_i / total_time)
weight: L4=4, L3=3, L2=2, L1=1
```

**含义**：某个 workspace 里高等级 target 占的耗时比例越大，该 workspace 对 kernel\_engineer 越友好。kernel\_agent 的优化按 workspace\_score 从高到低顺序开工。

**重要**：评分依赖真实的 L1/L2/L3/L4 判定结果（Step 3 输出），因此**必须在 Step 4 validate 完成后才能算**，不能在 KID 阶段预判。

### 1.2 Low\_level target：直接指定

用户已知要优化哪个 low\_level 接口（GEMM / RMSNorm / 某个 triton kernel），直接指定 `target_file + target_line`。KID 仍然会跑一次（只做源码定位、不做拆解和耗时统计），产出的 workspace 里只有一个 low\_level target。之后流程完全一致。

### 1.3 四级 Oracle

无论 high\_level 还是 low\_level，每个 low\_level target 都要走同一套四级 oracle 判定，由 `problem_translate` skill 完成。判定为**能力认证驱动**：能凑齐哪一级需要的资料 + 通过对应验证，就是那一级。

| Level             | 用户或 agent 能提供                                                        | task\_pack 附加资料                                                                           | 默认 tolerance        | allow\_alt\_math\_paths | 典型例子                                                                                         |
| ----------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------- | ----------------------- | -------------------------------------------------------------------------------------------- |
| **L4 (best)**     | 可执行的 pytorch/python 参考实现 + **能通过 snapshot 跑通 UT（golden check pass）** | `python_reference/ref_impl.py` + `tests/ut_lv4.py` + manifest 标记 origin                   | 严格（1e-3 \~ 1e-4）    | false                   | GEMM / RMSNorm / RoPE / 在 sgl 内部 tests、benchmark、或 third-party tests 里能找到 PyTorch 参考的 kernel |
| **L3 (good)**     | 无法生成完全一致的 UT，但能给出**计算逻辑等价的参考代码片段**                                   | `logic_mapping/`：拷贝源文件到 `logic_mapping/references/` + `mapping.md`（含 kind: L3 / 行范围 / 说明） | 适中（1e-2）            | true                    | `chunk_gated_delta_rule_fwd_intra` 类可在 transformers 找到循环等价但粒度对不齐                             |
| **L2 (normal)**   | 找不到 kernel 级等价，能给**上层模块级别**的参考代码                                     | `logic_mapping/`：拷贝源文件到 `logic_mapping/references/` + `mapping.md`（含 kind: L2 / 行范围 / 说明） | 适中（1e-2 \~ 5e-2）    | true                    | 只能提供上层模块整体代码                                                                                 |
| **L1 (just try)** | 只有原始框架实现                                                             | 无额外资料（KID 已抽取的四个源码文件仍在）                                                                   | 严格（snapshot golden） | false                   | 兜底；无任何参考可用；或用户未提供任何外部参考仓库线索                                                                  |

**L3 与 L2 的统一形式**：两者都通过 `logic_mapping/` 目录承载参考资料（源码文件 + `mapping.md` 行号索引）。区别仅在 `mapping.md` 里的 `kind` 字段（`L3` 表示逻辑等价片段、`L2` 表示上层模块）和参考代码粒度。这样 kernel\_engineer 消费时只需读 `logic_mapping/`，不再区分两个目录。

**L3 是关键增量**：承认现实——很多 kernel 写不出可执行 PyTorch 等价，但能指着 transformers 的某几行说"这就是它在做的事"。

**外部参考仓库是开放配置**：`problem_translate` 消费的外部参考不局限于 transformers，而是一份**开放配置**（`external_references`），每条记录含：仓库路径 + 参考文件列表 + 每个文件的行号范围（都可选，允许列表）。若用户什么线索都不给，在L4判定不成立之后判定直接**默认降级到 L1**（agent 不会主动去猜其他仓库）。

**判定顺序**：agent 依次尝试 L4 → L3 → L2 → L1，凑齐哪一级资料就停在哪一级。

***

## 2. 默认 workflow：从原始需求到 task\_pack

以 high\_level target 为主线描述。low\_level target 会在对应位置说明分支。

### 2.1 用户前置输入（一次性提交）

| 项目                                         | 谁提供    | 备注                                                                                       |
| ------------------------------------------ | ------ | ---------------------------------------------------------------------------------------- |
| **N 条启动命令** `service_cmds`                 | 用户     | 每条对应一条 backend 路径；已假定是最优命令（deploy-agent 阶段完成）                                            |
| Workload / 测试命令 `workload_cmd`             | 用户     | 触发目标模块被调用的请求                                                                             |
| Forward boundary 位置                        | 用户     | 用于 snapshot 分组                                                                           |
| **High\_level**：模块 target file/line        | 用户     | 例如 `Qwen3_5GatedDeltaNet.forward`                                                        |
| **Low\_level**：low\_level target file/line | 用户     | 跳过 KID 拆解，只做源码定位                                                                         |
| sglang 仓库源码根路径                             | 用户     | `sglang_repo_root`，Step 0.5 和 Step 3 都要用                                                 |
| **外部参考仓库配置** `external_references`         | 用户（可选） | 开放式列表，每条含 `repo_root` + 可选 `files` + 可选 `line_ranges`；供translate\_problem agnet做L3/L2判定用 |
| Third-party clone 目的路径                     | 用户     | `third_party_cache`，Step 0.5 拉仓库时的落地位置                                                   |

### 2.2 Step 0.5 —— 第三方仓库准备 + kernel 源码定位（两个 skill）

**目的**：在 KID 执行前把 kernel 涉及的第三方算子库仓库全部准备到本地，并沉淀一套「按形态溯源」的规则，供 KID 精确定位源码 + 给 problem\_translate 提供 UT 查找路径。

Step 0.5 含两个 skill（详见 8.1）：

- **`resolve-third-party`**：定版本 + clone，产出 `third_party_manifest.json`
- **`locate-kernel-source`**：给定 `(接口名, 形态)` 返回四层源码位置；其 deterministic helper 供 KID 直接复用，KID 因此**不必自己实现溯源**

> **收敛说明（以 [KID_and_locate_source_desgin_v2.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/kernel_agent_kadai/KID_and_locate_source_desgin_v2.md) 为准）**：`locate-kernel-source` 定稿为**三层**（Layer1 deterministic 定位 CLI / Layer2 agent 兜底 / **Layer3 物料抽取 = 原 `import-decomposition`**）。因此本文档正文里凡把 `import-decomposition` 归在 Step 1(KID) 名下的描述（§2.3 / §2.8 流程图 / §2.9 表 / §8.2）均已过时——该 CLI 现属 locate。

**执行者**：**skill 主路径**（agent 独立运行，隔离上下文）；不做成 CLI，因为定位过程需要综合多种线索、可能需要实际运行服务打点。

**为什么单独拆一步**：

- 切换到 flashinfer / fla / flash\_attn 等通用路径的核心收益之一就是"接口更清晰 + 有 UT 可参考"。若源码定位不到，收益直接废掉一半
- **problem\_translate 的 L4 认证会强制查这些仓库的测试目录**，仓库必须先在本地
- 溯源规则集中在一处（`locate-kernel-source`），KID 只产出 `(接口名, 形态)`，避免溯源逻辑在 KID 里重复维护
- 单独执行 skill 可以把定位过程中的临时上下文（打 log、grep、看编译输出）与后续工作隔离，只保留最终结论文件

**输入**

- N 条启动命令 + 测试命令
- `sglang_repo_root`
- `third_party_cache`（clone 目的路径）

**Agent 工作规则**

允许：

- 查阅 sglang 仓库的依赖锁定文件（pyproject / requirements / setup.py）
- 查阅 sglang 内嵌路径（如 `sgl-kernel/3rdparty/`）确定是否已有本地版本
- 查阅 `importlib.metadata` / site-packages 确认运行时实际版本
- 查阅 CMakeLists / Makefile / compile\_commands.json 定位 sgl-kernel 接口的实际实现归属
- 实际运行服务和测试打点确认（**不是必须**，仅在版本线索不明确时使用）
- 在临时目录里打 log / 加 print / 用 gdb 等方式做定位调查
- 对任何在依赖分析中出现的仓库 URL 执行 clone（**无白名单限制**）

**严格禁止**：

- 破坏当前运行环境：不重新编译 sgl-kernel、不修改 sglang 源码、不安装新包
- clone 下来的仓库仅作为参考，不链接编译进当前环境
- 定位失败时不做假设，直接标记 FAILED 并提示人工介入

**Clone 策略**：

- 优先 `git clone --depth=1 --branch=v{pinned_version}`
- 若 tag 不存在则 fallback 到具体 commit hash
- Cache 命中（同版本已存在）跳过 clone

**产物**

```
<output_root>/
  third_party_manifest.json           # 每个仓库记录：
                                      #   name / resolution_source / local_path /
                                      #   version / sglang_pinned_version / version_mismatch /
                                      #   tests_dir_candidates[] / examples_dir_candidates[] /
                                      #   evidence（简要说明为什么选这个仓库/版本）
  missing_repos.md                    # 若有 FAILED 项，说明缺什么、可能原因、建议手动补齐的路径
```

**特殊约定**：

- flash\_attn / flash\_mla 明确不 clone 上游（sgl-kernel 里只有头文件，实现在 sglang 仓库内的 `3rdparty/` 子目录，用这个路径即可）

**人工可修改**

- 编辑 `third_party_manifest.json` 补 P1 路径或修正版本
- 编辑 `missing_repos.md` 后手动补齐

### 2.3 Step 1 —— KID 多路径分解 + 源码定位

**目的**：对每条 backend 路径跑 KID，得到该路径下的 low\_level target 列表 + 耗时占比 + **每个 low\_level 的四个源码文件**。**不做**评分选优（评分需要真实 level 判定结果，见 Step 5）。

**执行者**：CLI 主路径（现有 KID 增强 + 新增 `import-decomposition`）。

**High\_level target 输入**

- N 条启动命令（backend\_paths\_to\_try）
- 测试命令
- high\_level 模块的 file/line
- `third_party_manifest.json`

**Low\_level target 输入**

- 单条启动命令 + 测试命令
- low\_level target 的 file/line
- `third_party_manifest.json`
- 只做源码定位，跳过拆解和耗时统计

**执行内容**

1. **KID 增强**（`framework_engineer/kernel_interface_decomposer` 现有工具升级）：
   - 入口支持 high\_level / low\_level 双模式
   - 支持一次性接受多条启动命令，逐路径产出独立分解结果（避免文件冲突）
   - 每个 low\_level target 的 schema 必须严格包含：
     - `interface_definition`：low\_level 接口定义所在 python 文件 + 行号（内含 kernel launch 语句）
     - `py_cpp_binding`：py↔c++ 接口绑定文件（如 `*_extension_*.cc` / pybind 注册代码）；triton/cuteDSL 类**置空**
     - `kernel_header`：kernel 定义所在头文件 `.h`（保留是为了注释信息）；triton/cuteDSL 类**置空**
     - `kernel_impl`：kernel 实现文件（py / cu / cpp）
   - 每个 low\_level 的四个字段都要带行号范围（不是整个文件）
2. **`import-decomposition`** **CLI**：读 KID schema，对每个 low\_level 把四个源码文件**拷贝**到该 low\_level 的专属文件夹，附一个 `read_hints.txt` 说明每个文件该 read 的行数范围；把最终文件夹路径**回填**到 KID schema。

**产物**（high\_level target 场景）

```
<output_root>/
  workspaces/
    <backend_name_1>/                  # 一条 backend 路径 = 一个 workspace
      decomposition_schema.json        # 该路径下所有 low_level 的分解结果
                                        # 含耗时占比 / kernel 类别 / 源码定位 /
                                        # 回填的 kernel_sources 文件夹路径
      kernel_sources/
        <low_level_1>/
          interface_definition.py     # 若适用
          py_cpp_binding.cc            # 若适用（否则空文件 + 说明）
          kernel_header.h              # 若适用
          kernel_impl.{py,cu,cpp}      # 若适用
          read_hints.txt               # 每个文件的 read 行数范围
        <low_level_2>/
          ...
    <backend_name_2>/
      ...
```

**产物**（low\_level target 场景）

```
<output_root>/
  workspaces/
    single/
      decomposition_schema.json        # schema 里不含耗时，只含单个 low_level 的源码定位
      kernel_sources/
        <low_level_1>/
          ...
```

**人工可修改**

- 直接编辑 `decomposition_schema.json`：删掉不想优化的 low\_level 或调整 task\_id 命名
- 补充 KID 抽不干净的 helper 文件到 `kernel_sources/<low_level>/`

### 2.4 Step 2 —— Task pack 批量创建（level 默认为 1）

**目的**：把每个 workspace 里的 K 个 low\_level target 各自转成一个 task\_pack。此时所有 task\_pack 的 oracle\_level 默认为 1（未定级）。

**执行者**：CLI 主路径 + 少量人工填写。

**输入**：Step 1 产出的 workspace（每个 workspace 独立处理）。

**执行流程**（每个 workspace 内）

1. **`workspace-to-config`** **CLI**：
   - 读 `decomposition_schema.json`
   - 生成一份 task\_pack 生成任务的 config 草稿：包含启动命令、K 个 low\_level 的 file/line 等信息
   - **剩余字段（forward\_boundary\_file / forward\_boundary\_line 等）留空供人工填写**
2. **人工填写 config 空白项** → 得到完整 config
3. **走现有 phase1 主链路**（已完备）：
   - `scaffold-task-pack` → `run-baseline` → `resolve-interface` → `probe-target-calls` → `capture-snapshots` → `select-snapshots` → `generate-harness` → `probe-env`
   - **需要小改造**：config schema 要支持一次性输入多个 low\_level target（因为分解已经在 KID 完成，不需要 fw\_engineer 再分）
4. **`import-kernel-sources-to-taskpack`** **CLI**：
   - 把 workspace 里每个 low\_level 对应的 `kernel_sources/<low_level>/` 子文件夹拷贝到对应 task\_pack 的 `original_source/kernels/<low_level>/`
   - 从 `decomposition_schema.json` 抽出该 low\_level 的耗时占比 / 源码索引，写入 task\_pack 的 `level_decision.yaml`（新文件）
   - `level_decision.yaml` 初始内容：
     ```yaml
     oracle_level: 1                                  # 默认 L1
     evidence_paths:                                  # 各 level 的证据文件位置（Step 3 填写）
       l4_python_reference: null
       l4_ut: null
       l3_logic_mapping: null
       l2_module_reference: null
     kernel_time_cost_us: <from schema>
     kernel_time_ratio: <from schema>
     source_index:
       interface_definition: <path>
       py_cpp_binding: <path>
       kernel_header: <path>
       kernel_impl: <path>
     ```

**产物**（每个 workspace 内）

```
workspaces/<backend_name>/
  decomposition_schema.json           # Step 1 已有
  kernel_sources/                     # Step 1 已有
  task_packs/                         # Step 2 新增
    <low_level_1>_task_pack/
      task.yaml
      snapshots/
      original_source/
        kernels/<low_level_1>/        # 从 workspace/kernel_sources 拷贝
      original_impl.py / reference_impl.py / candidate_impl.py
      correctness_test.py / benchmark.py
      level_decision.yaml             # 新文件，oracle_level=1
      scripts/ / env_manifest.yaml / docs/
    <low_level_2>_task_pack/
      ...
```

**注意**：validate-task-pack 在这个阶段跑一次（现有能力 + 新增对 L1 的检查），确认 task\_pack 完整。因为默认全是 L1，所以这一次跑主要是文件完整性验证。

### 2.5 Step 3 —— `problem_translate` agent 定级

**目的**：为每个 task\_pack 认证 oracle\_level（L4/L3/L2/L1），并写入对应参考资料。

**执行者**：**agent（独立上下文）** + CLI `run-problem-translate` 批量编排。

**关键定位**：这是一个**独立工作、独立上下文**的 agent。每个 task\_pack 一次调用，agent 只看这一个 task\_pack 相关的信息，不共享主 agent 对话历史。skill markdown 必须完全 self-contained。

**Agent 输入**（每次调用一个 task\_pack）

- task\_pack 目录（含 `level_decision.yaml` / `original_source/kernels/<low_level>/` / `snapshots/`）
- 外部信息：
  - `external_references`（用户提供的开放式配置：仓库路径列表 + 每个仓库可选的文件路径 + 可选的行号范围）
  - `sglang_repo_root`（用于开放搜索 tests / benchmarks）
  - `third_party_manifest.json`（用于开放搜索 tests\_dir / examples\_dir）

**Agent 工作流程**

1. **前置强制步骤（开放式搜索）**
   - 在 sglang 仓库任意位置搜索该 kernel API 的测试文件——**推荐但不限于** `sglang/sgl-kernel/benchmark/` 和 `sglang/sgl-kernel/tests/`
   - 在对应 third-party 仓库任意位置搜索匹配文件——**推荐但不限于** `tests/` / `examples/`
   - agent 自主判断搜索范围；测试内容可能不在固定目录下
   - 记录找到的候选到 report
2. **尝试 L4**
   - 优先复用测试文件里的 PyTorch 等价实现作为 python\_reference
   - 若无现成参考，agent 阅读 kernel 源码后 draft  PyTorch/基础python 等价
   - 生成 `tests/ut_lv4.py` 加载全部 selected snapshots，跑 python\_reference vs snapshot pre/post/outputs 全量比对
   - **UT 跑通** → level=4，保留 UT 作为证据
   - UT 跑不通 → 降 L3
3. **尝试 L3**（依赖 `external_references`；无外部参考直接跳到 L1）
   - agent 阅读 kernel 语义 + 遍历 `external_references` 里的仓库/文件/行号范围
   - 找到**计算逻辑等价的表达片段**（不要求一对一，可以是几行代码块）
   - 拷贝命中的源文件到 `logic_mapping/references/<repo>/<file>`（保留原始注释）
   - 生成 `logic_mapping/mapping.md`（`kind: L3` + 引用文件 + 行范围 + 对应关系 + 差异点）
   - → level=3；找不到 → 降 L2
4. **尝试 L2**（依赖 `external_references`；无外部参考直接跳到 L1）
   - 在 `external_references` 里定位上层模块级参考代码（比 L3 粒度更粗）
   - 拷贝源文件到 `logic_mapping/references/<repo>/<file>`
   - 生成/追加 `logic_mapping/mapping.md`（`kind: L2` + 引用文件 + 行范围 + 说明）
   - → level=2；找不到 → L1
5. **L1 保底**
   - 无额外操作（原始 kernel 源码已在 Step 1 抽取好）
   - 若用户根本未提供 `external_references`，L3/L2 尝试直接跳过，判定为 L1
6. **更新** **`level_decision.yaml`**
   - `oracle_level` 改为最终判定值
   - `evidence_paths` 填入对应文件路径
   - 追加 `translate_notes` 说明为什么停在这一级

**产物**（task\_pack 内新增）

```
task_pack/
  python_reference/                        # L4 才有
    ref_impl.py
    manifest.json                          # origin: sgl_internal_test | third_party_test | agent_generated
  tests/                                    # L4 才有
    ut_lv4.py                              # 加载 snapshot 跑 python_reference vs golden
  logic_mapping/                            # L3 或 L2 才有
    references/                            # 拷贝的参考源文件（保留原始注释）
      <repo>/<file>.py
      ...
    mapping.md                             # 索引 + 对应关系；含 kind: L3 | L2
    manifest.json                          # 引用文件路径 + 行范围 + sha256 + kind
  docs/
    problem_translate_report.json          # 判定过程 + 前置搜索结果 + 每一级尝试结论
  level_decision.yaml                      # 已更新 oracle_level 和 evidence_paths
```

**人工可修改**（关键闸门）

- review `problem_translate_report.json` 和 `level_decision.yaml`
- 允许 L4↔L3↔L2↔L1 任意方向调整、替换 python\_reference、修正 logic\_mapping.md 内容

### 2.6 Step 4 —— `validate-task-pack` 最终把关 + Deliver 准备

**目的**：确保每个 task\_pack 交付前，参考资料可靠、smoke 可跑、level 判定的证据完整；同时执行 deliver 前的**边界化清理**，把 kernel\_engineer 的工作范围限定在 task\_pack 内。

**执行者**：CLI（现有 validate-task-pack 扩展）。

**执行内容**（对每个 task\_pack）

1. **文件完整性**（现有）
2. **Correctness smoke**（现有）
3. **按** **`level_decision.yaml.oracle_level`** **分级检查**（新增）
   - **L4**：跑 `tests/ut_lv4.py`；必须 pass（这是"UT 跑通"的最终 gate；失败默认降级并生成 report）
   - **L3 / L2**：`logic_mapping/mapping.md` 存在；`logic_mapping/manifest.json` 里引用文件 sha256 与 `logic_mapping/references/` 下拷贝的实际文件一致（防止外部仓库版本升级后引用失效）；`kind` 字段与 `oracle_level` 一致
   - **L1**：只需 `original_source/kernels/<low_level>/` 里四个源码文件（或允许置空的字段）齐全
4. **Deliver 前边界化清理**（新增）
   - **删除** **`level_decision.yaml`** **里的** **`source_index`** **字段**：`source_index` 记录的是原始实现在**框架仓库里**的路径，会让 kernel\_engineer 的注意力跳出 task\_pack。task\_pack 内已通过 `original_source/kernels/<low_level>/` 自包含所有需要的信息
   - 保留 `source_index` 的元数据到 `docs/pre_deliver_snapshot.json`（供 framework\_engineer 后续回接框架时使用）
   - **创建** **`candidate_impl_workspace/`** **空目录**：作为 kernel\_engineer 后续工作的**唯一可写工作区**（可写 .cu / .cpp / .py 源文件、可编译产出 .so 动态库、可放临时脚本）；同时更新 `task.yaml` 里的合同条款
5. **环境探测**（可选）
6. **Benchmark smoke**（可选）

**Task.yaml 合同新增条款**（Deliver 前写入）

```yaml
kernel_engineer_scope:
  writable_paths:
    - candidate_impl.py               # 优化实现主入口
    - candidate_impl_workspace/       # 自定义 kernel 源码 / 编译产物 / 临时脚本
    - docs/iteration_log.md           # 追加自己产出的迭代日志
  read_only_paths:                    # 禁止修改
    - snapshots/
    - snapshot_runtime.py
    - original_source/
    - original_impl.py
    - reference_impl.py
    - correctness_test.py
    - benchmark.py
    - python_reference/               # L4 才有
    - tests/ut_lv4.py                 # L4 才有
    - logic_mapping/                  # L3/L2 才有
    - level_decision.yaml
    - task.yaml
  forbidden_actions:
    - "读取或写入 task_pack 外的任意路径（禁止跳出 sandbox）"
    - "修改 tolerance / timing rules / snapshot 数据"
    - "写入 correctness/benchmark harness"
```

**产物**

```
task_pack/docs/
  task_pack_validation_report.json
  l4_ut_report.json                      # L4 才有（ut_lv4.py 运行结果）
  pre_deliver_snapshot.json              # deliver 前保留的元数据（含删除的 source_index）
task_pack/
  candidate_impl_workspace/              # 空目录 + 一份 README 说明用途
    README.md
task_pack/level_decision.yaml             # 已删除 source_index 字段
task_pack/task.yaml                       # 已含 kernel_engineer_scope 合同
```

### 2.7 Step 5 —— `summarize-workspaces` 汇总 + 排序

**目的**：所有 workspace 里的 task\_pack 都定级 + validate 完成后，按公式对 workspace 排序，输出 kernel\_agent 下一阶段的优化顺序。

**执行者**：skill（含简单 CLI）。

**输入**：所有 workspace 目录（每个含 K 个 task\_pack + 各自的 `level_decision.yaml`）。

**执行内容**

1. 遍历每个 workspace 的所有 task\_pack，读取 `level_decision.yaml` 里的 `oracle_level` 和 `kernel_time_ratio`
2. 计算 `workspace_score = Σ (level_weight × kernel_time_ratio)`，其中 weight L4=4, L3=3, L2=2, L1=1
3. 按 score 降序排列
4. 生成排序报告 + 结构化 ranking

**产物**

```
<output_root>/
  workspace_ranking.json                 # 结构化：workspace → score / rank / 各 task_pack level 分布
  summary_report.md                      # 人可读版排序 + 每个 workspace 的组成分析 + 推荐理由
```

**下游使用**：kernel\_agent 按 `workspace_ranking.json` 从高到低顺序开工。若排名靠前的 workspace 优化收益不理想，降级到备选 workspace。

### 2.8 完整流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│ 用户提供：N 条 service_cmds / workload_cmd / target file+line /       │
│           sglang_repo_root / external_references / third_party_cache  │
├──────────────────────────────────────────────────────────────────────┤
│ Step 0.5: resolve-third-party + locate-kernel-source（两个 skill）    │
│   [Skill]   resolve-third-party：定版本 + clone 三方仓库              │
│             禁止破坏运行环境；产出 third_party_manifest.json          │
│   [Skill]   locate-kernel-source：按形态溯源四层源码（供 KID 复用）   │
├──────────────────────────────────────────────────────────────────────┤
│ Step 1: KID 多路径分解 + 源码定位（CLI）                                │
│   [CLI]     KID 增强：high_level/low_level 双入口 + 多启动命令支持     │
│             产出 (接口名, 形态)，四层定位委托 locate-kernel-source     │
│   [CLI]     import-decomposition：抽取四个源码文件到 workspace         │
│                                    + read_hints.txt                    │
├──────────────────────────────────────────────────────────────────────┤
│ Step 2: Task pack 批量创建（CLI + 少量人工）                            │
│   [CLI]     workspace-to-config：schema → task_pack config 草稿        │
│   [人工]    补齐 forward_boundary 等空白项                              │
│   [CLI]     phase1 主链路（现有 + 支持多 low_level）                   │
│   [CLI]     import-kernel-sources-to-taskpack：拷源码 + 生成            │
│              level_decision.yaml（默认 L1，含 source_index）           │
├──────────────────────────────────────────────────────────────────────┤
│ Step 3: problem_translate（agent 独立上下文）                          │
│   [Agent]   开放式搜索 sgl-tests / third-party tests                   │
│             L4 尝试：python_reference + ut_lv4.py                      │
│             L3 尝试：logic_mapping (kind: L3) —— kernel 等价片段       │
│             L2 尝试：logic_mapping (kind: L2) —— 上层模块级参考         │
│             L1 保底（external_references 为空时直接 L1）               │
│   [CLI]     run-problem-translate：批量编排 agent 调用                  │
│   [人工]    review：升降级 / 替换参考 / 修正 mapping                    │
├──────────────────────────────────────────────────────────────────────┤
│ Step 4: validate-task-pack + deliver 准备（CLI）                       │
│   [CLI]     A. 文件完整性 + correctness smoke                          │
│             B. 按 level 分级 check（L4 跑 ut_lv4.py；L3/L2 校验         │
│                logic_mapping 与 kind；L1 校验四类源码齐全）             │
│             C. Deliver 边界化：删除 level_decision.yaml.source_index   │
│                （备份到 docs/pre_deliver_snapshot.json）                │
│                + 创建 candidate_impl_workspace/                        │
│                + 写入 task.yaml.kernel_engineer_scope 合同             │
├──────────────────────────────────────────────────────────────────────┤
│ Step 5: summarize-workspaces（skill + CLI）                             │
│   [CLI]     遍历 workspace 计算 workspace_score，输出 ranking          │
│             kernel_agent 按此顺序开工                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.9 用户 / Agent / CLI / 人工修改点总览

| 环节       | 用户提供                               | Agent 执行               | 固化 CLI                                                                     | 允许人工修改                                              |
| -------- | ---------------------------------- | ---------------------- | -------------------------------------------------------------------------- | --------------------------------------------------- |
| 前置输入     | N 条启动命令 / workload / target / 仓库路径 | —                      | —                                                                          | 编辑 config                                           |
| Step 0.5 | —                                  | **skill 主路径**（独立上下文）   | —                                                                          | 编辑 `third_party_manifest.json` / `missing_repos.md` |
| Step 1   | —                                  | —                      | KID + `import-decomposition`                                               | 编辑 `decomposition_schema.json` / 补 kernel\_sources  |
| Step 2   | 补 forward\_boundary 等空白项           | —                      | `workspace-to-config` + 现有 phase1 链路 + `import-kernel-sources-to-taskpack` | 编辑 config / task\_pack 空白项                          |
| Step 3   | —                                  | **agent 独立上下文** + 开放搜索 | `run-problem-translate`                                                    | **必须** review：升降级 / 替换 / 修正 mapping                 |
| Step 4   | —                                  | —                      | `validate-task-pack`（扩展）                                                   | 不适用                                                 |
| Step 5   | —                                  | skill 内运行              | `summarize-workspaces`                                                     | 不适用                                                 |

### 2.10 完整 workflow 执行后的最终产物结构

**顶层结构**

```
<output_root>/
├── third_party_manifest.json           # Step 0.5 产出
├── missing_repos.md                    # Step 0.5 产出（若有 FAILED 项）
├── workspace_ranking.json              # Step 5 产出
├── summary_report.md                   # Step 5 产出
│
├── third_party_cache/                  # Step 0.5 clone 落地
│   ├── flashinfer/
│   ├── fla/
│   ├── deep_gemm/
│   └── ...
│
└── workspaces/
    ├── <backend_name_1>/               # 一条 backend 路径 = 一个 workspace
    │   ├── decomposition_schema.json   # Step 1 KID + import-decomposition 产出
    │   ├── kernel_sources/             # Step 1 抽取的原始源码（workspace 级参考）
    │   │   └── <low_level_id>/
    │   │       ├── interface_definition.py
    │   │       ├── py_cpp_binding.cc      # 或空文件（triton/cuteDSL 类）
    │   │       ├── kernel_header.h        # 或空文件
    │   │       ├── kernel_impl.{py,cu,cpp}
    │   │       └── read_hints.txt
    │   └── task_packs/                    # Step 2 生成 → Step 3/4 补齐
    │       ├── <low_level_id_1>_task_pack/    # ← kernel_engineer 消费单元
    │       ├── <low_level_id_2>_task_pack/
    │       └── ...
    │
    ├── <backend_name_2>/
    │   └── ...
    └── ...
```

**单个 task\_pack 完整结构**

```
<low_level_id>_task_pack/
│
├── task.yaml                                # 任务合同：ABI / 目标 / 禁改项 / 命令入口 /
│                                            #           kernel_engineer_scope 合同
├── level_decision.yaml                      # 决策数据（deliver 前删除了 source_index）：
│                                            #   oracle_level / evidence_paths /
│                                            #   kernel_time_ratio / translate_notes
│
├── snapshots/                               # replay 真值（Step 2 capture 产出，不可改）
│   ├── manifest.json                        # selected snapshots 主索引
│   ├── raw_index.json                       # raw capture 索引
│   ├── raw/
│   │   └── group_xxx/sample_xxxx/{meta.json, pre_inputs.pt, post_inputs.pt, outputs.pt}
│   └── selected/
│       └── group_xxx/
│           ├── group_meta.json
│           └── samples/sample_xxxx/{meta.json, pre_inputs.pt, post_inputs.pt, outputs.pt}
├── snapshot_runtime.py                       # 自包含 replay runtime
├── shape_list.json                          # selected snapshots 的 shape 摘要索引
│
├── original_source/                         # 原始 kernel 源码（不可改）
│   ├── manifest.json                        # target 自身元信息
│   └── kernels/<low_level_id>/              # Step 2 import-kernel-sources-to-taskpack 拷贝
│       ├── interface_definition.py
│       ├── py_cpp_binding.cc
│       ├── kernel_header.h
│       ├── kernel_impl.{py,cu,cpp}
│       └── read_hints.txt
│
├── python_reference/                        # === L4 才有 === Step 3 产出
│   ├── ref_impl.py                          # PyTorch/Python 等价实现（可执行）
│   └── manifest.json                        # origin: sgl_internal_test | third_party_test |
│                                            #         agent_generated | human_written
│
├── tests/                                    # === L4 才有 === Step 3 产出
│   └── ut_lv4.py                            # 加载 snapshot 跑 python_reference 全量比对
│
├── logic_mapping/                            # === L3 或 L2 才有 === Step 3 产出
│   ├── references/                          # 拷贝的参考源文件（保留原始注释）
│   │   └── <repo>/<file>.py
│   ├── mapping.md                           # 索引 + 对应关系；含 kind: L3 | L2
│   └── manifest.json                        # 引用文件路径 + 行范围 + sha256 + kind
│
├── original_impl.py                         # linked replay 入口（调用真实框架接口）
├── reference_impl.py                        # correctness reference（snapshot golden / linked）
├── candidate_impl.py                        # ← kernel_engineer 唯一可修改的入口
├── candidate_impl_workspace/                # === Step 4 deliver 前创建 ===
│                                            # ← kernel_engineer 唯一可写工作区
│   └── README.md                            #   （放自定义 kernel 源码 / .so 编译产物 / 临时脚本）
│
├── correctness_test.py                      # 不可改：分级 tolerance 与断言逻辑
├── benchmark.py                             # 不可改：三方计时规则
├── scripts/                                  # 稳定命令入口
│   ├── run_correctness.sh
│   ├── run_benchmark.sh
│   └── run_ncu.sh
│
├── env_manifest.yaml                        # 环境合同（Triton / CuTe DSL / CUDA / NCU 可用性）
├── env_probe/
│   ├── probe_triton.py
│   ├── probe_cutedsl.py
│   ├── probe_cuda_extension.py
│   └── probe_ncu.sh
│
└── docs/                                     # 各阶段过程记录（可审计）
    ├── baseline_result.json                 # Step 2 run-baseline
    ├── baseline_run_report.md
    ├── target_call_probe.jsonl              # Step 2 probe-target-calls
    ├── target_call_probe_report.json
    ├── target_call_probe_report.md
    ├── snapshot_capture_report.json         # Step 2 capture-snapshots
    ├── snapshot_selection_report.json       # Step 2 select-snapshots
    ├── snapshot_selection_report.md
    ├── env_probe_result.json                # Step 2 probe-env
    ├── problem_translate_report.json        # Step 3 skill 产出
    ├── task_pack_validation_report.json     # Step 4 validate 产出
    ├── l4_ut_report.json                    # Step 4 L4 UT 结果（L4 才有）
    └── pre_deliver_snapshot.json            # Step 4 保留的 source_index 元数据
                                              # （供 framework_engineer 回接框架用）
```

**每个文件的产出步骤 + kernel\_engineer 权限**

| 文件 / 目录                                                                                 | 产出步骤                                               | 用途                                                    | kernel\_engineer 权限 |
| --------------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------- | ------------------- |
| **顶层**                                                                                  | <br />                                             | <br />                                                | <br />              |
| `task.yaml`                                                                             | Step 2 → Step 4 追加 `kernel_engineer_scope`         | 任务合同 + 边界合同                                           | 只读                  |
| `level_decision.yaml`                                                                   | Step 2 生成默认 → Step 3 更新 → Step 4 删除 `source_index` | 决策数据                                                  | 只读                  |
| **snapshots 相关**                                                                        | <br />                                             | <br />                                                | <br />              |
| `snapshots/`                                                                            | Step 2 capture + select                            | replay 真值                                             | 只读                  |
| `snapshot_runtime.py`                                                                   | Step 2 generate-harness                            | 自包含 replay runtime                                    | 只读                  |
| `shape_list.json`                                                                       | Step 2 select-snapshots                            | shape 摘要，供快速理解                                        | 只读                  |
| **原始源码**                                                                                | <br />                                             | <br />                                                | <br />              |
| `original_source/kernels/<id>/*`                                                        | Step 2 import-kernel-sources-to-taskpack           | KID 抽取的四类源码                                           | 只读                  |
| **参考资料（按 level 变化）**                                                                    | <br />                                             | <br />                                                | <br />              |
| `python_reference/`                                                                     | Step 3（L4 才有）                                      | 可执行 PyTorch 参考                                        | 只读；可作为实现参考          |
| `tests/ut_lv4.py`                                                                       | Step 3（L4 才有）                                      | L4 golden UT                                          | 只读；validate 阶段自动跑   |
| `logic_mapping/`                                                                        | Step 3（L3 或 L2 才有）                                 | 外部参考源文件 + mapping 索引（含 kind: L3/L2）                   | 只读；理解算法背景           |
| **实现入口**                                                                                | <br />                                             | <br />                                                | <br />              |
| `original_impl.py`                                                                      | Step 2 generate-harness                            | linked 原始实现调用                                         | 只读                  |
| `reference_impl.py`                                                                     | Step 2 generate-harness                            | correctness reference 入口                              | 只读                  |
| `candidate_impl.py`                                                                     | Step 2 generate-harness（占位）                        | **优化实现主入口**                                           | **可改**              |
| `candidate_impl_workspace/`                                                             | Step 4 deliver 前创建                                 | **唯一可写工作区**（自定义 kernel 源码 / .so / 临时脚本）               | **可写**              |
| **评测 harness**                                                                          | <br />                                             | <br />                                                | <br />              |
| `correctness_test.py`                                                                   | Step 2 generate-harness                            | 分级 correctness 断言                                     | 只读                  |
| `benchmark.py`                                                                          | Step 2 generate-harness                            | 三方计时 harness                                          | 只读                  |
| `scripts/*.sh`                                                                          | Step 2 generate-harness                            | 稳定命令入口                                                | 只读                  |
| **环境**                                                                                  | <br />                                             | <br />                                                | <br />              |
| `env_manifest.yaml`                                                                     | Step 2 probe-env                                   | 环境合同                                                  | 只读                  |
| `env_probe/`                                                                            | Step 2 probe-env                                   | 环境探测脚本                                                | 只读                  |
| **过程记录**                                                                                | <br />                                             | <br />                                                | <br />              |
| `docs/baseline_*` / `docs/target_call_probe_*` / `docs/snapshot_*` / `docs/env_probe_*` | Step 2 各阶段                                         | 审计与决策依据                                               | 只读                  |
| `docs/problem_translate_report.json`                                                    | Step 3                                             | Level 判定过程                                            | 只读                  |
| `docs/task_pack_validation_report.json` / `docs/l4_ut_report.json`                      | Step 4                                             | Validate 结果                                           | 只读                  |
| `docs/pre_deliver_snapshot.json`                                                        | Step 4 deliver 前                                   | 保留的 source\_index 元数据（**仅供 framework\_engineer 回接用**） | 不可见                 |
| `docs/iteration_log.md`（若存在）                                                            | kernel\_engineer 自建                                | 迭代日志                                                  | **可写**              |

**level 决定文件存在性的对照表**

| task\_pack 目录 / 文件                          |      L4      |  L3  |      L2      |         L1         |
| ------------------------------------------- | :----------: | :--: | :----------: | :----------------: |
| `snapshots/`                                |       ✓      |   ✓  |       ✓      |          ✓         |
| `original_source/kernels/<id>/`             |       ✓      |   ✓  |       ✓      |          ✓         |
| `candidate_impl_workspace/`                 |       ✓      |   ✓  |       ✓      |          ✓         |
| `python_reference/`                         |       ✓      |   —  |       —      |          —         |
| `tests/ut_lv4.py`                           |       ✓      |   —  |       —      |          —         |
| `logic_mapping/`（含 `kind: L3` 或 `kind: L2`） |       —      |   ✓  |       ✓      |          —         |
| `docs/l4_ut_report.json`                    |       ✓      |   —  |       —      |          —         |
| correctness tolerance                       | 1e-3 \~ 1e-4 | 1e-2 | 1e-2 \~ 5e-2 | snapshot golden 严格 |
| allow\_alt\_math\_paths                     |     false    | true |     true     |        false       |

**kernel\_engineer 消费该 task\_pack 的推荐顺序**

1. 读 `task.yaml`（含 `kernel_engineer_scope` 边界合同） + `level_decision.yaml`（当前 level 与可用参考）
2. 按 level 选择对应参考：
   - **L4**：直接看 `python_reference/ref_impl.py` 理解数学，跑 `tests/ut_lv4.py` 建立信心
   - **L3 / L2**：读 `logic_mapping/mapping.md` 索引 + 拷贝的参考源文件；`kind` 字段区分粒度（L3 = kernel 逻辑等价片段；L2 = 上层模块级别）
   - **L1**：仅 `original_source/kernels/<id>/` 里四份源码
3. 读 `original_source/kernels/<id>/read_hints.txt` 定位每个源码文件重点行数
4. 读 `snapshots/manifest.json` 和 `shape_list.json` 了解 workload 分布
5. 在 `candidate_impl.py` 和 `candidate_impl_workspace/` 内完成优化，**不允许跳出 task\_pack**；用 `scripts/run_correctness.sh` / `run_benchmark.sh` 迭代

***

## 3. 已有实现（现状盘点）

### 3.1 [`tools/kernel_interface_decomposer/`](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer)

对应 Step 1 的基础能力，已完成：

- **服务生命周期管理** —— [runner.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer/runner.py)
- **运行时插桩**（NVTX 打点，forward\_id / stage）—— [runtime\_instrumentation.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer/runtime_instrumentation.py)
- **Trace 归因**（cuda\_correlation\_id + nvtx，stage 分类）—— [trace\_parser.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer/trace_parser.py)
- **热点排序**（per-invocation top-K + duration/share 阈值）—— [trace\_parser.py#L115-L129](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer/trace_parser.py#L115-L129)
- **源码定位**（wrapper API + file/line + category）—— [source\_resolver.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer/source_resolver.py)
- **结构化输出** `decomposition.schema.json` —— [runner.py#L149-L176](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer/runner.py#L149-L176)

### 3.2 [`framework_engineer/`](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/framework_engineer)

对应 Step 2 的完整能力，已完成：

- **CLI 主链路**：`validate-config` / `scaffold-task-pack` / `run-baseline` / `resolve-interface` / `probe-target-calls` / `capture-snapshots` / `select-snapshots` / `generate-harness` / `probe-env` / `validate-task-pack` / `run-phase1`
- **Snapshot 数据模型**（[snapshot/](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/framework_engineer/snapshot)）
- **Multi-target 支持** + `multi_target_report.md`
- **Env probe**（Triton / CuTe DSL / CUDA extension / NCU）

### 3.3 现状小结

| 步骤           | 覆盖状况                                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------------------- |
| Step 0.5     | **完全缺失**                                                                                                       |
| Step 1       | KID 已有单路径能力；缺多路径入口、缺 low\_level 入口、缺四类源码严格产出、缺 `import-decomposition`                                          |
| Step 2       | 主链路已完备；缺多 low\_level 批量 config、`workspace-to-config`、`import-kernel-sources-to-taskpack`、`level_decision.yaml` |
| Step 3       | 完全缺失（agent、skill、CLI 编排、ut\_lv4.py 生成、logic\_mapping 生成）                                                       |
| Step 4       | 部分覆盖；缺按 level 分级检查、缺 L4 UT 跑通                                                                                  |
| Step 5       | 完全缺失                                                                                                           |
| 四级 oracle 概念 | 完全缺失（level\_decision.yaml schema、tolerance 派生规则）                                                               |

***

## 4. 缺失项梳理（简要）

详细开发内容见 **Section 8 开发需求细节**。此处仅列缺失分类：

- **Config / Schema**：多启动命令 / 多仓库路径 / kind / oracle\_level（默认 1）等字段
- **CLI**：`import-decomposition` / `workspace-to-config` / `import-kernel-sources-to-taskpack` / `run-problem-translate` / `summarize-workspaces` / `validate-task-pack` 扩展
- **Skill**：`resolve-third-party.md` / `locate-kernel-source.md` / `problem_translate.md` / `summarize-workspaces.md`
- **Prompt**：`framework_engineer.md` 更新
- **文档**：`framework_engineer/README.md` + `task_pack_README.md` 扩展
- **产物 schema**：`third_party_manifest.json` / `decomposition_schema.json` 增强 / `level_decision.yaml` / `problem_translate_report.json` / `l4_ut_report.json` / `workspace_ranking.json`

***

## 5. TODO 清单（简版，按 workflow 步骤组织）

> 每一步都要**先开发再单独测试**。全部完成后进入最终整合阶段。详细开发内容见 Section 8。

### Step 0.5: resolve-third-party + locate-kernel-source

- [ ] 固化 9 种 kernel 形态分类 + 共享 helper 包 `third_party_solver/`
- [ ] 开发 skill `resolve-third-party.md`：固定 universe + 版本二分（importlib / sgl-kernel CMake pin）+ clone
- [ ] 开发 skill `locate-kernel-source.md` + `source_locator.py`：按形态分派四层溯源（机制①符号溯源 / 机制②JIT sources）
- [ ] 测试：GDN 场景人工验证 clone 结果 + 四层定位（F2/F3/F7/F8 各一个 case）

### Step 1: KID 多路径分解

- [ ] 增强 KID：high\_level/low\_level 双入口 + 多启动命令 + 形态标注（`_infer_category` 扩到 F0–F8）
- [ ] KID 四层定位委托 `locate-kernel-source`（复用 helper，不重复实现溯源）
- [ ] 开发 `import-decomposition` CLI
- [ ] 测试：high\_level 和 low\_level 各一个 case，验证 workspace 结构 + read\_hints.txt

### Step 2: task\_pack 批量创建

- [ ] 修改 phase1 config 支持多 low\_level
- [ ] 开发 `workspace-to-config` CLI
- [ ] 开发 `import-kernel-sources-to-taskpack` CLI
- [ ] 扩展现有 `validate-task-pack` 支持 L1 检查
- [ ] 测试：workspace → K 个 task\_pack，validate 通过（默认全 L1）

### Step 3: problem\_translate

- [ ] 开发 skill `problem_translate.md`
- [ ] 开发 CLI `run-problem-translate`（批量编排）
- [ ] 测试：人工找几个已知 level 的 low\_level，验证 agent 判定正确 + 证据文件正确

### Step 4: validate-task-pack 分级扩展

- [ ] 扩展 validate 支持 L2/L3/L4 分级检查（L4 跑 ut\_lv4.py）
- [ ] 测试：Step 3 产出的 task\_pack 全部通过 validate

### Step 5: summarize-workspaces

- [ ] 开发 skill / CLI `summarize-workspaces`
- [ ] 测试：跑通完整 workflow，验证 workspace\_ranking 排序结果符合预期

### 最终整合阶段

- [ ] 合并各步骤 config 文件到统一配置（现在是分散的：Step 0.5 / Step 1 / Step 2 各有独立 config）
- [ ] 完善步骤间输入输出的强类型约束（schema JSON schema 校验）
- [ ] 更新 prompt `framework_engineer.md`
- [ ] 更新 README + `task_pack_README.md`
- [ ] 端到端 smoke：GDN 完整跑通 Step 0.5 → Step 5，产出 L2/L3/L4 各一个 case

***

## 6. 跨阶段共同约定

| 约定                        | 内容                                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 单一流程                      | 所有 target 走同一套 low\_level 优化主链路，无独立 Mode 2 分支                                                                                                                                  |
| High\_level 触发            | 用户 config 里显式列 N 条启动命令                                                                                                                                                         |
| 多路径生成                     | 所有 workspace（每条 backend 一个）**一次性都生成**，最后统一排序                                                                                                                                   |
| Route 排序                  | Step 5 汇总，用真实 level 判定结果计算 `workspace_score`                                                                                                                                   |
| Third-party 仓库            | Step 0.5 前置准备（`resolve-third-party` 定版本+clone / `locate-kernel-source` 按形态溯源）；无白名单限制；不允许影响运行环境 |
| Oracle 判定                 | skill 认证驱动（能凑齐资料就是那一级）；agent 独立上下文；判定后**必须**人工 review                                                                                                                          |
| L4 python\_reference      | 优先复用 sgl 内部 tests/benchmark 或 third-party tests；agent 可 draft；`ut_lv4.py` 是 L4 的证据；validate 阶段跑 UT 兜底                                                                          |
| L3 / L2 统一形式              | 二者共用 `logic_mapping/{references/, mapping.md, manifest.json}` 目录；`kind` 字段区分粒度（L3=kernel 逻辑等价片段；L2=上层模块级别）                                                                     |
| 外部参考仓库                    | 由用户 `external_references` 开放配置提供；未提供时且无法定义成L4时 skill 直接判 L1（不主动猜其他仓库）                                                                                                          |
| Deliver 边界化               | Step 4 validate 后：删除 `level_decision.yaml` 的 `source_index`（元数据备份到 `docs/pre_deliver_snapshot.json`）+ 创建 `candidate_impl_workspace/` + 写入 `task.yaml.kernel_engineer_scope` 合同 |
| kernel\_engineer 沙箱       | 只允许写 `candidate_impl.py` / `candidate_impl_workspace/` / `docs/iteration_log.md`；禁止跳出 task\_pack                                                                               |
| Task\_pack self-contained | 所有参考资料（kernel\_sources / logic\_mapping / python\_reference）都**拷贝到 task\_pack**，不动态引用                                                                                          |
| Task\_pack 决策数据文件         | `level_decision.yaml`（顶层，与 task.yaml 平级）记录 oracle\_level / evidence\_paths / kernel\_time\_ratio / translate\_notes（deliver 后不含 source\_index）                                 |

***

## 7. 方案演变简述（Mode 2 → 统一四级 oracle）

### 7.1 演变阶段

**阶段 A：三级 oracle + Mode 2 二分**（初版）

- Scenario 1 模块级输入分 Mode 1（分解优化）和 Mode 2（重写模块）
- Scenario 2 low\_level 输入分 L1/L2/L3
- Mode 2 里需要手写 ResourceAdapter（框架资源翻译层）、reference\_module（transformers forward 改造版）、静态推理脚本兜底路径 (Path A / Path B)

**阶段 B：Mode 2 简化为"路径切换 + transformers 参考"**（中期）

- 意识到 sglang 的多后端本身就已经把粒度对齐问题处理好了，不需要重塑接口
- adapter 相关工作全部砍掉，Mode 2 变成"用户挑一条 sglang 后端路径 + transformers 对应 forward 作为 oracle"

**阶段 C：完全取消 Mode 2，改为多路径评分 + 四级 oracle**（最终）

- 阶段 B 的路径切换本质上和"多路径分解 + 排序选优"是同一个动作，只是决策粒度不同
- 索性把它下沉为 high\_level target 的通用前置流程，high\_level 和 low\_level 共享同一套 low\_level 优化主链路
- 三级 oracle 扩为四级，新增 L3（逻辑等价映射 markdown）

**并行增强**：Step 0.5 拉齐第三方仓库，让 KID 能精确定位源码 + problem\_translate 能在 tests 目录里"抄现成"作为 L4 的免费金矿。

### 7.2 放弃 Mode 2 的具体好处

| 维度                          | 保留 Mode 2                          | 取消 Mode 2                                           |
| --------------------------- | ---------------------------------- | --------------------------------------------------- |
| **工程复杂度**                   | 双分支：Mode 1 + Mode 2                | 单一分支                                                |
| **需要手写的抽象**                 | ResourceAdapter × framework × 资源类型 | 无                                                   |
| **UT 语义**                   | 三种模式：Path A / Path B / Mode 1      | 统一：candidate vs snapshot + 参考资料按 level 决定 tolerance |
| **切换到 flashinfer/fla 的收益**  | 需 adapter 包装                       | 天然支持：新增一条 backend 命令即可                              |
| **L4 python\_reference 来源** | agent 独立 draft                     | 优先从 sgl-tests / third-party tests **抄现成**           |
| **对 DSA / V4 类模型**          | Mode 2 硬做陷入 adapter 每模型一份的泥潭       | 只走单一主链路                                             |
| **给 kernel\_engineer 的信息量** | 完整 transformers module + adapter   | 分级：L4 可执行 / L3 逻辑映射 / L2 模块代码 / L1 原始实现             |
| **回接框架的成本**                 | 步骤多、不确定性大                          | 走原有 sglang 后端 ABI，回接零成本                             |
| **总体 ROI**                  | 少数场景收益，大量工程投入                      | 明确 ROI：多路径 = 更多描述方式 + 通用库天然带 UT                     |

### 7.3 最终定型的一句话

**"信任 sglang / vLLM 作为专业推理框架已经把计算划分做得接近最优；工具的核心价值不是重塑抽象，而是帮 kernel\_engineer 在多条已有路径里选一条最好理解的，并把参考资料备齐。"**

***

## 8. 开发需求细节

> 本节按 workflow 顺序，逐项描述每个待开发组件的：位置、类型、输入、输出、处理逻辑、约束、依赖、测试方式。其他 agent 看完本节结合当前代码即可开工。

### 8.1 Step 0.5 — 第三方仓库准备 + kernel 源码定位（两个 skill）

Step 0.5 拆成**两个独立 skill**，共享一个 helper 包 `kernel_agent/framework_engineer/third_party_solver/`：

- **8.1.1 `resolve-third-party`**：确定依赖仓库的正确版本 + clone 到本地缓存，产出 `third_party_manifest.json`（`name → local_path`）。
- **8.1.2 `locate-kernel-source`**：给定「接口名 + kernel 形态」，按形态规则定位 kernel 的四层源码（接口定义 / kernel 实现 / py↔cpp binding / 头文件）。

**核心设计原则（决定 KID 边界）**：把「溯源智能」下沉到 8.1.2 skill 与共享 helper。KID（Step 1）只负责产出 `(接口名, 形态标签)`，把四层定位委托给 `locate-kernel-source`，**不在 KID 里重复实现溯源逻辑**。deterministic 部分做成 helper 供 KID 的 `source_resolver.py` 直接 import；歧义部分由 skill 的 agent 上下文兜底。

#### 8.1.0 kernel 形态分类（两个 skill 与 KID 共用的 key）

一切规则都挂在「形态」上。固化 9 种形态：

| ID | 形态 | 运行产物 | 源码物理位置 | 要 clone? |
| -- | -- | -- | -- | -- |
| **F0** | pytorch / 成熟库 API（aten / cuBLAS / cuDNN） | vendor lib | 不找（API 即定义） | 否 |
| **F1** | sglang 自带 triton / DSL | 运行时 JIT | sglang python 树 | 否 |
| **F2** | sgl-kernel 内实现 | 预编译 `.so` | `sgl-kernel/csrc/` | 否（在 sglang 仓库） |
| **F3** | sgl-kernel `FetchContent` 三方编入 | 预编译 `.so` | **对应三方 clone** | **是** |
| **F4** | sglang-owned JIT（`jit_kernel/`） | 运行时 nvcc | sglang `jit_kernel/{csrc,include,data}` | 否 |
| **F5** | 三方 C++/cuda（AOT 编译） | 预编译 `.so` | 三方 clone | 是 |
| **F6** | 三方 triton / cuteDSL | 运行时 JIT | 装包 = clone（同版本逐字相同） | 可选 |
| **F7** | 三方 C++ JIT（flashinfer / deep\_gemm） | 运行时 nvcc | 装包 `data/csrc` 或 clone `csrc` | 可选 |
| **F8** | 下载预编译 cubin（trtllm / nv artifact） | 下载 `.cubin` | **无源码** | 否 → FAILED |

> 判据：溯源难度只取决于「源码在不在磁盘上」，跟是不是 JIT 无关。**JIT 不生成新源码**，它只是 nvcc/DSL 编译一组已提交在仓库里的文件（见 8.1.2 机制②）。真正无源可溯的只有 F8。

***

#### 8.1.1 `resolve-third-party`

**类型**：**CLI 为主**（`framework_engineer/third_party_solver/`），配套 skill 文档做运维手册。

> 本步已被完全确定性化——「明确仓库 → 找版本线索 → 版本映射成 ref → clone」四步全是查表/读取/规则/执行，无需 agent 编排。正常路径就是一条 CLI。agent 仅在规则未覆盖的新情况（新库、tag 命名变化、`version_mismatch`、clone 失败归因）时作为兜底，去把新规则补进 registry / 映射。

**位置**

- CLI + helper 包：`framework_engineer/third_party_solver/`（`registry.py` / `cmake_pins.py` / `version_resolver.py` / `cloner.py` / `manifest.py` / `cli.py` / `config.py`）
- skill 文档（运维手册）：`framework_engineer/skills/resolve_third_party.md`
- 样例配置：`framework_engineer/configs/resolve_third_party.example.py`

**入口**

```bash
python -m framework_engineer.third_party_solver.cli resolve --config <cfg> [--dry-run] [--https-proxy P]
```

**输入**（配置文件 `.py`/json/yaml）

- `service_cmds`（必填）：N 条 `{backend_name, cmd}`
- `sglang_repo_root`（必填，含 `sgl-kernel/` 源码树）
- `third_party_cache`（必填，按 `(name, version)` 分目录）
- `output_root`（必填）
- `workload_cmds` / `explicit_paths` / `extra_env` / `https_proxy`（可选）

**输出**

- `<output_root>/third_party_manifest.json`
- `<output_root>/missing_repos.md`（若有 FAILED 项）
- `third_party_cache/<name>/<version>/` 下 clone 的完整 git 源码树

***

##### 步骤 1：明确所有需要的仓库（固定 universe）

依赖集合有限，**做固定注册表（`registry.py` 的 `UNIVERSE`）而非现场发现**。当前 universe = 9 个源码库 + 1 个 F8：

```
Bucket A(pip):  flashinfer  deep_gemm  flash_attn_4
Bucket B(.so):  flash_attn(sgl-attn)  flash_mla(FlashMLA)  cutlass  mscclpp  flashinfer_embedded
F8(无源):       flashinfer_cubin
```

每条登记 `{ name, archetype(F*), version_source, dist_name, cmake_target, url, url_kind, on_default_path, backend_flags, ref_template }`。三条要点：

1. **不按 flag 裁剪 clone**：`backend_flags` 只用来给 manifest 注解 `triggered_by`；一律对整个 source-bearing universe 定版本 + clone。理由：`on_default_path` 类库（flashinfer/deep_gemm/cutlass/mscclpp）即使启动命令不显式指明也会被用到,按 flag 裁剪会漏依赖。宁多勿漏（整套实测 ~350M，相对模型可忽略）。
2. **同名多版本 → 缓存 key 是 `(name, version)`**：`flashinfer` 同时以 F7（运行时 pip `0.6.12`）和 `flashinfer_embedded`（F3，编进 sgl-kernel 的 `norm.cu`，pin `@bc29697`）存在，是两个不同 commit，用不同 name 区分,互不覆盖。
3. **故意不在 universe 的**：`fla` / `causal_conv1d`（sglang/sgl-kernel 自带,F1/F2,非外部仓库）、`triton_kernels`（sgl-kernel CMake 只 `install(DIRECTORY .../python/triton_kernels/)` 拷纯 python，非编译进 `.so`,随包已装）。它们由 `locate-kernel-source` 就地定位,本 skill 不 clone。

##### 步骤 2：找版本线索（严格二分：install 好的 vs sgl-kernel 编译配置）

版本线索**必须按 archetype 二分**，因为它们在不同地方：

**Bucket A —— 独立 pip 包（运行时 import）→ 线索在 `importlib.metadata`**
适用 `flashinfer` / `deep_gemm` / `flash_attn_4`。
```python
importlib.metadata.version(dist_name)   # 以“装好的”为准，不读 pyproject 声明（可能是范围）
```
> `dist_name` ≠ import 名（已核实）：`flashinfer`→`flashinfer_python`、`deep_gemm`→`sgl-deep-gemm`、`flash_attn_4`→`flash-attn-4`（import 名 `flash_attn`）。反查用 `importlib.metadata.packages_distributions()` 或 `pip show`。

**Bucket B —— 编进 sgl-kernel `.so`（无独立 pip 版本）→ 线索在 sgl-kernel 构建配置**
适用 `flash_attn` / `flash_mla` / `cutlass` / `mscclpp` / `flashinfer_embedded`。完整 4 步链：

```
1. importlib.metadata.version("sglang-kernel")   -> 0.4.3       # 运行时真实版本（锚点，分发名 sglang-kernel）
2. 用 0.4.3 对齐 sgl-kernel 源码树                                # 校验 sgl-kernel/python/sgl_kernel/version.py == 0.4.3
                                                               #   不一致 → manifest 标 version_mismatch=true，不自动 checkout
3. 读该源码树 FetchContent pin（cmake_pins.py 解析）：
     CMakeLists.txt:        sgl-attn@bcf72cc / flashinfer@bc29697 / cutlass@57e3cfb / mscclpp@51eca89
     cmake/flashmla.cmake:  FlashMLA@df022eb
4. 按 pin commit clone
```

> ⚠️ **断层**：装好的 `sgl_kernel` wheel **只有 `.so`，不含 CMakeLists**，`version.py` 只给版本号、不含 fetch commit。所以 pin 必须从**对应版本的源码树**读——这就是 config 要 `sglang_repo_root` 的原因。FetchContent 的 URL 直接写明 `sgl-project/...` vs `NVIDIA/...`，即「sgl fork 还是官方」不用猜。

##### 步骤 3：版本线索 → clone ref 的映射规则

"版本线索"有三种形态，映射成 git `ref` 的规则不同（由 registry 的 `ref_template` + `_format_ref` 承载）：

| 线索形态 | 来源 | 映射成 ref | 例子 |
| --- | --- | --- | --- |
| **commit hash** | Bucket B 的 CMake pin | 直接用（不加工），cache 目录名取前 12 位 | `bcf72ccc6816...` |
| **规范 release tag** | Bucket A，pip 版本 == tag | 默认模板 `v{version}` | `0.1.2`→`v0.1.2`（deep_gemm 等大多数） |
| **非规范 tag（需变换）** | Bucket A，pip 版本 ≠ tag 拼法 | 自定义 `ref_template` + 占位符 | FA4：pip `4.0.0b17`→tag `fa4-v4.0.0.beta17` |

`ref_template` 可用占位符：`{version}`（原始 pip 版本）、`{pep440_beta}`（`bN`→`.betaN`，如 `4.0.0b17`→`4.0.0.beta17`）。FA4 写成 `"fa4-v{pep440_beta}"` 自动得 `fa4-v4.0.0.beta17`（升级 beta18 自动跟随）。`None` = clone 默认分支（丢版本精度，最后手段）。

> **加新库/新规则**：先 `git ls-remote --tags <url>` **实际查** tag 命名（别凭"beta 大概没 tag"猜——deep_gemm/FA4 都是查了才发现有 tag），再定 `ref_template`；现有占位符表达不了就在 `_format_ref` 加一个。

##### 统一 clone（Option A）：install vs 源码差异

即使某库已 pip 装好，也**一律按正确版本 clone git 源码**，不复用 site-packages。因为 **wheel ≠ git 源码树**：

| 内容 | git 仓库(clone) | pip wheel(installed) |
| --- | --- | --- |
| DSL/triton kernel（`.py`） | 有 | 有 |
| JIT 用 C++ 源 `.cu/.cuh` | 顶层 `csrc/` | 挪到包内 `<pkg>/data/csrc/`（**路径不同**） |
| tests / benchmarks | 顶层有 | **通常剥掉** |
| Bucket B 库的源 | 完整 | 不作为独立包存在；同名多是无关库（`cutlass`→`nvidia_cutlass_dsl`、`flash_attn`→FA4） |

后果：JIT 溯源路径会错、拿不到 L4 要抄的 tests、同名撞库。故统一 clone，下游只处理一种 git 布局。仅 P1（`explicit_paths`）/ P2（sgl-kernel 内嵌 git 树）跳过 clone。JIT kernel 溯源统一按 clone 的顶层 `csrc/` 布局（`~/.cache/.../cached_ops` 只有编译产物无源）——详见 8.1.2 机制②。

##### 产物 `third_party_manifest.json`（每条记录）

```json
{
  "name": "flash_attn",
  "archetype": "F3",
  "version": "bcf72ccc6816",
  "version_source": "importlib | sgl_kernel_cmake_pin",
  "clone_source": "official | sgl_fork | embedded | explicit",
  "resolution": "cloned | embedded | explicit | none",
  "local_path": "third_party_cache/flash_attn/bcf72ccc6816",
  "url": "https://github.com/sgl-project/sgl-attn",
  "ref": "bcf72ccc6816b36a5fae2c5a3c027604629785e0",
  "triggered_by": ["flashinfer"],
  "on_default_path": false,
  "version_mismatch": false,
  "status": "ok | clone_failed | failed",
  "clone_command": "https_proxy=... git clone ... && ... checkout <ref>",
  "evidence": "可选"
}
```

`status` 三态：`ok`（`local_path` 填本地路径）/ `clone_failed`（`local_path` 空 + 可复跑 `clone_command`）/ `failed`（无源 F8 或定位不到）。`name`+`version`+`archetype`+`status` 必填。

**helper 模块**：`registry.py`（universe）/ `cmake_pins.py`（Bucket B pin 解析）/ `version_resolver.py`（二分 + `_format_ref`）/ `cloner.py`（P1/P2/clone，失败只记录）/ `manifest.py` / `config.py`。

**约束**

- 严格禁止修改 sglang 源码、重编 sgl-kernel、装新包；clone 仅作参考，不链接进环境
- clone 失败**只记录不解决**（不重试/不换镜像）；F8 无源直接 FAILED
- clone 用 argv 列表执行（非 shell 拼接），代理经环境变量注入

**依赖**：无（独立第一步）


**测试**（Qwen3.5 GDN，三条 backend：triton / flashinfer / cutedsl；已在真实 H 卡容器验证）

1. `flashinfer` / `deep_gemm`：Bucket A 定版本（`0.6.12` / `0.1.2`），clone 对应 tag 成功
2. `flash_attn(sgl-attn)` / `flash_mla` / `cutlass` / `mscclpp` / `flashinfer_embedded`：Bucket B，commit == CMake pin，`clone_source` 正确区分 `sgl_fork`/`official`
3. `deep_gemm`（tag `v0.1.2`）/ `flash_attn_4`（pip `4.0.0b17` → tag `fa4-v4.0.0.beta17`）：版本→ref 映射正确（均 `git ls-remote` 验证 tag 存在）
4. `flashinfer`（F7 pip）与 `flashinfer_embedded`（F3 pin）两条 `(name, version)` 互不覆盖
5. 每个 `status:ok` 的 `local_path` 是完整 git 源码树；F8（`flashinfer_cubin`）进 `missing_repos.md`
6. 单测：`framework_engineer/tests/test_third_party_solver.py`（cmake_pins / flags 不裁剪 / version 二分 / ref 映射 / cloner 失败只记录 / manifest 三态）

***

#### 8.1.2 `locate-kernel-source` skill

> **2026-07-19 收敛说明：本小节以下旧 F0–F8 分派、`source_locator.py`、Layer 1/2、
> `needs_agent/source` 和 KID `runtime_event.implementation` 设计均已废弃，仅保留为历史调研。**
> 当前实现是 `framework_engineer/skills/source_locate.md` +
> `prompts/start_source_locate.md`：用户只提供 `source-locate-agent-config/v1`；入口 Prompt
> 自主编排公开 `locate`、四层语义判断、finalize 和公开 `extract`。私有
> `agent_helper prepare-run/inspect-target/search/finalize/evaluate/validate-run` 负责配置预检、
> 候选搜索和机械校验。当前 contract 与实现以
> `KID_and_locate_source_desgin_v2.md` §4 为准。

**类型**：Skill（agent 上下文兜底歧义）+ 共享 deterministic helper（供 KID 直接 import）

**位置**

- 主文件：`framework_engineer/skills/locate_kernel_source.md`
- helper：`framework_engineer/third_party_solver/source_locator.py`

**输入**

- `interface`：接口名（如 `torch.ops.sgl_kernel.fwd` / `chunk_gated_delta_rule` / `get_gdn_prefill_module().gdn_prefill`）
- `archetype`：形态标签（F0–F8，由 KID 运行时抓取时打标）
- `third_party_manifest.json`（来自 8.1.1，提供各仓库 `local_path`）
- `sglang_repo_root`

**输出**：该接口的四层 `source_locations`（每层 `{file, line_start, line_end}` 或 `null`）

##### 四层信息（KID 需要的定位目标）

- **a. `interface_definition`**：原始接口定义（python），含 kernel launch 语句
- **b. `kernel_impl`**：原始 kernel 实现（`.cu` / `.cpp` / `.py`）
- **c. `py_cpp_binding`**：py↔cpp 绑定文件（主要对 sgl-kernel）
- **d. `kernel_header`**：头文件 `.h/.cuh`（主要对 sgl-kernel 和三方 raw cuda）

##### 点 3：按形态分派表（`—` = 该层不适用，允许 null）

| 形态 | a. 接口定义(py) | b. kernel 实现 | c. py↔cpp binding | d. 头文件 | 机制 |
| -- | -- | -- | -- | -- | -- |
| F0 | sglang 调用点 | —（aten/cuBLAS） | — | — | 停在 API |
| F1 | sglang wrapper | sglang 内 `@triton.jit` fn | — | — | grep `def <name>` |
| F2 | `sgl_kernel/*.py` | `sgl-kernel/csrc/*.cu` | `*_extension.cc` 的 `m.impl` | `csrc/**/*.h(.cuh)` | 机制① |
| F3 | `sgl_kernel/*.py` | **clone 内** impl | `*_extension.cc` | `include/*ops.h` | 机制①→跨仓 |
| F4 | sglang `jit_kernel/*.py` | `sources[]` 里 `.cu` | `sources[]` 里 `*_jit_binding.cu` | `sources[]` / include 的 `.cuh` | 机制② |
| F5 | 三方 `*.py` | clone `csrc/*.cu` | 三方 pybind | clone `*.h` | pybind 溯源 |
| F6 | 三方 `*.py` | 同文件 DSL fn（`.py`） | — | — | grep `def`/`class` |
| F7 | 三方 `*.py`（如 `gdn_prefill.py`） | `sources[]` `.cu` | `sources[]` binding | `sources[]` `.cuh` | 机制② |
| F8 | 三方 `*.py` | **FAILED 无源** | — | — | deny-list |

**机制① —— sgl-kernel 符号溯源（F2/F3，复用 KID 现有 `_sgl_kernel_registry`）**

```
接口 torch.ops.sgl_kernel.<op>
 → csrc/*_extension.cc 找 m.def("<op>",...) + m.impl("<op>", &<symbol>)      # 层 c
 → grep <symbol> 定义：
     命中 sgl-kernel/csrc/*.cu                     → F2，就地（层 b；层 d 同目录 .h）
     命中被 fetch 的仓库（对照 CMake SOURCES 路径） → F3，去 manifest 里该库 clone 定位（层 b/d）
```

实例：`torch.ops.sgl_kernel.fwd` → `flash_extension.cc` 的 `m.impl("fwd", &mha_fwd)`（层 c）→ `mha_fwd` 在 clone 的 sgl-attn `hopper/flash_api.cpp`（层 b）→ `sgl_flash_kernel_ops.h`（层 d）；wrapper `flash_attn.py`（层 a）。

**机制② —— JIT 源码溯源（F4/F7，回答「运行时编的代码在哪」）**

```
接口 get_xxx_module().<op>()
 → 找 gen_xxx_module() 定义 → 读 gen_jit_spec(sources=[...])
 → sources 列表即层 b/c/d（binding = *_jit_binding.cu，头 = .cuh）
 → 文件锚点看路径前缀：FLASHINFER_CSRC_DIR = <pkg>/data/csrc → 该包 / 对应 clone
 → 永远忽略 ~/.cache/flashinfer/cached_ops 里的编译产物（无新源码）
```

注意跨包：sglang 的 JIT `sources` 可能同时引用 sglang overlay + flashinfer `data/csrc` + 下载 artifact（后者 → F8）。

##### 两层溯源架构

溯源分两层，第一层确定性 helper 尽力而为，第二层 agent 兜底：

- **第一层：确定性 helper**（`source_locator.py`）。对每个接口按 archetype 分派，能定位就定位；**即便定位不到或有歧义，也先跑一遍**，把每层结果结构化标出（`resolved` / `not_applicable` / `ambiguous` / `not_found`）。歧义给多个候选，失败给该库在 manifest 里的原始仓库根路径（`repo_hint`），不写失败原因——helper 是固定逻辑，做不好灵活搜索。
- **第二层：agent 兜底**。读第一层结果，只对 `ambiguous` / `not_found` 的层，拿 `repo_hint` 的仓库根**自己主动去找**；找到补齐，再找不到 → 该层标 `missed`。

**信息源优先级**：第一层 helper **优先消费 KID 运行时 event**，event 缺失才 fallback 到静态解析（机制①/②）。KID 的 [runtime_instrumentation.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py) 已 patch `load_jit`，对每个接口记录：

```python
{
  "api": "...", "file": "...", "line": ...,          # wrapper 源文件+行 → 层 a 几乎白送
  "category": "...", "stage": ..., "forward_mode": ...,
  "implementation": {                                 # 仅 JIT / sgl_kernel 有
    "source_files": ["/abs/.../csrc/xxx.cu", ...],    # JIT 已解析源路径 → F4/F7 层 b/c/d 白送
    "symbols": ["mha_fwd"], "export_name": "fwd", "compile_flags": {...}
  }
}
```

> 边界：`source_files` 对 **F3（AOT 编进 `.so`）不会有**（不走 `load_jit`），F3 层 b 仍需 helper 静态「符号 grep 到 clone」。

##### helper 接口契约

```python
def locate_kernel_source(
    interface: str, archetype: str, *,
    runtime_event: dict | None,   # KID event（可空）
    manifest: dict,               # name -> {local_path, ...}
    sglang_repo_root: Path,
) -> LayerResolution

@dataclass
class LayerResult:
    status: Literal["resolved", "not_applicable", "ambiguous", "not_found"]
    hits: list[LayerHit]          # resolved=1 条；ambiguous=多候选；not_found=[]
    repo_hint: str | None         # 失败时给 manifest 里该库原始仓库根，交 agent 主动找

@dataclass
class LayerResolution:
    interface: str; archetype: str
    source: Literal["runtime_event", "static", "mixed"]
    layers: dict[str, LayerResult]  # a/b/c/d 四层各一
    needs_agent: bool               # 任一必填层 != resolved/not_applicable → True
```

单入口，内部按 `archetype` 分派。每层独立四状态（不是整体 bool）：`not_applicable` 是形态决定的合法 null，不算失败。

##### determinism 分档（helper 能做到哪一步）

- **档 1 完全确定**（helper 命中，`needs_agent=False`）：
  - **F0**：识别 aten/cuBLAS 前缀 → 各层 `not_applicable`，层 a=调用点
  - **F4/F7**：读 `runtime_event.source_files` 按后缀分层（`.cu/.cpp`→b、`*_jit_binding.cu`→c、`.cuh/.h`→d），层 a=event `file/line`
  - **F2**：复用 `_sgl_kernel_registry` → 层 c；grep symbol 在 `sgl-kernel/csrc/` → 层 b/d
- **档 2 尝试 + 常见歧义**（层 a 稳，层 b 可能 `ambiguous`/`not_found` 带 `repo_hint`）：
  - **F3**：层 a/c 同 F2；symbol grep 命中 fetch clone，多实例化 → `ambiguous`+多 hits，或 `not_found`+repo_hint
  - **F1 / F6**：`def/class <name>` 唯一即 resolved，多处/re-export → repo_hint
  - **F5**：pybind 有注册即 resolved，否则 repo_hint
- **档 3 helper 无能**（直接标记）：
  - **F8**：`not_found` 无 repo_hint（无源）；KID agent 已判为 F8 → 直接 `missed`，不复确认
  - **跨包混合 sources**：逐层分别定状态，F8 那层 `missed`，其余正常
  - **event 与静态锚点双缺失**：`not_found` + repo_hint（若有），交 agent

##### 约束

- `interface_definition` / `kernel_impl` 不允许最终为 null；`py_cpp_binding` / `kernel_header` 仅 F1/F4/F6/F7 的 DSL/triton 情形可 `not_applicable`
- 两层都失败的必填层最终标 `missed`，报出接口名 + 形态 + repo_hint，交人工

**依赖**：`resolve-third-party`（需要 manifest 里的 clone 路径）

**测试**

1. F3：`sgl_kernel.fwd` 四层齐全，层 b 落在 sgl-attn clone
2. F7：`chunk_gated_delta_rule`（v0.6.12，C++ JIT）→ 层 b 命中 flashinfer `csrc/*.cu`；换新版（cuteDSL）→ 层 b 命中 `.py`、层 c/d 为 null
3. F2：`causal_conv1d` 层 b/d 在 `sgl-kernel/csrc/mamba/`，无需跨仓
4. F8：下载 cubin 的 op 正确报 FAILED

### 8.2 Step 1 — KID 增强（`import-decomposition` 已迁出）

> **⚠️ 本节已被 [KID_and_locate_source_desgin_v2.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/kernel_agent_kadai/KID_and_locate_source_desgin_v2.md) 收敛，以后者为准。** 两处关键修订：
> 1. 旧 `import-decomposition` 已由公开 `extract` CLI 取代。extract 是 source_locate Agent
>    之后的外层阶段，不再称为 Layer 3；§8.3.3 的 task_pack 组装不受影响。
> 2. **KID 入口统一为 high_level_target**（取消 `target_kind` 双模式），且 KID 只产 `(interface, archetype, runtime_event)`、**不再自己做四层源码定位**（委托 locate）。下方保留的旧描述（`--target-kind`、KID 内做跨仓/JIT 溯源）已过时，仅存档参考。

**类型**：现有 `framework_engineer/kernel_interface_decomposer/` 增强 + 新增 CLI

**位置**

- KID 增强：`framework_engineer/kernel_interface_decomposer/` 现有模块
- 新 CLI：`framework_engineer/cli.py` 新增 `import-decomposition` 子命令 + 新文件 `framework_engineer/decomposition_importer.py`

**KID 增强详细项**

1. **入口双模式**：
   - 现有 `run` / `analyze` 子命令保留
   - 新增支持 `--target-kind {high_level, low_level}` 参数
   - `high_level` 模式：现有拆解 + 耗时统计逻辑
   - `low_level` 模式：跳过拆解和耗时统计，仅做单个 kernel 的源码定位
2. **多启动命令支持**：
   - 现有 `service_cmd` 字段扩展为 `service_cmds: list[dict]`，每条形如 `{"backend_name": "triton", "cmd": "..."}`
   - 逐条启动、跑 profile、产出独立的 `decomposition_{backend_name}.schema.json`
   - 避免多 backend 结果混在同一份 schema 里
3. **四类源码定位 —— 委托给 `locate-kernel-source`（8.1.2），KID 不重复实现**：
   - KID 只负责为每个 low\_level target 产出 `(interface, archetype)`：运行时抓到的接口名 + 形态标签（F0–F8，见 8.1.0）
   - 调用 `locate-kernel-source`（或直接 import 其 helper `source_locator.py`）拿回四层 `source_locations`
   - 修改 `source_resolver.py`：不再自己实现跨仓/JIT 溯源，改为组织 `(interface, archetype)` 输入 + 消费返回结果，写入 schema entry：
     ```json
     {
       "interface": "torch.ops.sgl_kernel.fwd",
       "archetype": "F3",
       "source_locations": {
         "interface_definition": {"file": "...", "line_start": 100, "line_end": 200},
         "py_cpp_binding":       {"file": "...", "line_start": 50,  "line_end": 80} or null,
         "kernel_header":        {"file": "...", "line_start": 1,   "line_end": 60} or null,
         "kernel_impl":          {"file": "...", "line_start": 300, "line_end": 500}
       }
     }
     ```
   - null 规则由 8.1.2 保证：`interface_definition` / `kernel_impl` 不允许 null；`py_cpp_binding` / `kernel_header` 仅 DSL/triton 形态可 null。定位失败（非 F8）报错
4. **形态标注（KID 的唯一新增溯源职责）**：
   - KID 运行时已有 wrapper API / `torch.ops` / JIT module 调用信息，据此给每个 target 打 `archetype` 标签（复用现有 `_infer_category` 扩展到 F0–F8）
   - 打完标签即把源码定位交给 8.1.2，`third_party_manifest.json` 的路径由 skill 内部查，KID 无需感知 clone 细节

**`import-decomposition`** **CLI 详细**

**输入**

- `--schema <path>`：KID 产出的 `decomposition_{backend}.schema.json`
- `--workspace-out <path>`：workspace 目标目录

**处理逻辑**

1. 读 schema，遍历每个 low\_level target
2. 对每个 target 的四类源码位置：
   - 按行号范围拷贝对应内容到 `<workspace>/kernel_sources/<low_level_id>/<file_type>.{py,cc,h,cu,cpp}`
   - 若字段为 null，创建空文件 + 注释说明"该类型不适用（triton kernel 无 py↔c++ binding）"
3. 生成 `read_hints.txt`：
   ```
   interface_definition.py:  read lines 100-200
   py_cpp_binding.cc:        N/A (triton kernel)
   kernel_header.h:          N/A (triton kernel)
   kernel_impl.py:           read lines 300-500
   ```
4. 把生成的 `kernel_sources/<low_level_id>/` 路径**回填到 schema**（`decomposition_schema.json` 新增字段 `kernel_sources_dir`）

**输出**

```
<workspace>/
  decomposition_schema.json           # 已含 kernel_sources_dir 回填
  kernel_sources/
    <low_level_id>/
      interface_definition.py
      py_cpp_binding.cc               # 或空文件 + 注释
      kernel_header.h                 # 或空文件 + 注释
      kernel_impl.{py,cu,cpp}
      read_hints.txt
```

**约束**

- 拷贝时保留注释和 docstring
- 若源码文件超过 5000 行，只拷贝指定行号范围前后 200 行 padding，避免体积膨胀

**依赖**：Step 0.5 的 `resolve-third-party`（`third_party_manifest.json`）+ `locate-kernel-source`（四层定位）

**测试**

1. High\_level 场景：Qwen3.5 GDN，三条 backend（triton / flashinfer / cutedsl），验证：
   - 三个 workspace 目录生成
   - 每个 workspace 内 kernel\_sources/ 结构正确
   - read\_hints.txt 行号范围准确
2. Low\_level 场景：直接指定 `chunk_gated_delta_rule_fwd_intra`，验证只产出源码定位、无耗时统计

### 8.3 Step 2 — task\_pack 批量创建（三个新 CLI + 一个现有能力扩展）

#### 8.3.1 `workspace-to-config` CLI

**位置**：`framework_engineer/cli.py` 新增 + 新文件 `framework_engineer/workspace_to_config.py`

**输入**

- `--workspace <path>`：Step 1 产出的 workspace 目录
- `--config-out <path>`：task\_pack 生成任务的 config 草稿目标路径

**处理逻辑**

1. 读 `<workspace>/decomposition_schema.json`
2. 提取所有 low\_level target 的 file/line/name/kernel\_type，转成 phase1 config 的 `targets` 数组
3. 从 workspace 上下文（`<workspace>/schema.json` 里可能带的 launch metadata）填入 service\_cmd
4. 未知字段（forward\_boundary\_file / forward\_boundary\_line / task\_group\_id / output\_root 等）**留空 + 加醒目 TODO 注释**
5. 写出 config 草稿

**输出**：task\_pack 生成任务的 python config 文件（与 [phase1\_targets.example.py](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/framework_engineer/configs/phase1_targets.example.py) 结构一致，但 targets 已批量填好）

**约束**：不猜测空白字段的值，明确标 TODO

**测试**：给一个 workspace，生成的 config 补齐后能被现有 `validate-config` 通过

#### 8.3.2 Phase1 config schema 扩展 + 主链路小改造

**位置**：`framework_engineer/configs/phase1_targets.example.py` + `framework_engineer/cli.py`

**改动**

- 现有 `targets: list[dict]` 已支持多 target，无需大改；但需确认 `run-phase1` 批处理下**每个 target 独立成一个 task\_pack**（当前应该已经这么做，验证即可）
- 新增字段：`workspace_backend_name`（顶层，用于 task\_pack 顶层 manifest 标记归属哪个 backend workspace）

**测试**：多 low\_level 的 config 能正常跑完整个 phase1 链路

#### 8.3.3 `import-kernel-sources-to-taskpack` CLI

**位置**：`framework_engineer/cli.py` 新增 + 新文件 `framework_engineer/kernel_sources_importer.py`

**输入**

- `--workspace <path>`：Step 1 产出的 workspace
- `--task-packs-root <path>`：Step 2 phase1 生成的 task\_packs 根目录

**处理逻辑**
遍历 workspace 里每个 low\_level\_id：

1. 定位到对应的 task\_pack 目录（按 task\_id 匹配）
2. 把 `<workspace>/kernel_sources/<low_level_id>/` 整个拷贝到 `<task_pack>/original_source/kernels/<low_level_id>/`
3. 从 `decomposition_schema.json` 提取该 low\_level 的：
   - `kernel_time_cost_us`（KID 产物里的耗时；low\_level 直接指定模式下这个可为 null）
   - `kernel_time_ratio`（占总耗时的比例；low\_level 模式为 null）
   - 四个源码路径的相对路径
4. 生成 `<task_pack>/level_decision.yaml`：
   ```yaml
   oracle_level: 1                      # 默认 L1，Step 3 会更新
   evidence_paths:
     l4_python_reference: null
     l4_ut: null
     l3_logic_mapping: null
     l2_module_reference: null
   kernel_time_cost_us: 1234.5          # 或 null
   kernel_time_ratio: 0.23              # 或 null
   source_index:
     interface_definition: original_source/kernels/<id>/interface_definition.py
     py_cpp_binding:       original_source/kernels/<id>/py_cpp_binding.cc     # 或 null
     kernel_header:        original_source/kernels/<id>/kernel_header.h       # 或 null
     kernel_impl:          original_source/kernels/<id>/kernel_impl.py
   translate_notes: null                # Step 3 填
   ```

**输出**：每个 task\_pack 内新增 `original_source/kernels/<id>/` 目录 + `level_decision.yaml`

**约束**：拷贝时保持文件权限和行结束符不变

**依赖**：`workspace-to-config` + phase1 主链路已跑完（task\_pack 骨架已就绪）

**测试**：给一个跑完 phase1 的 workspace，验证每个 task\_pack 内 `level_decision.yaml` 字段完整、`original_source/kernels/` 拷贝正确

#### 8.3.4 现有 `validate-task-pack` 扩展 L1 检查

**位置**：`framework_engineer/cli.py` validate 子命令

**改动**

- 新增：读 `level_decision.yaml`，若 `oracle_level == 1`：
  - 检查 `source_index` 里各文件路径存在（允许 null 项）
  - 检查文件非空（对 null 项跳过）
- 若 `level_decision.yaml` 缺失或字段不全，validate 失败并明确报错

**测试**：Step 2 结束后跑 validate，所有 task\_pack（默认 L1）都能通过

### 8.4 Step 3 — `problem_translate` skill + `run-problem-translate` CLI

#### 8.4.1 `problem_translate` skill

**位置**：`framework_engineer/skills/problem_translate.md`

**类型**：Skill（**独立上下文** agent，每次调用处理一个 task\_pack）

**输入**（skill 通过 CLI 传给 agent）

- task\_pack 完整路径
- `sglang_repo_root`
- `external_references`（开放式列表，每条 `{repo_root, files?: list[str], line_ranges?: list[tuple[int,int]]}`；允许空列表）
- `third_party_manifest.json` 路径

**Agent 工作流程**（skill 内明确写出）

**Step A 前置强制搜索**（开放式，不限固定目录）

- 在 sglang 仓库任意位置搜索该 kernel API：**推荐但不限于** `sgl-kernel/benchmark/`、`sgl-kernel/tests/`、`python/sglang/test/`、其他 `test_*.py` / `bench_*.py` 文件
- 在对应 third-party 仓库任意位置搜索：**推荐但不限于** `tests/`、`examples/`、`benchmarks/`
- agent 自主判断搜索深度和范围
- 记录所有候选文件路径到 `problem_translate_report.json`

**Step B 尝试 L4**

1. 若找到测试文件里有 PyTorch 等价实现：
   - 抽取该实现，包装成 `python_reference/ref_impl.py`
   - manifest 标记 `origin: sgl_internal_test | third_party_test`
2. 若无现成参考，agent 阅读 `original_source/kernels/<id>/interface_definition.py` 和 `kernel_impl.*` 后 自行实现 PyTorch/基础python实现
3. 生成 `tests/ut_lv4.py`：
   ```python
   # 加载 task_pack/snapshots/ 全部 selected samples
   # 对每个 sample:
   #   1. 用 python_reference 跑 pre_inputs
   #   2. 与 snapshot 的 output / post_inputs 全量比对
   #   3. 使用 tolerance 检查
   ```
4. 跑 `ut_lv4.py`：
   - 全部 pass → `oracle_level = 4`，保留 UT 作为证据
   - 失败 → 降 L3

**Step C 尝试 L3**（依赖 `external_references`；若空则跳到 L1）

1. agent 分析 `original_source/kernels/<id>/` 内所有源码，理解数学问题
2. 遍历 `external_references` 里每个 `{repo_root, files?, line_ranges?}`，定位**计算逻辑等价的表达片段**（不要求一对一，可以是几行代码块）
3. 拷贝命中的源文件到 `logic_mapping/references/<repo>/<file>`（保留原始注释）
4. 生成 `logic_mapping/mapping.md`：
   ````markdown
   # op_mapping: <kernel_id>

   ---
   kind: L3

   ## sglang 侧
   kernel: <name> (<type>)
   文件: original_source/kernels/<id>/kernel_impl.py
   输入/输出: ...

   ## 外部参考等价片段
   文件: logic_mapping/references/<repo>/<file>:<line_range>
   ```python
   <抽取的代码块>
   ````
   ## 对应关系
   - \<sglang 概念> ↔ <外部概念>
   - ...
   ## 差异点
   - ...
   ```
   ```
5. 生成 `logic_mapping/manifest.json` 记录引用的源文件 sha256 + `kind: L3`
6. → `oracle_level = 3`；找不到 → 降 L2

**Step D 尝试 L2**（依赖 `external_references`；若空则跳到 L1）

- 在 `external_references` 里定位上层模块级参考代码（比 L3 粒度更粗，如整个类 / 整个 forward）
- 拷贝源文件到 `logic_mapping/references/<repo>/<file>`
- 生成/追加 `logic_mapping/mapping.md`：结构同 L3，`kind: L2`，说明部分聚焦"模块整体在做什么、与 sglang kernel 的层级关系"
- 追加 `logic_mapping/manifest.json` 中的 kind: L2 条目
- → `oracle_level = 2`；找不到 → L1

**Step E 更新** **`level_decision.yaml`**

- 更新 `oracle_level` 和 `evidence_paths`
- 追加 `translate_notes` 字段说明理由

**Step F 生成** **`docs/problem_translate_report.json`**

- 记录：搜索路径 / 找到的候选 / 每一级尝试的结果 / 最终 level / 决策理由

**产物**：见 Section 2.5

**约束**

- Agent 独立上下文；skill 必须 self-contained（不假设外部对话历史）
- Draft 的 python\_reference 必须能被 `ut_lv4.py` 加载
- L4 UT 失败必须诚实降级，不允许通过放宽 tolerance 强通过

**依赖**：Step 0.5 / Step 1 / Step 2 全部完成

#### 8.4.2 `run-problem-translate` CLI

**位置**：`framework_engineer/cli.py` 新增 + 新文件 `framework_engineer/problem_translator.py`

**输入**

- `--workspace <path>`：处理该 workspace 下所有 task\_pack
- 或 `--task-pack <path>`：处理单个 task\_pack
- `--external-references <path>`：一份 YAML/JSON 文件，含 `list[{repo_root, files?, line_ranges?}]`
- `--third-party-manifest <path>`

**处理逻辑**

1. 遍历 task\_pack（单个或批量）
2. 对每个 task\_pack 调用 `problem_translate` skill 一次
3. 收集每次调用的产物 + 状态
4. 生成汇总报告

**输出**：各 task\_pack 内的产物（见 8.4.1） + 一份汇总 `problem_translate_summary.md`

**约束**：批量模式下失败一个不影响其他 task\_pack

**测试**：

1. 人工准备 3 个 low\_level：一个明确 L4（如 RMSNorm，sgl-tests 有参考）、一个 L3（如 chunk\_intra，能找到 transformers 循环等价）、一个 L2（如某 sglang 独有 fused kernel）
2. 跑 skill，验证每个 task\_pack 被正确判定并产出证据文件

### 8.5 Step 4 — `validate-task-pack` 分级扩展 + Deliver 准备

**位置**：现有 `framework_engineer/cli.py` validate 子命令 + 新文件 `framework_engineer/l4_ut_runner.py` + `framework_engineer/deliver_preparer.py`

**改动 A：分级检查**
读 `level_decision.yaml.oracle_level`，按 level 执行：

| level   | 检查内容                                                                                                                                                                         |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L4      | 跑 `tests/ut_lv4.py`；必须全部 sample 通过；若失败：`level_decision.yaml.oracle_level` 自动降级为 3（或用户配置的 fallback level）并产出 `l4_ut_report.json`                                              |
| L3 / L2 | `logic_mapping/mapping.md` 存在；`logic_mapping/references/` 下的拷贝文件 sha256 与 `logic_mapping/manifest.json` 记录一致；manifest 里的 `kind` 字段与 `oracle_level` 一致（L3↔kind:L3、L2↔kind:L2） |
| L1      | 检查 `original_source/kernels/<id>/` 里四类源码文件齐全（允许 py\_cpp\_binding / kernel\_header 为空占位）                                                                                      |

**改动 B：Deliver 前边界化清理**（分级检查通过后执行）

1. 备份 `level_decision.yaml` 里的 `source_index` 字段到 `docs/pre_deliver_snapshot.json`（framework\_engineer 后续回接框架时用）
2. 从 `level_decision.yaml` 里**删除** **`source_index`** **字段**
3. 在 task\_pack 顶层创建 `candidate_impl_workspace/` 空目录，附一份 `README.md`：
   ```markdown
   # candidate_impl_workspace/

   kernel_engineer 的唯一可写工作区。允许放：
   - 自定义 kernel 源码（.cu / .cpp / .cuh / .py）
   - 编译产物（.so / .a / .o）
   - 临时脚本 / 构建脚本（Makefile / CMakeLists.txt / build.sh）
   - iteration_log 附属文件

   candidate_impl.py 通过相对导入或 dlopen 访问本目录下的内容。
   不允许写出 task_pack 外的任何路径。
   ```
4. 更新 `task.yaml`，追加 `kernel_engineer_scope` 合同段：
   ```yaml
   kernel_engineer_scope:
     writable_paths:
       - candidate_impl.py
       - candidate_impl_workspace/
       - docs/iteration_log.md
     read_only_paths:
       - snapshots/
       - snapshot_runtime.py
       - original_source/
       - original_impl.py
       - reference_impl.py
       - correctness_test.py
       - benchmark.py
       - python_reference/               # L4 才有
       - tests/ut_lv4.py                 # L4 才有
       - logic_mapping/                  # L3/L2 才有
       - level_decision.yaml
       - task.yaml
     forbidden_actions:
       - "读取或写入 task_pack 外的任意路径"
       - "修改 tolerance / timing rules / snapshot 数据"
       - "写入 correctness/benchmark harness"
   ```

**新增字段** `--l4-fail-mode {downgrade, error}`：

- `downgrade`（默认）：L4 UT 失败自动降级到 L3
- `error`（CI 用）：L4 UT 失败直接返回非零退出

**新增字段** `--skip-deliver-prep`：调试用，跳过改动 B（保留 source\_index / 不创建 workspace）

**产物新增**

- `docs/l4_ut_report.json`（L4 才有）
- `docs/pre_deliver_snapshot.json`（deliver 前备份的 source\_index 元数据）
- `candidate_impl_workspace/README.md`
- `task.yaml` 追加 `kernel_engineer_scope` 段

**测试**：

1. Step 3 产出的 task\_pack 全部 validate 通过
2. 人为破坏一个 L4 python\_reference 让 UT 失败，验证降级正确
3. Validate 后确认 `level_decision.yaml` 里 `source_index` 已删除、pre\_deliver\_snapshot.json 里已备份
4. 确认 `candidate_impl_workspace/` 存在、`task.yaml` 含 `kernel_engineer_scope`

### 8.6 Step 5 — `summarize-workspaces` skill + CLI

**位置**

- Skill：`framework_engineer/skills/summarize_workspaces.md`
- CLI：`framework_engineer/cli.py` 新增子命令 + 新文件 `framework_engineer/workspace_summarizer.py`

**输入**

- `--workspaces-root <path>`：所有 workspace 的根目录

**处理逻辑**

1. 遍历每个 workspace（每个 workspace = 一条 backend 路径）
2. 遍历该 workspace 下所有 task\_pack，读 `level_decision.yaml`
3. 计算 `workspace_score = Σ (level_weight × kernel_time_ratio)`
   - weight: L4=4, L3=3, L2=2, L1=1
   - 若 kernel\_time\_ratio 缺失（low\_level 直接指定模式），使用等权重（1/K）
4. 按 score 降序排列 workspace
5. 生成结构化 ranking + 人可读报告

**输出**

```
<workspaces-root>/
  workspace_ranking.json               # {workspace_name, score, rank, level_distribution, task_pack_count}
  summary_report.md                    # 人可读版：排名 / 每个 workspace 的组成分析 / 推荐理由 /
                                       # 降级路径建议（若推荐路径效果不好，用哪个备选）
```

**Skill 内容**：定义人可读报告的模板，让 agent 生成时保持结构一致

**约束**：不做任何 tolerance 或修改性动作，纯汇总

**测试**：手工构造 3 个 workspace（level 分布不同），验证排序结果与手工计算一致

### 8.7 最终整合阶段

按 Section 5 TODO 里最后一节的清单执行：

- 合并 config：把 Step 0.5 / Step 1 / Step 2 各步骤的独立 config 合并成一份统一 config，各步骤的 CLI 都能从这一份 config 读取自己需要的字段
- Schema 强类型：为每个中间产物（`decomposition_schema.json` / `level_decision.yaml` / `problem_translate_report.json` / `workspace_ranking.json`）写 JSON Schema，各 CLI 消费前先 validate
- 更新 `framework_engineer/prompts/framework_engineer.md`：加入 Step 0.5 / 多路径 / Step 3 / Step 5 相关职责
- 更新 `framework_engineer/README.md` + `framework_engineer/templates/task_pack_README.md`
- 端到端 smoke：GDN 完整跑通全部六个 step，输出至少 L2/L3/L4 各一个 case

***

## 9. 后续阅读入口

- 现状回顾：[kernel\_agent/backup/design\_notes/kernel\_agent\_phase1.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent/backup/design_notes/kernel_agent_phase1.md)
- 讨论过程：[kernel\_agent\_kadai/framework\_engineer\_design\_review.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent_kadai/framework_engineer_design_review.md)
- 核心问题总结：[kernel\_agent\_kadai/core\_problems.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent_kadai/core_problems.md)
- 用户 TODO 反馈：[kernel\_agent\_kadai/TODO\_and\_feedback.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent_kadai/TODO_and_feedback.md)
- 入口探讨：[kernel\_agent\_kadai/kernel\_agent\_entrance.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent_kadai/kernel_agent_entrance.md)
- sglang 侧多后端路径分解调研：[attention\_decompose\_sglang.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/attention_decompose_sglang.md)
- transformers 侧对应路径调研：[attention\_decompose\_transformers.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/attention_decompose_transformers.md)
- Attention adapter 调研（backend 路径参考）：[attention\_framework\_adapter.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/attention_framework_adapter.md)
- L3 logic\_mapping 参考格式：[op\_mapping\_qwen3\_5\_gdn\_sglang\_transformers.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/op_mapping_qwen3_5_gdn_sglang_transformers.md)
- 现有 KID 用法：[tools/kernel\_interface\_decomposer/README.md](file:///Users/bytedance/Desktop/remote_dev_project/model_ana/tools/kernel_interface_decomposer/README.md)

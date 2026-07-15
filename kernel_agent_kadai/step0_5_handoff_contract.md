# Step 0.5 → Step 2 交付契约（third_party_solver + KID + locate 的最终产物）

> 本文回答两个问题：
> 1. `resolve-third-party` + KID + `locate-kernel-source` 全跑完后，**理想交付给下一阶段（Step 2，已基本跑通的 phase1 task_pack 主链路）的产物完整形态**——按文件结构描述，每个文件标注**来自哪个 step / 由 CLI 还是 skill 产**。
> 2. 这份产物里**哪些部分 agent 可能做不到、必须人工介入**——从**最小化人工介入**角度界定：人工只补「判断/定位」这类 agent 真做不了的，**绝不做 CLI 能做的搬运/拷贝/组装**。
>
> 关联：[KID_and_locate_source_desgin_v2.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/kernel_agent_kadai/KID_and_locate_source_desgin_v2.md)（KID/locate 三层设计）、[framework_engineer_design_v2.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/kernel_agent_kadai/framework_engineer_design_v2.md)（Step 2 主链路）。
> 日期：2026-07-13。

---

## 0. 三个 step 的产物职责（先对齐谁产什么）

| Step | 组件 | 形态 | 产物 |
| --- | --- | --- | --- |
| 0.5-a | `resolve-third-party` | **CLI**（+ 兜底 skill） | `third_party_manifest.json`、`missing_repos.md`、`third_party_cache/<name>/<version>/` clone 源码树 |
| 1 | KID | **CLI**（无 agent） | `decomposition_<backend>.schema.json`（每 kernel：`interface`+`archetype`+`runtime_event`，**无** source_locations） |
| 0.5-b L1 | locate `locate` | **CLI** | 就地给 schema 每 kernel 补 `source_locations`+finalize archetype+`needs_agent`；`ref/locate_report.json`（参考） |
| 0.5-b L2 | locate agent | **skill** | 只补 L1 标 `ambiguous`/`not_found` 的层；`ref/locate_agent_notes.md`（参考） |
| 0.5-b L3 | locate `extract` | **CLI** | 按 `source_locations` 抽四层文件 → `kernel_sources/<id>/` + `read_hints.txt` + 回填 `kernel_sources_dir` |

> 关键：**Step 2 的入口是 `workspace-to-config`（§2.4）**，它读 `decomposition_<backend>.schema.json` 生成 phase1 config 草稿，再走已跑通的主链路。所以「交付产物」= **一个或多个 backend workspace**，每个 workspace 必须自包含到能喂给 `workspace-to-config` + phase1。

> **实现状态（2026-07-13）**：L3 `extract` 已实现（[source_location/extractor.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/source_location/extractor.py) + `source_location.cli extract`）。本契约描述的「人工介入后交付链跑通」已由 dry-run 机制（[dry_run/](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/dry_run)，`kid`/`locate`/`extract` 三步）在无 GPU 下端到端验证通过。archetype 产物用明文类别名 + `archetype_code`。

---

## 1. 理想交付产物的完整文件结构

一次 high_level 多 backend 运行（N 条 `service_cmds`）后，`<output_root>/` 的理想终态：

```
<output_root>/
│
├── third_party_manifest.json              # [0.5-a CLI] 每个外部库: name/archetype/version/
│                                           #   local_path/url/ref/status/clone_command...
│                                           #   顶层含 sgl_kernel 版本对齐信息 + failed 汇总
├── missing_repos.md                        # [0.5-a CLI] 仅当有 clone_failed/failed 时生成
│
├── third_party_cache/                      # [0.5-a CLI] clone 落地（按 (name,version) 分目录）
│   ├── flashinfer/<ver>/                    #   完整 git 源码树（供 locate 跨仓定位）
│   ├── deep_gemm/<ver>/
│   ├── sgl-attn/<commit>/ ...               #   Bucket B 的 pin commit
│   └── ...
│
└── workspaces/
    ├── <backend_1>/                         # 一条 backend 路径 = 一个 workspace
    │   ├── decomposition_<backend_1>.schema.json
    │   │        # [1 KID]    顶层 target(起点,用户指定的 high_level) + coverage_report(漏检自曝)
    │   │        #            每 kernel: interface/archetype(provisional 可 F2|F3)/
    │   │        #            metrics(耗时占比)/runtime_event(call_site+implementation)
    │   │        # [0.5-b L1] + source_locations{archetype(finalized)/source/needs_agent/
    │   │        #            layers{a,b,c,d: status/hits/repo_hint}}  # 层 a hits = 交 Step2 的 target file/line
    │   │        # [0.5-b L2] + ambiguous/not_found 层被补齐或标 missed
    │   │        # [0.5-b L3] + kernel_sources_dir 回填
    │   │
    │   ├── ref/                              # 参考资料目录（无固定消费者，仅供 agent/人参考）
    │   │   ├── locate_report.json           # [0.5-b L1] needs_agent 列表 + 统计（schema 的派生视图）
    │   │   └── locate_agent_notes.md        # [0.5-b L2] agent 产出：对 locate 结果的证据阐述 + 结果报告（内容由 prompt 约束）
    │   │
    │   └── kernel_sources/                   # [0.5-b L3 CLI] 抽取的四层源码（Step 2 拷进 task_pack 的物料）
    │       ├── <low_level_id_1>/
    │       │   ├── interface_definition.py   #   层 a（单文件；low_level_target 自身定义）
    │       │   ├── py_cpp_binding/           #   层 c（目录；多文件多格式；按 py→cpp 编号；triton/DSL → 空占位）
    │       │   │   ├── 1_<mod>.py            #     py 侧 build_and_load/load_jit
    │       │   │   └── 2_<mod>_binding.cu    #     c++ 侧 FFI 导出（flashinfer JIT）
    │       │   ├── kernel_impl/              #   层 b（目录；调用链多文件，按调用序编号）
    │       │   │   ├── 1_<launcher>.cu       #     launcher
    │       │   │   └── 2_<kernel>.cuh        #     真正的 __global__（可能跨仓库）
    │       │   ├── kernel_header/            #   层 d（目录；与 impl 源文件一一对应，不编号；triton/DSL/.cuh合一 → 空占位）
    │       │   │   └── <header>.h
    │       │   └── read_hints.txt            #   每个抽出文件的 read 行号范围（missed 层写占位说明）
    │       └── <low_level_id_2>/ ...
    │
    ├── <backend_2>/ ...
    └── ...
```

**Step 2 从这里接手**：`workspace-to-config` 读每个 `decomposition_<backend>.schema.json` → 生成 phase1 config（每个 low_level → 一个 target：`task_id`/`target_file`/`target_line`）→ 走主链路 → `import-kernel-sources-to-taskpack`（§8.3.3，Step 2 的 CLI）把 `kernel_sources/<id>/` 拷进各 task_pack 的 `original_source/kernels/<id>/`。

> **交给 Step 2 的 `target_file`/`target_line` 到底是 schema 里哪个字段？**（澄清双坐标，避免歧义）
> - **是每个 kernel entry 的 `source_locations.layers.interface_definition.hits[0].{file, def_line}`** —— 层 a（接口自身定义）。`workspace-to-config` 从这里读，转成 phase1 config 的 `targets[].target_file` / `target_line`。
> - schema 顶层还有个 KID 原始的 `target`（= 用户最初指定的 high_level 模块，分解的**起点**），**那不是**分解出来的 low_level target，别拿它喂 Step 2。
> - 链路：`source_locations.layers.interface_definition` →（`workspace-to-config`）→ `targets[].target_file/line` → task_pack。所以「最终交给下一阶段跑 task_pack 的文件/行号」确实**在 `decomposition_<backend>.schema.json` 里**，具体是每个 low_level 的层 a hits。

> 注意区分两个 importer：**L3 `extract`**（本文，产 `kernel_sources/`）vs **Step 2 `import-kernel-sources-to-taskpack`**（拷进 task_pack）。前者是 0.5-b 的收尾，后者是 Step 2 的组装。

---

## 2. 每个文件的「产出方 + 是否可能需要人工」总表

状态列含义：✅ 全自动（CLI/agent 完成，无需人工）；⚠️ 可能需人工（仅补判断/定位，不搬文件）；🚫 从不需人工。

| 产物 | 产出方 | 状态 | 人工介入点（若有） |
| --- | --- | --- | --- |
| `third_party_manifest.json` | 0.5-a CLI | ⚠️ | 仅 `version_mismatch=true` 或新库/新 tag 规则时，人工核对/补 registry（见 §3.1） |
| `missing_repos.md` | 0.5-a CLI | ✅ | 只读报告；clone_failed 由人跑给定命令，但不改产物结构 |
| `third_party_cache/<name>/<ver>/` | 0.5-a CLI | ⚠️ | clone_failed 时人工执行 manifest 里的 `clone_command`（一条命令，非搬文件） |
| `decomposition_*.schema.json` · KID kernel 列表 | 1 CLI | ⚠️ | **常态风险**：wrap 漏捕获 → 覆盖率低时人工决定补追踪配置 or 接受风险，重跑 KID（见 §3.2） |
| `decomposition_*.schema.json` · `coverage_report` | 1 CLI | ✅ | KID 自动算 + 告警；只读，供人判断是否有遗漏 |
| `decomposition_*.schema.json` · source_locations L1 | 0.5-b L1 CLI | ✅ | — |
| `decomposition_*.schema.json` · L2 兜底 | 0.5-b L2 skill | ⚠️ | agent 仍 `missed` 的必填层 → **硬停**，人工填 file+行号 或 明确放空（见 §3.3，**最主要人工点**） |
| `ref/locate_report.json` | 0.5-b L1 CLI | ✅ | 参考资料，无固定消费者 |
| `ref/locate_agent_notes.md` | 0.5-b L2 skill | ✅ | 参考资料，无固定消费者；内容由 prompt 约束（证据 + 结果报告） |
| `kernel_sources/<id>/*` 四层文件 | 0.5-b L3 CLI | 🚫 | **绝不需要人工搬/建**——含空文件与占位，全由 CLI 按 source_locations 生成。人工补的是「行号」（回 schema），CLI 重跑即生成 |
| `read_hints.txt` | 0.5-b L3 CLI | 🚫 | 同上，CLI 自动（含 N/A / MISSING 说明） |

**核心原则（回答问题 2）**：人工只在**两类"判断"**上介入——(1) KID 覆盖率低时「补追踪配置 or 接受风险」，(2) locate 定位失败时「补 file+行号 or 明确放空」——且介入点都在**结构化数据（schema / registry / 追踪配置）**里，改完 **重跑对应 CLI** 即产最终物料。人工**从不**直接动 `kernel_sources/` 下的文件、从不搬运/拷贝/新建空文件。

---

## 3. 可能需要人工的三处（按最小化介入排序）

### 3.1 third_party：版本/来源存疑（低频，规则维护）
- **何时**：`version_mismatch=true`（sgl-kernel 源码树版本≠装的版本）；或出现 registry 未覆盖的新库 / tag 命名变化 / clone 失败。
- **agent 做不到的部分**：判断"该不该 checkout 匹配版本"、新库的 `ref_template` 该怎么写（需 `git ls-remote` 实证）。
- **人工介入形态**：改 `registry.py` 条目 或 手动 checkout 源码树，**然后重跑 CLI**。
- **不需要人工**：clone 本身、manifest 生成、目录组织——全 CLI。

### 3.2 KID：kernel launch 漏捕获（**非低频，是真实主风险**）+ 覆盖率自曝机制

- **为什么非低频**：KID 的 wrap 插桩**尚未经端到端验证**，且已知盲点多——torch 白名单外的算子（`torch.matmul`/`a@b`/`aten.*` 不走 functional）、新版 triton 换启动路径、sgl_kernel 的 `torch.ops` / re-export 不满足 owner 门槛。**实际落地大概率会漏掉部分 kernel launch**。漏掉的 kernel 不会进 schema，用户若不知情就会**静默少优化**。
- **agent 做不到的部分**：KID 是纯 CLI 无 agent；"某个 kernel 该不该被 track"是用户的优化意图，agent 不能替判。
- **必须新增的机制——覆盖率自曝**（KID 收尾输出，让用户感知漏检风险）：
  - **指标**：`coverage = Σ(被 wrap 归因+选中的 kernel 耗时) / Σ(target 时间窗内 GPU 上实际跑的所有 kernel 耗时)`。
    - 分子：拆分后已捕获的（现有 `total_us` 逻辑，[trace_parser.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/trace_parser.py#L119) 已有）。
    - 分母：**口径②**——落在该 target range 时间窗内的**全部** CUDA kernel 耗时和（含未被 NVTX 归因的漏网 kernel，nsys 仍记录了它们）。KID 需新增这步统计。
    - `coverage → 1`：几乎全被捕获，放心；`coverage 明显 < 1`：**有大量未 wrap 的 kernel，遗漏风险高**。
  - **产出**：每个 backend 的 schema 顶层写 `coverage_report`：`{per_invocation: [{call_id, stage, covered_us, total_gpu_us, coverage}], min_coverage, uncaptured_hint}`；并在 CLI 收尾**打印醒目提示**（stderr）：覆盖率低于阈值时告警 + 指引"如何手动补配置"。
- **用户的两种手动补法**（CLI 告警里说明）：
  1. **补 target/接口**：把漏掉的 kernel 所属接口显式加进 KID 的追踪配置（如扩 wrap 名单 / 显式列接口），重跑 KID。
  2. **明确排除**：用户判断该 kernel 不优化 → 接受当前覆盖率，视为**已知风险**继续。
- **不在产物里手填 kernel**：补的是「追踪配置」，重跑 KID 由 CLI 重新捕获，不手动往 schema 塞 kernel entry。

### 3.3 locate：四层定位失败 → **硬停，提示补全**（最主要的人工点）
- **何时**：L1 deterministic 定位不了、L2 agent 也 `missed` 的**必填层**（`interface_definition` / `kernel_impl` 不允许最终 null）。
- **agent 做不到的部分**：某些跨仓/生成代码/罕见形态，agent 读代码也定位不到准确的 file+行号范围。
- **执行序（硬停，不降级放行）**：
  1. L2 跑完仍有 `missed` 必填层 → **L3 `extract` 不自动往下跑，停下并提示用户补全**（列出哪些接口的哪层 missed + `repo_hint`）。
  2. 用户在 schema 的 `source_locations.layers.<layer>` 填入 `{file, def_line}`、status 改 `resolved`；**或**用户明确决定**放空**该层（= 已知风险，自担）。
  3. 用户补完 → **重跑 L3 `extract` CLI**：对 `resolved` 层正常抽取；对用户放空/仍 missed 的层，**由 CLI 固化生成占位空文件 + 对应 read_hints 说明**（见 §4）。
- **人工只碰 schema 的行号，绝不手动建/拷 `kernel_sources` 下的文件**——空文件和 line_hint 也归 L3 CLI 产（固化，见 §4）。
- **为什么硬停而非降级**：定位（判断源码在哪）是智能问题、抽取（按定位搬文件）是机械问题；硬停强制"判断"这一步有人负责（补或明确放空），杜绝"流水线悄悄放行一个空 target"。

---

## 4. 「承上启下」的 L3 extract：重新明确定位

你指出 L3 extract 是"承上启下的一层 CLI：接 L2 输出、生成最终 kernel_sources、对接已完成的 Step 2 流程"。据此明确其契约：

- **上游依赖（承上）+ 硬停闸门**：读 L1+L2 富化后的 `decomposition_*.schema.json`。**若仍有 `missed` 必填层（`interface_definition`/`kernel_impl`）→ 硬停**（不自动放行），列出待补项提示用户（§3.3）。用户补成 `resolved` 或**明确放空**后，才继续。
  - **填错=没填**：闸门不仅查 status/占位，还校验 `hits[].file` **真实存在**且 `def_line` 有效（在文件行数范围内）。用户填了不存在的路径/文件或越界行号时，该层**视同未定位**——必填层触发硬停（原因如 `file not found: ...`、`def_line N out of range`），非必填层记占位（`read_hints` 标 `MISSING (file not found: ...)`）。杜绝"填错被静默放行"。
- **本层动作（纯机械，全部固化进 CLI）**：
  1. `resolved` 层：**拷贝整个源文件**到产物（单文件层 `interface_definition` → `kernel_sources/<id>/interface_definition.{ext}`；目录层 `kernel_impl`/`kernel_header`/`py_cpp_binding` → `kernel_sources/<id>/<layer>/` 下每 hit 一个文件，有序目录层 `kernel_impl`/`py_cpp_binding` 带 `<n>_` 前缀）。不截断内容，定义范围由 CLI 按文件类型（py 用 AST/缩进、cpp/cu 用花括号配对，跳过注释/字符串）算出 `def_line`~end 记进 `read_hints.txt`。
  2. `not_applicable` 层（形态决定的合法 null，如 triton 无 binding/header）：**CLI 生成空文件 + 注释「该层形态不适用」**。
  3. 用户放空 / 仍 `missed` 的层：**CLI 生成占位空文件 + 注释「该层未定位，见 ref/locate_agent_notes.md，用户已知风险」**。
  4. `read_hints.txt`：**CLI 为每层写 read 行号范围**；对空/占位层写「N/A（不适用）」或「MISSING（待补）」说明。
  5. 回填 `kernel_sources_dir` 到 schema。
  - **要点**：2/3/4 里的「空文件 + line_hint 生成」是 **L3 CLI 的固化职责**，人工**绝不手动创建这些文件**。
- **下游对接（启下）**：产出的 `kernel_sources/<id>/` 结构，正是 Step 2 `import-kernel-sources-to-taskpack` 期望拷进 `original_source/kernels/<id>/` 的形态——**字段/目录名对齐**，Step 2 不需改造即可消费。
- **幂等**：人工补了 schema 行号后重跑 extract，覆盖式重生成对应 `kernel_sources`，不产生脏文件。

---

## 5. 一句话交付定义

**理想交付 = `third_party_cache/`（clone 源码）+ 每个 backend 的 `decomposition_*.schema.json`（KID 分类 + `coverage_report` 漏检自曝 + locate 四层定位，全 resolved 或用户明确放空）+ `kernel_sources/<id>/`（L3 抽取的四层物料，含空/占位文件）**。人工介入压缩到**两类判断**：(1) KID 覆盖率低时决定补追踪配置 or 接受风险；(2) locate 定位失败时在 schema 补 file+行号 or 明确放空。两者改完都**重跑对应 CLI 自动出物料**——**人工永不搬/建 `kernel_sources` 下的文件**。

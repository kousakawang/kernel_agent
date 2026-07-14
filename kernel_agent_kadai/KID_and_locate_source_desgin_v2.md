# KID 与 locate-kernel-source 联合设计 V2

> 本文聚焦 **Step 1（KID 分解）与 Step 0.5 第二个 skill（locate-kernel-source）之间的职责边界与对接契约**，是对
> [framework_engineer_design_v2.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/kernel_agent_kadai/framework_engineer_design_v2.md) §8.1.2 / §8.2 的**收敛与细化**。
> 这两步跑完直接对接已开发好的 phase1 主链路（locate 的 Layer 3 抽取产出 `kernel_sources/` → task_pack），所以边界必须先钉死。
>
> 关联现状：`resolve-third-party`（Step 0.5 第一个 skill）已完全落地并在真实 H 卡容器验证；KID 已有单路径能力；本文要解决的是「KID 里已经写了一半的溯源逻辑」和「计划中的 locate-kernel-source」之间的重叠。
>
> 日期：2026-07-12

---

## 0. 本文要回答的三个问题

用户提出的核心问题：

1. **调用关系**：`locate-kernel-source` 是**在 KID 内部调用**，还是 **KID 先输出自己的结果、再单独调用 locate**？二者当前功能有重叠，要重新划分。
2. **KID 最新设计**：输入（含用户输入 + 前序 skill 结果）、执行流程（哪些 CLI / 哪些 agent 兜底）、输出。
3. **locate-kernel-source 设计**：输入（KID 给的 + resolve-third-party 给的 + 用户给的）、输出（文件 + 工作目录状态）、workflow（CLI 跑一轮 + agent 读代码兜底）+ CLI 接口。

**先给结论（§1），再给现状盘点（§2）、边界表（§3）、两个组件的详细设计（§4/§5）、对接契约（§6）、落地改造清单（§7）。**

---

## 1. 核心决策：KID 输出结果，locate 单独跑（分离两遍 + 共享 helper）

**结论：KID 只做「运行时观测 + 形态分类」，把结果（含 `runtime_event`）序列化进 schema；`locate-kernel-source` 作为独立的第二遍，读 KID schema + `third_party_manifest.json` + `sglang_repo_root`，做四层源码定位。二者不在同一进程里耦合，但共享同一个 deterministic helper 包。**

### 1.1 为什么这样切（决定性理由）

真正决定边界的是一条物理约束：**「机制②（JIT 源码列表）+ triton kernel 的 file/line + wrapper 的 api/file/line」只能在服务运行时抓到**——`gen_jit_spec(sources=[...])` 的实参、`JITFunction.fn.__code__.co_filename` 都是运行态对象，只有把 sglang 服务真跑起来的那个进程（KID）看得到。而**四层源码的静态定位（符号 grep 到 clone、跨仓、按后缀分层）只需要磁盘上的文件 + manifest**，跟运行时无关。

沿这条「运行时 vs 静态」的缝切，四条好处：

1. **重跑不必重新 profile**：nsys 起真实 GPU 服务要几分钟；源码定位是纯文件 IO + grep。定位逻辑/agent 兜底要反复迭代，分离后可以只重跑定位、不碰服务。若塞进 KID，等于每调一次定位逻辑就重启一次服务。
2. **agent 兜底天然是「后置独立上下文」**：locate 的第二层是 agent 读第一层结果、只处理歧义层（见 §5）。agent 没法干净地嵌进 nsys 包裹的子进程流水线里；它必须读一份**已完成的 schema**。既然 Layer 2 一定是后置独立的，Layer 1 也后置才一致。
3. **依赖方向干净**：KID 保持成一个「只观测 + 分类」的 profiler，**完全不需要知道 third-party manifest / clone 缓存的存在**。只有 locate 碰 manifest 路径。这正好接上已有的 `resolve-third-party → manifest → locate` 依赖链。
4. **不丢信息**：KID 把运行时独有的事实（JIT sources、triton file/line、wrapper）**逐字序列化**进 schema 的 `runtime_event`（这些本就是字符串/路径，序列化零损失），locate 后置消费即可。

> **反方（在 KID 内部调用 locate）为什么不选**：即便内联，agent 兜底仍得后置独立跑；而且你迟早要给 KID 加一个「跳过 profile、只重定位」的模式——那就是把「分离的第二遍」在 KID 内部重新发明一次。内联唯一的好处是「一条命令出全量 schema」，但代价是把 profiler 和三方库布局耦合死。**共享 helper 仍允许 KID `import` 它做 in-process 快路径**（若将来真需要），但默认按两遍跑。

### 1.2 重叠具体怎么消除（这是本文最实的产出）

当前 [source_resolver.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/source_resolver.py) 里做了两类事，把它们**拆开**：

| 现在在 KID `source_resolver.py` 里的东西 | 归属 | 动作 |
| --- | --- | --- |
| `locate_function_at`（target/wrapper 的 qualname+行号） | **留 KID** | 分类/wrapper 元数据需要 |
| `_infer_category`（pytorch/sgl_kernel/triton/third_party 粗分类） | **留 KID**，扩成形态族分类 | 见 §4.3 |
| `_resolve_implementation` 的静态分支 | **移到 locate** | 删掉 KID 侧 |
| `_resolve_triton` / `_find_triton_definition`（grep `def <name>`） | **移到 locate** | 成为 locate 的静态兜底 |
| `_resolve_sgl_kernel` / `_sgl_kernel_registry` / `_find_symbol_sources`（机制①） | **移到 locate** | locate 的核心静态逻辑 |
| `_implementation_from_events`（优先读运行时 event） | **移到 locate** | 成为 locate 的「runtime_event 优先」读取器（§5.3 信息源优先级） |

拆完后：
- **KID 的 `source_resolver.py` 收缩**成「形态分类器 + target/wrapper 定位器」，不再做任何跨仓/JIT/符号级源码定位。
- KID 的 [trace_parser.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/trace_parser.py) 的 `_record_to_schema` **不再调用 `resolver.resolve()` 去解析 implementation**，改为把 `runtime_event`（wrapper dict + 命中 event 的 implementation dict）**原样**写进 schema，再附一个 `archetype` 标签。
- 那些静态逻辑搬进新包 `framework_engineer/source_location/`（见 §5.1），由 locate 的 CLI（Layer 1）和 agent（Layer 2）共用。

---

## 2. 现状盘点：KID 已经实现了什么

（读 [framework_engineer/kernel_interface_decomposer/](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer) 全部模块后的事实梳理，用来精确定位重叠。）

| 模块 | 现状 | 与本设计的关系 |
| --- | --- | --- |
| [runtime_instrumentation.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py) | 运行时插桩：NVTX 打点 target/wrapper；patch `torch.nn.functional`、`sgl_kernel.*`、`triton.runtime.jit/autotuner`、`sglang.jit_kernel.utils.load_jit` | **KID 独有的运行时观测层**。`load_jit` 已抓 `cpp_files/cuda_files/cpp_wrappers/cuda_wrappers/compile_flags`（= **机制② 原始数据**）；triton patch 已抓 `kernel_file/kernel_line`（= F1/F6 层 b 白送）。**保留 + 增强：补 patch flashinfer/deep_gemm JIT 入口以覆盖 F7**（见 §4.5） |
| [trace_parser.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/trace_parser.py) | 解析 nsys sqlite；per-invocation top-K 热点选择；已有 `RuntimeEvent`、`implementation` 字段 | **保留**。改动点：`_record_to_schema` 输出 `runtime_event` 而非静态 resolved implementation |
| [source_resolver.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/source_resolver.py) | `_infer_category` + 机制①（`_sgl_kernel_registry`+`_find_symbol_sources`）+ triton def grep + `_implementation_from_events` | **重叠所在**。分类留下，静态溯源全部移出（见 §1.2） |
| [runner.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runner.py) | 服务生命周期（起服务/等 ready/跑 test/停）+ export sqlite + `_build_schema` | **保留 + 加多 backend 循环**（见 §4.4） |
| [config.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/config.py) | 单 `service_cmd`/`test_cmd`/单 `target` | **扩** `service_cmds: list`（统一 high_level_target，不加 `target_kind`） |
| [cli.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/cli.py) | `run` / `analyze` 子命令 | **保留**，`run` 支持多 backend |

**关键发现**：KID 的 `source_resolver.py` 里 `_implementation_from_events` 已经实现了「优先用运行时 event、缺了才 fallback 静态」的优先级——这正是 §8.1.2 里 locate 的「信息源优先级」。所以这段逻辑不是删除，是**搬家**到 locate。

`resolve-third-party` 侧（本文的上游输入）已定稿：[registry.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/third_party_solver/registry.py) 的 `UNIVERSE`、[manifest.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/third_party_solver/manifest.py) 的 `RepoRecord`（`name/archetype/version/local_path/url/ref/status`…）。locate 直接消费其产出的 `third_party_manifest.json`。

---

## 3. 职责边界表（谁拥有什么）

| 能力 | KID（Step 1） | locate（Step 0.5-b） | 备注 |
| --- | :---: | :---: | --- |
| 起服务 / nsys profile / trace 解析 | ✅ | — | 运行时，唯 KID |
| 热点选择（top-K + duration/share） | ✅ | — | — |
| 运行时抓 `load_jit` sources（机制②原始数据） | ✅ | — | 只运行时可见 |
| 运行时抓 triton `kernel_file/line` | ✅ | — | F1/F6 层 b 白送 |
| 运行时抓 wrapper `api/file/line` | ✅ | — | 层 a 几乎白送 |
| 形态**族**分类（F0/F1/F4/F6/F7/F8） | ✅ | — | 用运行时信号 + 路径前缀，见 §4.3 |
| **F2 vs F3** 最终判定 | — | ✅ | 需静态符号解析（AOT 编进 .so，运行时看不到源） |
| 机制①：`m.impl` 符号注册表 + 符号 grep 到源 | — | ✅ | 从 KID 移出 |
| 机制②：把 JIT `sources[]` 按后缀分成层 b/c/d | — | ✅ | 消费 KID 的 `runtime_event` |
| 跨仓定位（符号命中 fetch 的 clone） | — | ✅ | 用 manifest 的 `local_path` |
| triton `def <name>` 静态 grep | — | ✅（兜底） | 运行时已给 file/line 时不需要 |
| 四层 `source_locations` 产出 | — | ✅ | 层 a/b/c/d |
| agent 兜底（歧义/未命中层） | — | ✅ Layer 2 | KID 无 agent |
| 读 `third_party_manifest.json` | ❌（不感知） | ✅ | 依赖方向：只有 locate 碰三方布局 |

**一句话**：KID 回答「**跑了哪些 kernel、各是什么形态、运行时看到的源在哪**」；locate 回答「**这个 (接口, 形态) 的四层源码文件到底在磁盘哪个位置**」。

---

## 4. KID 最新设计（Step 1）

### 4.1 类型

**纯 CLI，无 agent**。现有 [framework_engineer/kernel_interface_decomposer/](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer) 增强。任何需要 agent 判断的模糊性都**下沉到 locate 的 Layer 2**（因为模糊性本质都是「源码在哪」，属于定位问题）。

> **入口统一为 high_level_target（取消 low_level/high_level 二分）**：一个 target 底下 launch 几个 kernel，是 KID profile 完才知道的运行时事实，用户在指定时无从判断——所以「让用户声明 target_kind」是要求他预知无法预知之事的伪参数。KID 对 target 永远只做同一套「打标 → 归因 → 分类 → 存证」，缝回来 1 个 kernel（用户以为的 low_level）还是 N 个（high_level），只是同一动作的结果规模不同。**真正的分叉在「给几条 `service_cmds`」而非「几个 kernel」**：`len==1` = 单路径（不触发 Step 5 workspace 排序），`len==N` = 多 backend。用户只需回答他真的知道的问题（要试几条启动命令），不再声明 kind。

### 4.2 输入

**用户提供**：

| 项 | 说明 |
| --- | --- |
| `service_cmds: list[{backend_name, cmd}]` | 每条一条 backend 路径。1 条=单路径；N 条=多 backend（Step 5 才排序）。**KID 强制给每条 cmd 追加 `--disable-cuda-graph`**（见下方约束） |
| `test_cmd` | 触发 target 被调用的**静态测试脚本**；须满足下方「一次 prefill + 若干 decode」约束 |
| `target: {file, line}` | high_level_target 的位置（统一入口，不再区分 kind） |
| `sglang_repo_root` | 分类需要（判断 triton def 是否在 sglang 树内）；**不含 clone 路径** |
| `selection` / `profiling` | 现有字段（`top_k` / 阈值 / nsys 选项） |

**前序 skill 提供**：**无**。这是本次重划的直接收益——KID **不消费 `third_party_manifest.json`**。manifest 是 locate 的输入，不是 KID 的。KID 唯一的跨来源输入是 `sglang_repo_root`（用于分类启发式），不涉及三方 clone 缓存。

> 对比旧设计（§8.2 曾把 `third_party_manifest.json` 列为 KID 输入 + `target_kind` 双模式）：本文明确**移除**这两者。KID 打完形态标签就交给 locate，clone 路径由 locate 内部查。

**用户侧硬约束（非 KID 实现范围，但决定 KID 归因是否正确）**：

1. **必须非 cuda-graph（eager）模式**。cuda-graph 下 kernel 先 capture 后 replay，replay 时整个 graph 一次性提交、**不再逐个经过 KID patch 的 Python wrapper**，`type=wrap` 的 NVTX range 与 correlation_id 归因会大面积失效。**KID 兜底**：对每条 `service_cmds[i].cmd` 强制追加 `--disable-cuda-graph`（必要时 `--disable-cuda-graph-padding`），已存在则跳过。
2. **test 脚本须「一次 prefill + 若干 decode」的静态脚本**（如单条固定 prompt 的一次生成），不要压测/多并发。配合下方「只取代表性 invocation」，避免把 warmup/JIT 编译期的数据统计进热点。

### 4.3 形态族分类（KID 唯一的新增溯源职责）

KID 用**运行时信号 + 文件路径前缀**给每个 selected kernel 打 `archetype`，多数形态运行时即可确定；只有 F2/F3 的边界留给 locate 精化。扩展现有 `_infer_category`：

| 运行时信号 | KID 打的 archetype | 备注 |
| --- | --- | --- |
| api 以 `torch.`/`aten::`/`torch.nn.functional` 开头 | `F0` | 停在 API |
| triton launch，且 `kernel_file` 在 `sglang_repo_root` 内 | `F1` | sglang 自带 triton |
| triton launch，且 `kernel_file` 在 sglang 树**外**（三方装包/clone） | `F6` | 三方 triton/cuteDSL |
| 走 `sglang.jit_kernel.utils.load_jit`（KID 已 patch） | `F4` | sglang-owned JIT |
| 走 `flashinfer.jit.core.gen_jit_spec`（KID 增补 patch，见 §4.5） | `F7` | flashinfer C++ JIT；`source_files` 运行时白送 |
| 走 deep_gemm 入口（KID 不 patch） | `F7` | source_files 空，层 b/c/d 交 locate 静态（源集固定） |
| api 以 `sgl_kernel`/`torch.ops.sgl_kernel` 开头 | **`F2|F3`（provisional）** | 运行时看不到 .so 里的源，F2/F3 交 locate 判 |
| `implementation.source_files` 指向下载的 `.cubin` / deny-list 前缀 | `F8` | 无源 |
| 其余 | `unknown` | 交 locate + agent |

> **为什么 F2/F3 只能 provisional**：sgl_kernel 的算子是 AOT 编进 `.so` 的，运行时没有任何源文件路径（`implementation.source_files` 为空，见 [runtime_instrumentation.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py) 里 sgl_kernel 分支不带 source_files）。「符号定义在 `sgl-kernel/csrc/`（F2）还是在被 fetch 的 clone 里（F3）」本身就是一个**静态源码定位事实**，必须等 locate 做符号 grep 才知道。KID 硬猜只会错，所以老实标 `F2|F3`。

### 4.4 执行流程（全 CLI）

```
for each service_cmds[i] = {backend_name, cmd}:
  0. 给 cmd 追加 --disable-cuda-graph（若缺）                    [新：eager 强制]
  1. 写 runtime inject 文件（sitecustomize + runtime_config）   [现有 _write_runtime_files]
  2. nsys 起服务(cmd) → 等 ready → 跑 test_cmd(静态脚本) → 停    [现有 runner]
  3. export sqlite                                              [现有 export_sqlite]
  4. 解析 trace → 按 call_id 分 invocation → 取代表性 invocation
       → 每个代表 invocation 内 per-invocation 选热点 kernel     [trace_parser + 新聚合]
  5. 对每个 selected kernel：形态族分类（§4.3）                 [改造后的 source_resolver]
  6. 附 runtime_event（wrapper + JIT/triton 运行时源信息）      [新：原样序列化]
  7. 写 decomposition_<backend_name>.schema.json
```

**热点统计 + 多次 forward 的处理（步骤 4，重点）**：
- **每次 target 调用 = 一个 `call_id` = 一条独立 invocation**（runtime 侧 `_CALL_COUNTER` 自增；kernel 按落在哪个 call_id 的 target range 归组）。同一 target 被调 100 次 → 100 条 invocation。
- **热点是 per-invocation 的**：一条 invocation 内所有缝进来的 kernel 按 `duration_us` 降序，过 `min_duration_us` / `min_share_in_invocation` 阈值，取 `top_k`（现有 [`_select_records`](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/trace_parser.py#L115-L129)）。
- **只取代表性 invocation，剔除 warmup**（新增，替代旧的"输出全部 invocation"）：配合"静态脚本只跑一次 prefill + 若干 decode"的约束，KID **按 stage 各取代表**——取**第一次 `prefill` invocation + 第一次 `decode` invocation** 两条（`skip_target_invocations` 仍可用来先跳过启动/JIT 编译期的前几次）。这样每条 backend 的 schema 里 target 只留「一次 prefill 热点 + 一次 decode 热点」两组，不再有上百条近乎重复的 invocation。

- **多 backend**：外层循环，每条命令产出独立 `decomposition_<backend>.schema.json`，互不覆盖。

**无 agent 兜底**。

### 4.5 运行时插桩增强：patch flashinfer/deep_gemm 的 JIT 入口（已采纳，确定项）

**这是本设计的确定项，不是可选优化。** 选它是因为：F7 的 JIT 源文件列表在运行时可**零歧义**抓到，比 locate 静态解析 `gen_jit_spec` 更稳——静态解析碰到「`sources` 是动态循环/条件拼出来的」「需自己解路径锚点（`FLASHINFER_CSRC_DIR` 指哪）」「需先判断接口调的是哪个 gen 函数」这几种情况都会变脆。

**现状缺口**：当前 [runtime_instrumentation.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py) 只 patch 了 `sglang.jit_kernel.utils.load_jit`（覆盖 F4），**没 patch flashinfer 自己的 JIT 入口**（它不走 sglang 的 `load_jit`，走自己的 `gen_jit_spec`，属 F7）。后果：F7 的 `runtime_event.implementation.source_files` 为空，被迫走 locate 静态兜底。

**这个 patch 到底是什么（澄清，避免误解）**：和 KID 现有对 torch/triton/sgl_kernel/`load_jit` 的插桩是**完全同一套机制**——**运行时内存里的 monkey-patch，不是去改磁盘上 pip 装的 flashinfer 文件**：

- KID 通过注入的 `sitecustomize.py` + import hook，在服务进程 import flashinfer 后，对**内存里的 module 对象** `setattr` 换成包一层的函数（同 [`_patch_sglang_jit_utils`](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py#L522-L523) 里的 `setattr(module, "load_jit", patched_load_jit)`）。
- 包一层的函数**透明转发**：先把 `sources=[...]` 实参（此刻已求值成绝对路径）记录下来，再调用原函数，行为不变。
- 只在 `KID_ENABLE=1` 启的那次 profiling 进程内生效，进程退出即消失。**不写任何文件、不装包、不需要对 site-packages 有写权限**——不违反「不破坏运行环境」红线（与现有插桩同风险等级）。

**落地位置**：改的是 **KID 自己的代码**（新增 `_patch_flashinfer_jit(module)` + 在 [`_instrument_module`](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py#L526-L539) / [`_should_consider_module`](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py#L570-L584) 注册 flashinfer 的 JIT 模块），**flashinfer 本身一个字节不动**。

##### 入口已实测确认（flashinfer 0.6.12 / deep_gemm 0.1.2）

第 1 级探测（下方脚本）已在目标容器跑完（输出见 `jit_probe.txt`），结论钉死：

| 库 | 版本 | 是否 patch | hook 点 | `sources` 位置 | 依据 |
| --- | --- | --- | --- | --- | --- |
| **flashinfer** | 0.6.12 | **是（高价值）** | `flashinfer.jit.core.gen_jit_spec(name, sources, ...)` | **第 2 个位置参**（关键字名 `sources`） | 所有 `gen_xxx_module()`（attention/gemm/**gdn**/norm/comm/mamba…）全汇聚到此；调用点均 `gen_jit_spec(uri, source_paths)`（core.py:404-469、gdn.py:86、modules.py:315…） |
| **deep_gemm** | 0.1.2 | **否（静态即可）** | 无带 sources 参数的调用可 hook | — | 构建入口 `deep_gemm/__init__.py:41 _build_module`，`cpp_files=[.../csrc/tvm_ffi_api.cpp]` 是写死字面量（line 99）；真实 kernel 由 `include/` 模板 NVRTC 编译，源集每版本固定 → locate 静态定位足够 |

**flashinfer 的关键收益：单点覆盖整个 F7。** 只 patch `gen_jit_spec` 一个函数，就抓到所有 flashinfer JIT 模块（含 GDN prefill）的 `sources` 列表。sources 是已求值的绝对 `Path`，锚在 `jit_env.FLASHINFER_CSRC_DIR / "*.cu"`（= 机制②的锚点；attention/gemm 还会把模板生成的 `.cu` append 进 `source_paths`，**这类生成文件的真实路径正是运行时 hook 才拿得到、静态解析拿不全的**——进一步印证 patch 价值）。触发编译的是 `JitSpec.build_and_load()`（core.py:307），若要「只记实际用到的」可再 hook 它；否则 hook `gen_jit_spec` 记 `name→sources` 已够。

> **F8 边界也一并确认**：`get_trtllm_gen_*` + `cubin_loader.py`（decode.py:326 / prefill.py:216 等）走下载 cubin，不经 `gen_jit_spec`、无真实源文件 → 对应 F8 → `missed`，与设计一致。

**复核脚本（版本升级时重跑，确认 hook 点未变）**：flashinfer 的 JIT 入口符号随版本可能变，升级后用下方脚本重跑一遍即可（首选内省 installed 包，不 clone——它是纯 Python、逐字打进 wheel，版本精确；resolve-third-party 的 cache clone 仅作交叉核对）。

  ```python
  # probe_jit_entry.py — 找 flashinfer / deep_gemm 的 JIT 入口符号 + sources 参数位
  import importlib, importlib.metadata as md, pathlib, re
  NAME_PAT = re.compile(r'(jit|spec|build|compile|load_cuda|gen_|nvrtc|nvcc|runtime)', re.I)
  SRC_PAT  = re.compile(r'\b(sources|cuda_files|cpp_files|source_paths|srcs)\b')
  def dump(imp, dist):
      print("="*70); print(f"{imp}  (dist={dist})")
      try: print("version:", md.version(dist))
      except Exception as e: print("version: ERR", e)
      try: mod = importlib.import_module(imp)
      except Exception as e: print("import ERR:", e); return
      root = pathlib.Path(mod.__file__).parent; print("path:", root)
      jit_files = [p for p in root.rglob("*.py") if NAME_PAT.search(str(p.relative_to(root)))]
      print("\n-- JIT-ish files --")
      for p in jit_files: print("  ", p.relative_to(root))
      print("\n-- def/class(带 jit/spec/build 名) + 任何提到 sources 的行 --")
      for p in sorted(set(jit_files) | set(root.glob("*.py"))):
          try: lines = p.read_text(errors="ignore").splitlines()
          except Exception: continue
          for i, ln in enumerate(lines, 1):
              s = ln.strip()
              if (s.startswith(("def ", "class ")) and NAME_PAT.search(s)) or SRC_PAT.search(ln):
                  print(f"  {p.relative_to(root)}:{i}: {s[:120]}")
  dump("flashinfer", "flashinfer_python"); dump("deep_gemm", "sgl-deep-gemm")
  ```
  跑法：`/path/to/sglang-python probe_jit_entry.py 2>&1 | tee jit_probe.txt`。

**第 2 级（写 patch 时做）——确认真被调用 + 抓到绝对路径**：在一次 KID run 里对 `gen_jit_spec` 加临时 print，确认跑 GDN flashinfer 路径时 `gen_gdn_prefill_*_module` → `gen_jit_spec` 命中、`sources` 求值后为绝对路径。依赖第 1 级（已完成）。

**静态兜底仍保留**：patch 是主路径；locate 的静态 `gen_jit_spec` 解析（§5.4 机制②）作为**兜底**继续存在，覆盖「patch 未命中某个新入口」「只有 clone、没跑 KID 就想定位」等情形。即**运行时白送为主、静态为辅**。

### 4.6 输出

```
<output_dir>/
  decomposition_<backend>.schema.json      # 每条 backend 一份
  profile.nsys-rep / profile.sqlite
  events/events_<pid>.jsonl                # 运行时 event 原始记录
  service.log / test.log
```

**每个 selected kernel 的 schema entry**（KID 产出，**不含** `source_locations`）：

```json
{
  "rank": 1,
  "kernel": { "raw_name": "...", "normalized_name": "...", "category": "sgl_kernel" },
  "archetype": "F2|F3",
  "metrics": { "duration_us": 123.4, "share_in_invocation": 0.21 },
  "interface": "torch.ops.sgl_kernel.fwd",
  "runtime_event": {
    "wrapper": { "api": "...flash_attn", "file": "/abs/....py", "line": 88,
                 "category": "sgl_kernel", "stage": "prefill", "forward_mode": "..." },
    "implementation": {
      "kind": "sgl_kernel_source | runtime_jit_source | triton_source | pytorch_native | null",
      "source_files": ["/abs/.../csrc/xxx.cu"],   // JIT: 真实; triton: [kernel_file]; sgl_kernel AOT: []
      "symbols": ["mha_fwd"], "export_name": "fwd", "definition_line": 45, "compile_flags": {}
    },
    "attribution": { "method": "cuda_correlation_id+nvtx", "confidence": "high" }
  }
}
```

**三个「位置」别混（wrapper 白送两个，schema 里各就各位）**：运行时其实抓到三个不同的代码位置，实现/消费时要分清：

| 位置 | 含义 | schema 字段 | 白送情况 | 对应四层 |
| --- | --- | --- | --- | --- |
| **① launch 点** | `xxx[grid](...)` / `torch.ops...()` 这行调用 | `runtime_event.wrapper.file/line` | **所有类型都白送**（运行时调用栈 [`_caller_location`](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py#L384-L393)） | 层 a 的**锚点**（收窄到 launch 语句处），非层 a 本身 |
| **② kernel 定义** | `@triton.jit def` / JIT 的 `.cu` / AOT symbol | `runtime_event.implementation.source_files + definition_line` | triton/JIT **白送**；**sgl_kernel AOT 为空**（留 locate 静态补） | 层 b |
| **③ target 接口** | 用户指定的那个函数（如 `_layer_norm_fwd`） | schema 顶层 `target` 段（非本 entry 内） | target wrapper 白送 | 层 a 主体 |

> 关键区分：**「给 launch 行为加 wrapper」天然把 ①（launch 点）在运行时确定了**（调用栈白送），并顺带白送 ②（triton/JIT 类的 kernel 定义）；唯一例外是 sgl_kernel AOT，其 ② 运行时为空。① 是层 a 的锚点、③ 才是层 a 主体——实现 locate 时别把 ① 当成层 a 全部。

`interface` = 运行时抓到的接口名（`torch.ops.sgl_kernel.<op>` / triton fn 名 / `get_xxx_module().<op>`）。`runtime_event.implementation` 可能为 null（如纯 F0）或 `source_files=[]`（如 F2/F3 AOT）——这正是 locate 要静态补的部分。

---

## 5. locate-kernel-source 设计（Step 0.5 第二个 skill）

### 5.1 类型与位置

**三层**：Layer 1 deterministic 定位（CLI 跑一轮）+ Layer 2 agent 兜底（读代码补歧义层）+ **Layer 3 物料抽取**（按定位结果把四层源码抽成文件，即原 `import-decomposition`）。前两层产「位置」，Layer 3 按位置落盘物料。三层共用一个包。

**新包 `framework_engineer/source_location/`**（与 `third_party_solver/` 平级；只读后者产出的 manifest JSON，无状态耦合）：

```
framework_engineer/source_location/
  contracts.py     # LayerHit / LayerResult / LayerResolution 数据类
  archetype.py     # F2/F3 精化 + 形态→分派规则
  registry_probe.py# 机制①：sgl-kernel m.impl 符号注册表（从 KID 移入）
  symbol_grep.py   # 符号/def 在源码树/clone 里 grep（从 KID 移入）
  jit_sources.py   # 机制②：runtime_event.source_files 按后缀分层 + 静态 gen_jit_spec 兜底
  locator.py       # 单入口 locate_kernel_source(...)，内部按 archetype 分派（Layer 1）
  extractor.py     # Layer 3：按 source_locations 抽四层文件 + read_hints.txt（原 import-decomposition）
  cli.py           # 子命令 locate（Layer1）/ extract（Layer3）批量驱动
  __main__.py
```

skill 文档：`framework_engineer/skills/locate_kernel_source.md`（驱动 Layer 2 agent）。

> 与 §8.1.2 的差异：原文把 helper 放 `third_party_solver/source_locator.py`。本文改为**独立包 `source_location/`**——它逻辑上属于「定位」而非「拉三方库」，独立更清晰；它只消费 `third_party_solver` 输出的 manifest JSON，不 import 其代码。
>
> **`import-decomposition` 归属变更（2026-07-13）**：框架文档 §8.2 原把 `import-decomposition` 挂在 Step 1(KID) 名下，但它的输入（四层 `source_locations`）**完全来自 locate、不是 KID 产的**——消费者被编在生产者之前，不自洽。故 V2 把它收进 locate 作 **Layer 3 抽取阶段**（`extractor.py`）。KID 因此彻底纯化为 profiler，不碰任何源码抽取落盘。注意：这里指的是 §8.2 的 `import-decomposition`（抽到 `workspace/kernel_sources/`）；§8.3.3 的 `import-kernel-sources-to-taskpack`（把 kernel_sources 拷进 task_pack + 生成 `level_decision.yaml`）**仍属 Step 2**，不动。

### 5.2 输入

| 来源 | 输入 | 说明 |
| --- | --- | --- |
| **KID** | `decomposition_<backend>.schema.json`（每 kernel 的 `interface` + `archetype` + `runtime_event`） | 主输入 |
| **resolve-third-party** | `third_party_manifest.json`（`name → local_path` / url / ref / status） | 提供 clone 路径，用于跨仓（F3/F5/F7） |
| **用户** | `sglang_repo_root` | sgl-kernel 符号表 + sglang 树内定位 |

**不需要** `external_references`（那是 Step 3 `problem_translate` 的输入，跟定位无关）。

### 5.3 四层信息 + 三层架构

四层（KID 需要的定位目标，供 Layer 3 抽取）：
- **a. `interface_definition`**：接口定义（python），含 kernel launch 语句
- **b. `kernel_impl`**：kernel 实现（`.cu`/`.cpp`/`.py`）
- **c. `py_cpp_binding`**：py↔cpp 绑定（主要 sgl-kernel）
- **d. `kernel_header`**：头文件 `.h/.cuh`

**三层架构**：
- **Layer 1 — deterministic 定位（CLI）**：对每个 kernel 按 `archetype` 分派，**能定就定，定不了也先跑一遍**，把每层标 `resolved / not_applicable / ambiguous / not_found`。歧义给多候选，失败给 `repo_hint`（manifest 里该库仓库根），**不写失败原因**（helper 是固定逻辑，做不好灵活搜索）。
- **Layer 2 — agent 兜底**：读 Layer 1 结果，**只对 `ambiguous`/`not_found` 的层**，拿 `repo_hint` 的仓库根主动去找；补齐或标 `missed`。
- **Layer 3 — 物料抽取（CLI，原 `import-decomposition`）**：定位齐了之后，按 `source_locations` 把四层源码抽成 `kernel_sources/<id>/` 文件 + `read_hints.txt`（详见 §5.6 末）。

**信息源优先级**（Layer 1 内部）：**先消费 `runtime_event`，缺了才静态解析**（机制①/②）。这段就是从 KID `_implementation_from_events` 搬来的逻辑：
- `runtime_event.implementation.source_files` 非空（F4/F7 JIT、F1/F6 triton）→ 层 b/c/d 几乎白送，按后缀分（`.cu/.cpp`→b、`*_jit_binding.cu`→c、`.cuh/.h`→d）。
- `runtime_event.wrapper.file/line` → 层 a 白送。
- `source_files` 为空（F2/F3 AOT）→ 走机制①静态符号解析。

### 5.4 按形态分派（分档）

| 形态 | 层 a | 层 b | 层 c | 层 d | 主要机制 | determinism 档 |
| --- | --- | --- | --- | --- | --- | --- |
| F0 | 调用点 | not_applicable | n/a | n/a | 停在 API | 档1 全确定 |
| F1 | wrapper（runtime_event 白送） | triton fn（runtime file/line 白送；否则 grep `def`） | n/a | n/a | runtime_event / grep | 档1~2 |
| F2 | `sgl_kernel/*.py` | `sgl-kernel/csrc/*.cu`（机制①符号 grep） | `*_extension.cc` 的 `m.impl` | 同目录 `.h/.cuh` | 机制① | 档1 |
| F3 | 同 F2 | **clone 内** impl（符号命中 fetch 仓库） | `*_extension.cc` | `include/*ops.h` | 机制①→跨仓 | 档2（层b 可能 ambiguous） |
| F4 | sglang `jit_kernel/*.py` | `sources[]` 的 `.cu` | `sources[]` 的 `*_jit_binding.cu` | `sources[]` 的 `.cuh` | 机制②（runtime_event） | 档1 |
| F5 | 三方 `*.py` | clone `csrc/*.cu` | 三方 pybind | clone `*.h` | pybind 溯源 | 档2 |
| F6 | 三方 `*.py` | 同文件 DSL fn（runtime file/line 白送） | n/a | n/a | runtime_event / grep | 档1~2 |
| F7 | 三方 `*.py` | `sources[]` `.cu`（runtime_event，或静态 `gen_jit_spec`） | `sources[]` binding | `sources[]` `.cuh` | 机制② | 档1（有 runtime）/档2（静态兜底） |
| F8 | 三方 `*.py` | **FAILED 无源** | n/a | n/a | deny-list | 档3 |

**机制①（F2/F3，从 KID `_sgl_kernel_registry`+`_find_symbol_sources` 移入）**：

```
接口 torch.ops.sgl_kernel.<op>
 → csrc/*_extension.cc 找 m.def("<op>",...) + m.impl("<op>", &<symbol>)     # 层 c
 → grep <symbol> 定义：
     命中 sgl-kernel/csrc/*.cu           → F2，就地（层 b；层 d 同目录 .h）
     命中被 fetch 的仓库（对照 CMake）    → F3，去 manifest 里该库 clone 定位（层 b/d）
```
实例：`torch.ops.sgl_kernel.fwd` → `flash_extension.cc` 的 `m.impl("fwd", &mha_fwd)`（c）→ `mha_fwd` 在 clone 的 sgl-attn `hopper/flash_api.cpp`（b）→ `sgl_flash_kernel_ops.h`（d）；wrapper `flash_attn.py`（a）。**这里就是 F2/F3 的最终判定点。**

**机制②（F4/F7）**：优先读 `runtime_event.implementation.source_files`（KID 白送）；若为空，走静态兜底。两种三方库布局不同，静态解析要分开处理：
- **flashinfer**（有 patch，静态仅兜底）：找 `gen_xxx_module()` → `gen_jit_spec(uri, source_paths)`，源锚点 `FLASHINFER_CSRC_DIR = <clone>/csrc`；注意 attention/gemm 会把**模板生成的 `.cu`** append 进 `source_paths`（生成文件静态解析拿不全，正是 patch 白送的价值）。
- **deep_gemm**（无 patch，静态为主）：入口 `deep_gemm/__init__.py:_build_module`，binding 是写死的 `csrc/tvm_ffi_api.cpp`（层 c），真实 kernel 是 `include/**` 的 C++ 模板经 NVRTC 运行时编译（层 b/d）——布局是 **TVM-FFI + include 模板**，不是 `gen_jit_spec`。源集每版本固定，直接按 `<clone>/csrc` + `<clone>/include` 定位。

**永远忽略 `~/.cache/.../cached_ops` 编译产物**。跨包混合 sources 时逐层分别定状态，落到下载 artifact 的那层 → F8 → `missed`。

### 5.5 helper 接口契约（`locator.py`）

```python
def locate_kernel_source(
    interface: str,
    archetype: str,                 # KID 给的，可能是 "F2|F3" provisional
    *,
    runtime_event: dict | None,     # KID 序列化的运行时事实（可空）
    manifest: dict,                 # third_party_manifest.json 解析后：name -> {local_path, url, ...}
    sglang_repo_root: Path,
) -> LayerResolution

@dataclass
class LayerHit:
    file: str; line_start: int | None; line_end: int | None

@dataclass
class LayerResult:
    status: Literal["resolved", "not_applicable", "ambiguous", "not_found", "missed"]
    hits: list[LayerHit]            # resolved=1；ambiguous=多候选；not_found=[]
    repo_hint: str | None           # 失败时给 manifest 里该库仓库根，交 agent
    source: Literal["locate_layer1", "locate_layer2_agent", "manual", "dry_run"]
                                    # 最后更新“这一层”的角色（逐层溯源）

@dataclass
class LayerResolution:
    interface: str
    archetype: str                  # 已 finalize（F2/F3 已判定）
    source: str                     # 派生聚合：任一层被 agent/人工动过则取之，否则 layer1
    layers: dict[str, LayerResult]  # a/b/c/d 四层各一
    needs_agent: bool               # 任一必填层 != resolved/not_applicable → True
```

单入口，内部按 `archetype` 分派；`F2|F3` 在此 finalize。每层四状态独立，`not_applicable` 是形态决定的合法 null，不算失败。

> **`source` 的语义（逐层溯源）**：`source` = **最后更新该层的角色**，落在**每个 layer** 上。Layer 1 CLI 定到的层记 `locate_layer1`；Layer 2 agent **改过**的层更新为 `locate_layer2_agent`，**没动**的层保持 `locate_layer1`；人工手填记 `manual`；dry-run 骨架记 `dry_run`。顶层 `source_locations.source` 是**派生聚合**（优先级 `locate_layer2_agent` > `manual` > `locate_layer1` > `dry_run`），一眼看出这个 kernel 有没有被 agent/人工动过；权威信息在每层。

### 5.6 workflow + CLI 接口

**Layer 1 CLI**（deterministic，批量）：

```bash
python -m framework_engineer.source_location.cli locate \
    --schema <decomposition_*.schema.json>            # 单份；或 --workspace <dir> 处理该目录下全部
    --manifest <third_party_manifest.json> \
    --sglang-repo-root <path> \
    [--out <dir>]                                      # 默认写回 schema 同目录
```

对 schema 里每个 kernel 调 `locate_kernel_source(...)`，产出：
- **就地把 `source_locations` + finalize 后的 `archetype` + `needs_agent` 写回该 kernel 的 schema entry**（Layer 3 抽取只读这一份文件）。
- 另写 `locate_report.json`：汇总 `{total, resolved, needs_agent:[{interface, archetype, layer, repo_hint}]}`，供 Layer 2 agent 定位要处理的项。
- 进度到 stderr、JSON 小结到 stdout（沿用 resolve-third-party 的 stdout 纯净约定）。

**Layer 2 agent**（skill `locate_kernel_source.md` 驱动，独立上下文）：
1. 读 `locate_report.json` 的 `needs_agent` 列表 + 对应 schema entry。
2. 对每个 `ambiguous`/`not_found` 层，用 `repo_hint` 的仓库根**主动 grep/读代码**找源。
3. 找到 → 更新 schema entry 该层为 `resolved`+hits；仍找不到 → 标 `missed`。
4. 追加 `locate_agent_notes.md` 记录每个兜底项的结论 + 理由。

**Layer 3 CLI — 物料抽取（原 `import-decomposition`）**：定位（Layer 1/2）跑完、schema 里 `source_locations` 齐了之后执行。

```bash
python -m framework_engineer.source_location.cli extract \
    --schema <decomposition_*.schema.json>            # 已被 locate 富化过（含 source_locations）
    --workspace-out <dir>                              # workspace 目标目录
```

`extractor.py` 遍历 schema 每个 kernel，按 `source_locations.layers.<layer>.hits[0]` 的 file + 行号范围，把四层源码抽成文件写 `<workspace>/kernel_sources/<id>/{interface_definition,py_cpp_binding,kernel_header,kernel_impl}.{py,cc,h,cu,cpp}` + `read_hints.txt`，并回填 `kernel_sources_dir` 到 schema。规则：
- `not_applicable` 层 → 建空文件 + 注释「该层形态不适用（如 triton 无 py↔cpp binding）」。
- `missed`/`not_found` 层 → 写占位 + 注释「该层未定位，见 locate_agent_notes.md」，**不阻断**，交人工在 workspace 补。
- 源文件超大时按行号范围前后加 padding 抽取，避免体积膨胀（沿用 §8.2 约束）。

> 为什么归 locate：抽取要懂「四层语义 + null 规则 + missed 处理」，这套规则本就在 locate 手里；且输入 `source_locations` 是 locate 刚写回的，同包零跳步。KID 不懂四层，不该承担。

### 5.7 输出 + 工作目录状态

Layer 1 之后：
```
<schema 同目录>/
  decomposition_<backend>.schema.json    # 每 kernel entry 新增 source_locations + finalize archetype + needs_agent
  locate_report.json                     # needs_agent 列表 + 统计
```
Layer 2（agent）之后：
```
  decomposition_<backend>.schema.json    # ambiguous/not_found 层被补齐或标 missed
  locate_agent_notes.md                  # agent 兜底过程 + 每个 missed 的接口/形态/repo_hint
```
Layer 3（extract）之后：
```
  <workspace>/
    decomposition_<backend>.schema.json  # 回填 kernel_sources_dir
    kernel_sources/<id>/                 # 四层源码文件（not_applicable/missed 为占位+注释）
      interface_definition.py
      py_cpp_binding.cc                  # 或空文件 + 注释
      kernel_header.h                    # 或空文件 + 注释
      kernel_impl.{py,cu,cpp}
      read_hints.txt                     # 每层 read 行数范围
```

`source_locations` 写进 schema 的形状（每 kernel entry 新增）：

```json
"source_locations": {
  "archetype": "F3",
  "source": "locate_layer2_agent",
  "needs_agent": false,
  "layers": {
    "interface_definition": { "status": "resolved", "hits": [{"file":"...flash_attn.py","line_start":40,"line_end":95}], "repo_hint": null, "source": "locate_layer1" },
    "kernel_impl":          { "status": "resolved", "hits": [{"file":".../sgl-attn/hopper/flash_api.cpp","line_start":300,"line_end":520}], "repo_hint": null, "source": "locate_layer2_agent" },
    "py_cpp_binding":       { "status": "resolved", "hits": [{"file":".../csrc/flash_extension.cc","line_start":50,"line_end":80}], "repo_hint": null, "source": "locate_layer1" },
    "kernel_header":        { "status": "resolved", "hits": [{"file":".../include/sgl_flash_kernel_ops.h","line_start":1,"line_end":60}], "repo_hint": null, "source": "locate_layer1" }
  }
}
```

> 上例：层 a/c/d 由 Layer 1 CLI 定到（`locate_layer1`），层 b 是 Layer 2 agent 补的（`locate_layer2_agent`）→ 顶层派生聚合取 `locate_layer2_agent`。

### 5.8 约束

- `interface_definition` / `kernel_impl` 不允许最终 null；`py_cpp_binding` / `kernel_header` 仅 F1/F4/F6/F7 的 DSL/triton 情形可 `not_applicable`。
- 两层都失败的必填层最终标 `missed`，报接口名 + 形态 + repo_hint，交人工。
- **不改运行环境**，不重编，不装包；只读源码/clone。
- **依赖**：`resolve-third-party`（manifest 里的 clone 路径）+ KID（schema 里的 runtime_event）。

---

## 6. KID ↔ locate ↔ phase1 的对接契约

```
┌────────────────────────────────────────────────────────────────────────────┐
│ resolve-third-party (Step 0.5-a, 已完成, CLI)                                │
│   → third_party_manifest.json  (name → local_path / url / ref / status)      │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │ manifest（只给 locate，不给 KID）
                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ KID (Step 1, CLI, 无 agent)                                                  │
│   输入: service_cmds[](各+--disable-cuda-graph) / test_cmd(静态) / target /  │
│         sglang_repo_root                                                     │
│   做:   profile → 取代表 invocation(1 prefill+1 decode) → 选热点             │
│         → 形态族分类 → 附 runtime_event                                       │
│   出:   decomposition_<backend>.schema.json                                  │
│         每 kernel: interface + archetype(可 F2|F3) + runtime_event（无 4 层） │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │ schema（interface + archetype + runtime_event）
                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ locate-kernel-source (Step 0.5-b, 三层)                                      │
│   Layer 1 CLI(locate): 读 schema + manifest + sglang_repo_root               │
│                → 每 kernel 补 source_locations + finalize archetype + needs_agent
│                → locate_report.json                                          │
│   Layer 2 agent:  对 needs_agent 的层用 repo_hint 兜底 → 补齐 / 标 missed     │
│   Layer 3 CLI(extract, 原 import-decomposition):                             │
│                按 source_locations 抽 4 层文件 → workspace/kernel_sources/<id>/│
│                + read_hints.txt；回填 kernel_sources_dir 到 schema            │
└───────────────┬──────────────────────────────────────────────────────────────┘
                          ▼  之后接已开发好的 phase1 主链路（scaffold → ... → validate）
```

**契约要点**：
1. KID schema 的 `source_locations` 字段**由 locate（Layer 1/2）填**，KID 留空/不产。
2. Layer 3 抽取（原 `import-decomposition`）只在定位跑完后消费；它读 `source_locations.layers.<layer>.hits[0]` 给出 file + 行号范围。它已收进 locate 包（`extractor.py`），不再是 KID/Step 1 的一部分。
3. `missed` 层：Layer 3 生成对应文件时写占位 + 注释（"该层未定位，见 locate_agent_notes.md"），不阻断，交人工在 workspace 里补（对应框架文档 §2.3「补充 KID 抽不干净的 helper 文件」）。
4. `archetype` 在 KID 是 provisional（可 `F2|F3`），进 workspace/task_pack 的是 locate finalize 后的值。

---

## 7. 落地改造清单

### 7.1 KID（`framework_engineer/kernel_interface_decomposer/`）

> **顺序约束：本轮开发前先补 KID 插桩 UT**（见 §8「KID 插桩首验」）。现状 KID 无任何测试，插桩正确性从未端到端验证过；先用轻量 UT 锁住"各类算子 launch 捕获成功"，再动下面的重构。

- [ ] **（先做）插桩 UT**：见 §8，各形态各一个典型算子，验证 launch wrapper 捕获成功（打印/断言 `target_wrapped` + 各类 `wrap` event 出现）。
- [ ] [config.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/config.py)：`service_cmd` → `service_cmds: list[{backend_name, cmd}]`；**取消 `target_kind`**（统一 high_level_target，见 §4.1）。移除对 `third_party_prefixes` 之外任何三方路径依赖。
- [ ] [runner.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runner.py)：多 backend 循环，每条产 `decomposition_<backend>.schema.json`；**对每条 cmd 强制追加 `--disable-cuda-graph`（已存在则跳过）**（§4.2 约束 1）。
- [ ] [trace_parser.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/trace_parser.py)：(a) `_record_to_schema` 不再 resolve implementation，改为写 `interface` + `archetype` + `runtime_event`；(b) **新增「代表性 invocation」选取**：按 stage 只保留第一次 prefill + 第一次 decode invocation（配合 `skip_target_invocations` 跳 warmup），替代当前"输出全部 invocation"（§4.4）。
- [ ] [source_resolver.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/source_resolver.py)：**收缩**——保留 `locate_function_at` + `_infer_category`（扩成 §4.3 的形态族分类，产 `archetype`，sgl_kernel 产 `F2|F3`）；**删除** `_resolve_implementation` 静态分支、`_resolve_triton`/`_find_triton_definition`、`_resolve_sgl_kernel`/`_sgl_kernel_registry`/`_find_symbol_sources`、`_implementation_from_events`（这些移到 `source_location/`）。
- [ ] [runtime_instrumentation.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/runtime_instrumentation.py)：**增补 `_patch_flashinfer_jit`**（内存 monkey-patch，同现有 `load_jit` 插桩，不改 installed 文件），hook 点已实测确认为 **`flashinfer.jit.core.gen_jit_spec(name, sources, ...)`**（`sources` = 第 2 位置参），单点覆盖整个 F7（含 GDN）。**deep_gemm 不 patch**（无 sources 参数可 hook，源集固定，走 locate 静态）。见 §4.5。
- [ ] [README.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/kernel_interface_decomposer/README.md)：说明「KID 不再输出 source_locations，交 locate」+「统一 high_level_target、需 eager 模式」。

### 7.2 locate-kernel-source（新包 `framework_engineer/source_location/`）

- [ ] `locator.py`：单入口 `locate_kernel_source(...)`，内部按 archetype 分派 + runtime_event 优先。
- [x] `extractor.py`：**Layer 3 物料抽取（原 `import-decomposition`）已实现**——按 `source_locations` 抽四层文件到 `workspace/kernel_sources/<id>/` + `read_hints.txt` + 回填 `kernel_sources_dir`；`not_applicable`/`missed` 写占位注释，`missed` 必填层硬停（`--allow-empty` 放行）。见 [source_location/extractor.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/source_location/extractor.py)、单测 [test_source_location_extract.py](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/tests/test_source_location_extract.py)。
- [x] `cli.py`(extract 子命令) + `__main__.py`：`python -m framework_engineer.source_location.cli extract`（`locate` 子命令为未实现 stub）。
- [ ] `contracts.py`：`LayerHit/LayerResult` 读取侧已随 extractor 落地；`LayerResolution`（生产侧）待 locator 补。
- [ ] `registry_probe.py` / `symbol_grep.py` / `jit_sources.py` / `archetype.py` / 完整 `locator.py`：locate 定位层，后续开发。
- [ ] `skills/locate_kernel_source.md`：Layer 2 agent 兜底（读 report、按 repo_hint 找、补齐/标 missed、写 notes）。

> **Dry-run 验证机制（已实现）**：`framework_engineer/dry_run/`（`kid`/`locate`/`extract` 三子命令）在无 GPU/无 profiling 下验证「人工介入后交付链跑通」——KID/locate 出骨架（只留 agent 定位不到的 file/line 占位），extract passthrough 调真实 L3。见 [dry_run/](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/framework_engineer/dry_run)、[dry_run_and_layer3_cli_plan.md](file:///Users/bytedance/Desktop/infra_agent/.trae/documents/dry_run_and_layer3_cli_plan.md)。archetype 产物用明文类别名（`sglang_triton` 等）+ `archetype_code`（F*）。

### 7.3 对接

- [ ] **`import-decomposition` 归属迁移**：从框架文档 §8.2（Step 1/KID 名下）移除，落为 locate 的 Layer 3（`extractor.py`）。消费 locate 富化后的 `source_locations.layers.<x>.hits`；`missed`/`not_found` 层写占位 + 注释，不阻断。
- [ ] 保持不动：§8.3.3 `import-kernel-sources-to-taskpack`（Step 2 的 task_pack 组装，与本迁移无关）。
- [ ] 更新 [framework_engineer_design_v2.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/kernel_agent_kadai/framework_engineer_design_v2.md) §8.1.2/§8.2 指向本文（helper 路径改 `source_location/`、KID 不再产 source_locations、KID 不吃 manifest、`import-decomposition` 归 locate Layer 3）。

---

## 8. 测试

**KID 插桩首验（本轮开发前先做，§7.1 顺序约束）**：轻量 UT，不追求覆盖率，核心验证「各类算子的 launch 捕获成功」——这是整套归因的地基，现状从未验证过。做法：各形态各挑一个典型算子，跑一次最小 profile（或直接对 wrapper 单元测），检查 events/schema：

| 形态 | 典型算子（GDN 场景可取） | 验证点 |
| --- | --- | --- |
| target wrap | `_layer_norm_fwd`（[layernorm_gated.py:205](file:///Users/bytedance/Desktop/infra_agent/sglang/python/sglang/srt/layers/attention/fla/layernorm_gated.py#L205)） | events 出现 `target_wrapped`（非 `target_wrap_failed`） |
| F1 triton | `_layer_norm_fwd_1pass_kernel`（[:68](file:///Users/bytedance/Desktop/infra_agent/sglang/python/sglang/srt/layers/attention/fla/layernorm_gated.py#L68)） | 出现 `wrap`(category=triton_dsl)，implementation 带 `source_files`+`definition_line` |
| F0 torch | `torch.nn.functional.linear` 等 | 出现 `wrap`(category=pytorch_native) |
| F2/F3 sgl_kernel | 某 `torch.ops.sgl_kernel.*` | 出现 `wrap`(category=sgl_kernel)，`source_files=[]`（AOT） |
| F7 flashinfer JIT | GDN flashinfer 路径某 op | patch `gen_jit_spec` 后出现 `jit_module_loaded`/wrap，`source_files` 是真实 `.cu` 绝对路径 |

> **首个最小 smoke**：就用你正看的 `_layer_norm_fwd`（F1 单 kernel）——若它能正确产出 target range + triton wrap event + 缝出该 1 个 kernel，机制即立住；再上 GDN 全 forward。已知脆点重点看：target hook 时机（是否被提前 import/拷贝引用绕过）、triton `__getitem__` patch（新版 triton 是否走别的启动路径）、`attribution.method` 是否为 `cuda_correlation_id+nvtx`（非退化的 `nvtx_time_containment`）。

**KID 功能测试**：
1. 统一入口（GDN 三 backend triton/flashinfer/cutedsl）：产 3 份 schema，每 kernel 有 `interface`+`archetype`+`runtime_event`，**无** `source_locations`。
2. 代表性 invocation：静态脚本一次 prefill + 若干 decode，验证 schema 里 target 只留「一次 prefill + 一次 decode」两组热点，warmup/JIT 期不计入。
3. 单 target 退化（用户以为的 low_level，如直接指 `_layer_norm_fwd`）：走同一套逻辑，缝出的 kernel 恰为少量（可能就 1 个），**无需任何特殊分支**。
4. 形态族分类正确：F0/F1/F4/F6/F7/F8 运行时可判；sgl_kernel op 标 `F2|F3`。
5. cuda-graph 约束：验证 runner 对缺 `--disable-cuda-graph` 的 cmd 自动追加。

**locate**：
1. F3：`torch.ops.sgl_kernel.fwd` 四层齐全，层 b 落在 sgl-attn clone，archetype finalize 成 F3。
2. F7：`chunk_gated_delta_rule`（flashinfer C++ JIT）→ 层 b 命中 flashinfer `csrc/*.cu`（有 runtime_event 走白送；无则静态 `gen_jit_spec`）；换 cuteDSL 版 → 层 b `.py`、c/d `not_applicable`。
3. F2：`causal_conv1d` 层 b/d 在 `sgl-kernel/csrc/mamba/`，无需跨仓，finalize F2。
4. F8：下载 cubin 的 op → 层 b `not_found` 无 repo_hint → `missed`。
5. Layer 2：人为把某 F3 层 b 制造成 `ambiguous`（多实例化命中），验证 agent 用 repo_hint 收敛到唯一 hit。

6. Layer 3 抽取：给一份富化过的 schema，`extract` 后 `kernel_sources/<id>/` 四层文件 + `read_hints.txt` 生成正确；`not_applicable` 层为空文件+注释、`missed` 层为占位+注释。

**端到端**：resolve-third-party → KID → locate(locate→extract) 跑通 GDN，workspace/kernel_sources/<id>/ 四层文件 + read_hints.txt 正确。

---

## 9. 一句话总结

**KID 是「运行时观测 + 形态分类」的纯 CLI，产 `(interface, archetype, runtime_event)`；locate 是「静态四层定位 + 物料抽取」的三层（Layer1 定位 / Layer2 agent 兜底 / Layer3 抽取，原 `import-decomposition` 收于此），消费 KID schema + manifest 产 `source_locations` 并落盘 `kernel_sources/`。二者分离两遍、共享 helper，重叠的静态溯源逻辑从 KID 全部搬进 locate。沿「运行时只 KID 能看见、静态只需磁盘」这条缝切，换来重跑不必重 profile、agent 兜底后置独立、KID 不碰三方布局三个干净收益。**

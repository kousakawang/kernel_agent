# KID 与 source_locate 联合设计 V2

> 本文是 KID（Kernel Interface Decomposer）与 source_locate 的当前权威设计。
> 它覆盖 KID 的 execution capture、semantic target 识别、Nsight Systems 归因，以及
> source_locate 的自主 Agent 工作流。旧版的 F0–F8 源码形态分派、`binding_provider`
> 注册表和 locate Layer 1/2/3 流水线均已废弃，见 §8。
>
> 最后更新：2026-07-19

---

## 0. 核心结论

1. **KID 的运行时捕获基于 execution common interface**：针对 PyTorch、Triton、
   sgl-kernel、JIT/FFI 和需要单独适配的 third-party execution 入口做统一 capture。
2. **execution target 不等于 semantic low-level target**。capture 时保存从用户指定的
   `high_level_target` 到 common interface 的 Python 调用链，随后由 KID 内的
   **Semantic Resolver Agent** 阅读调用链和源码，选出真正的 semantic target。
3. **NVTX 负责划定 high-level invocation，CUDA correlation 负责把 CPU launch 连接到
   GPU kernel**。耗时使用关联 GPU kernel duration 之和，而不是 NVTX CPU range 时长。
4. **`archetype` 只表示 execution 被哪类 capture mechanism 捕获**，不再描述源码交付
   形态，也不再驱动 source locate。
5. **`provider` 是可选的具体包/仓库名称**，能可靠识别就填，例如 `flashinfer`、
   `deepgemm`、`sgl-attn`；不能识别就留空。它不是 binding locator 的注册 key。
6. **最终 schema 保持精简**。完整调用栈、candidate、execution id、选择理由等只存在于
   KID 原始 event/内部工作数据中，不进入下游 schema。
7. **source_locate 是一个自主 Agent，不再分 Layer 1/2/3**。interface locator、源码
   extract、validator 和符号搜索均只是 Agent 可调用的确定性 CLI/工具。
8. **source_locate Agent 独立负责四层定位**：`interface_definition`、`kernel_impl`、
   `py_cpp_binding`、`kernel_header`。它不依赖 `archetype/provider` 做规则分派。

---

## 1. 术语与职责边界

### 1.1 high-level target

用户显式提供、希望进行运行时拆解的 Python 接口。KID 对它打 NVTX range，并为每次调用
分配 `high_call_id`。一次模型运行中可以出现多次 high-level invocation。

### 1.2 execution target

运行时可以在共通入口稳定捕获的执行行为，例如：

- PyTorch Dispatcher/operator 调用；
- Triton/DSL kernel launch；
- `torch.ops` custom op；
- sglang/third-party JIT 或 FFI launch；
- 没有共通协议的 third-party adapter 所捕获的执行入口。

execution target 是 GPU kernel 归因的锚点，但不一定是最终 task_pack 的优化接口。

### 1.3 semantic low-level target

最终用于以下工作的 Python 接口：

- 热点排序；
- 输入输出 snapshot；
- problem_translate；
- task_pack 边界；
- Kernel Engineer 的替换 ABI。

它应具有清晰语义、稳定调用方式和可 snapshot 的输入输出。一个 semantic target 下面
可能有一个或多个 execution，也可能由一个 execution launch 多个 GPU kernel。

### 1.4 source_locate

在 semantic target 已确定以后，定位其四层源码的自主 Agent。source_locate 不参与
semantic target 选择，也不参与 KID 的耗时归因。

---

## 2. 为什么不能用固定栈深选择 semantic target

下面的调用结构中，从 execution 回溯到 high-level 的直接子节点会得到 `mid_level_interface`：

```text
high_level_target
└── mid_level_interface
    ├── low_level_target_1
    │   └── execution_1
    └── low_level_target_2
        └── execution_2
```

但正确的优化边界可能是 `{low_level_target_1, low_level_target_2}`，而不是
`mid_level_interface`。

即使只有：

```text
high_level_target
└── mid_level_interface
    └── low_level_target
        └── execution
```

也无法仅凭调用树判断应选择 `mid_level_interface` 还是 `low_level_target`。前者可能只是
透明 wrapper，也可能才是具有完整数学语义和 UT 的公开接口。这需要结合源码、接口稳定性、
输入输出、UT/reference 和调用上下文进行语义判断。

因此，调用栈提供的是**候选路径**，Semantic Resolver Agent 选择的是调用树上的
**semantic frontier**，而不是固定层级。

默认判断原则：

- 跳过只做参数透传、alias 或 backend dispatch 的透明 wrapper；
- 跳过循环、控制流和模型编排等 orchestration 层；
- 优先选择具有稳定 Python ABI 和清晰 tensor 输入输出的接口；
- 子接口各自有独立语义、UT/reference 且可替换时，倾向拆开；
- private helper 只有在父接口下才构成完整语义时，选择父接口；
- 多个 execution 共同构成不可分割的 fused operation 时，允许选择共同祖先；
- 无法可靠判断时交人工确认，不通过扩张正式 schema 来容纳不确定性。

---

## 3. KID 最新设计

### 3.1 类型

KID 由两部分组成：

1. **Runtime Capture CLI**：启动服务、执行测试、采集 Nsight trace 和 Python runtime event；
2. **Semantic Resolver Agent**：分析调用链和源码，生成最终 semantic low-level entries。

KID 整体不再是纯 CLI，但 profile、trace 解析、correlation 和候选调用树构造仍然是确定性逻辑。

### 3.2 输入

| 输入 | 说明 |
| --- | --- |
| `backend_name` | 当前单-backend 运行的稳定名称 |
| `cmd` | 可选的服务启动命令；无服务时为 `null` |
| `test_cmd` | 唯一的 high-level target 触发命令，内部自行完成 warmup |
| `target: {file, line, qualified_name?}` | 用户明确指定的 high-level Python 接口 |
| `selection/profiling` | invocation 的 `unique_decomposition/all/last_n/single` 选择和 Nsight 选项 |
| 源码仓库 | Semantic Resolver Agent 阅读 high→execution 调用链所涉及源码 |

KID 不依赖 `third_party_manifest.json` 完成运行时捕获。若 Semantic Resolver Agent 在 provider
兜底时能访问 manifest，可将其作为辅助信息，但不是 capture 的前置条件。

### 3.3 execution capture registry

每类 capture adapter 对一个稳定的 execution common interface 负责。adapter 至少产出：

- execution interface/name；
- capture 类别，即最终 `archetype` 值；
- 当前 high-level call context；
- CPU launch timestamp；
- 可用的 correlation 信息；
- 可选 implementation 线索，例如 Triton kernel file/line、JIT source path；
- 从 high-level frame 到 common interface 的 Python 调用链；
- 能确定时的 provider hint。

capture 类别的具体枚举由
`framework_engineer/kernel_interface_decomposer/CAPTURE_MECHANISMS.md` 定义，代码 registry 位于
`framework_engineer/kernel_interface_decomposer/capture_registry.py`。PoC 验证并固化的当前值为
`pytorch_dispatch`、`triton_launch`、`cute_dsl_launch`、`tilelang_launch`、`tvm_ffi_call`、
`inductor_launch`、`python_binding`。要求是：

- 每个值对应一种真实、可观测的 capture mechanism；
- adapter 在 capture 时即可确定，不需要 source_locate finalize；
- 不再复用旧 F0–F8 的“源码交付形态”含义；
- source_locate 不按该值分派实现。

### 3.4 嵌套 capture

不同 common interface 可能嵌套，例如一个 `torch.ops` implementation 内部再触发 JIT/DSL launch。
runtime instrumentation 保留全部嵌套 capture，并记录 `parent_capture_id`。同一个 GPU kernel 仅
归因给 CUDA launch API 所在的最内层 capture，外层 capture 只保留上下文和可选 inclusive 聚合，
不能重复计入热点总时间。Semantic Resolver Agent 填写 `archetype` 时以最底层有效 capture 为准。

两个顺序执行、且外层 capture 已退出的 common interface 应分别记录。若它们最终属于同一个
semantic target，由 Semantic Resolver Agent 在内部完成分组。

### 3.5 调用链原始数据

每次 execution capture 保存 high→execution 的完整 Python frame 链。每个 frame 至少需要：

- `filename`；
- `qualname/co_name`；
- `co_firstlineno`；
- 父 frame 当前 `f_lineno`，用于恢复这条调用边的 call site。

common interface 若没有 Python frame，则作为 synthetic execution leaf 记录其 API/module 元数据。

这些数据写入 `capture_events/events_<pid>.jsonl`，供 Semantic Resolver Agent 使用。
它们**不进入最终 schema**。

### 3.6 Nsight Systems 归因与耗时

NVTX high-level range 只表示 Python/CPU push-pop 时间，不能直接作为 GPU kernel duration。
正确关联路径为：

```text
high-level NVTX range
  → range 内的 CUDA Runtime/Driver launch
  → process/correlation id
  → GPU kernel activity
```

GPU kernel 即使在 NVTX pop 后才开始执行，也通过 correlation id 归到原 high-level invocation 和
execution event。

耗时定义：

- 单个 kernel：GPU activity 的 `end - start`；
- execution：与该 execution 关联的 kernel duration 之和；
- semantic target：Agent 归入该 target 的 execution/kernel duration 之和；
- share：semantic target GPU sum / high-level invocation 全部 GPU kernel sum。

多 stream 下 duration 之和用于热点比较，不等于端到端 wall time。

### 3.7 Semantic Resolver Agent

Agent 输入：

- high-level target；
- execution events；
- 去重后的动态调用树；
- 与候选 frame 对应的源码；
- 可选的测试、UT 和 provider/repository 信息。

Agent 对每个热点 execution 完成：

1. 在 high→execution 候选路径中选择 semantic target；
2. 必要时将一个中间节点下的多个 execution 拆成多个 semantic target；
3. 必要时将多个 execution 聚合到一个不可分割的 semantic target；
4. 确定 semantic interface 名称；
5. 确定 semantic call site，即选中接口在其父 frame 中被调用的位置；
6. 对 provider 做兜底识别；
7. 将关联 GPU 时间聚合到最终 target 并排序。

Agent 的候选 id、execution id、理由、置信度和替代项均为内部工作数据。正式 schema 只接收最终
结果。极少数混合 provider/多 execution 且无法可靠表达的 case 交人工修改 JSON，不新增
`runtime_components` 等通用结构。

正式实现采用 Prompt Agent + deterministic helper：

```text
prepare → kid-semantic-context/v1
Agent → kid-semantic-decisions/v1 + free-form notes
finalize → kernel-interface-decomposition/v2
validate → publish gate
```

Context 只整理 direct kernel owner、ancestor、完整 stack edge、源码片段/AST call expression
和仓库线索，不携带 semantic oracle。Decisions 必须将每个 direct owner 恰好分配一次，选择的
semantic call site 必须是该 owner 的真实 stack edge；只有 `high|medium` confidence 可发布。
Agent 不手算 duration、share、rank、representative kernel、coverage 或最终 archetype。

### 3.8 `archetype` 与 `provider`

#### archetype

`archetype` = execution capture 类别。它回答：

> 这条 low-level 记录背后的主要 execution 是通过哪一种 common interface 捕获的？

它不再回答：

- 源码属于哪个仓库；
- AOT/JIT 的完整交付形态；
- sgl-kernel built-in 还是 FetchContent third-party；
- source_locate 应走哪套代码分支。

#### provider

`provider` 是可选的自由字符串，表示能可靠识别的具体包或仓库，例如：

```text
flashinfer
deepgemm
sgl-attn
flash-attention
```

识别优先使用 execution module/namespace、kernel source path、Python package 和 repo root。Triton
只能确定 DSL capture、无法可靠确定来源时，由 Agent 尝试补充；仍不确定则省略或写 `null`。

字段名不再使用 `binding_provider`，也不维护 provider→binding locator 注册表。

### 3.9 最终 schema：保持精简

KID schema 每条 entry 表示一个最终 semantic low-level target。保留现有主要结构，不加入调用树
分析细节：

```json
{
  "rank": 1,
  "kernel": {
    "raw_name": "<代表性/最热 GPU kernel>",
    "normalized_name": "..."
  },
  "interface": "sgl_kernel.silu_and_mul",
  "archetype": "<capture category>",
  "provider": null,
  "metrics": {
    "duration_us": 123.4,
    "share_in_invocation": 0.21
  },
  "runtime_event": {
    "call_site": {
      "file": "/abs/path/caller.py",
      "line": 88
    },
    "attribution": {
      "method": "cuda_correlation_id+nvtx",
      "confidence": "high"
    }
  }
}
```

约束：

- `interface` 和 `call_site` 属于 semantic target，不是 common execution interface；
- `archetype` 来自主要 execution 的 capture adapter；
- `provider` 可省略/为空；
- `kernel` 保留代表性 GPU kernel，完整 kernel/correlation 列表留在原始 profiling artifact；
- `metrics.duration_us` 是该 semantic target 关联 GPU kernel duration 的聚合值；
- `runtime_event` 只包含 semantic `call_site` 与运行时归因；源码文件和 symbols 由后续 source_locate 填写；
- schema 不含 `candidate_id`、`execution_ids`、完整 stack、选择理由、`capture_mechanism`、
  `provider_candidate`、`runtime_components`；
- KID 不产 `source_locations` 和 `kernel_sources_dir`。

### 3.10 多 invocation 与多 backend 配置

- 每次 high-level 调用由独立 `high_call_id` 标识；
- warmup/JIT compilation 应在 high-level 外执行，也可通过 `skip_invocations` 额外排除；
- `unique_decomposition` 为默认策略：按最底层 execution owner、嵌套深度、完整调用路径和未归因
  kernel 数量分组，每组保留时间上最后一次；stage、耗时、运行时 ID、kernel 名称/次数不参与分组；
- `all` 保留全部 eligible invocation；`last_n` 对每个 stage 选择末尾 N 次，`single` 等价于末尾 1 次；
- SQLite/JSONL 始终保留所有 invocation，只有 `runtime_capture.schema.json` 收敛为代表调用；
- Semantic Resolver 对多个代表拆分做 semantic target 并集，不由 Runtime CLI 跨 invocation 聚合耗时；
- 并集 identity 只按 semantic `interface`：跨 stage/call site 的同接口合并，最终 call site、
  archetype 和 representative kernel 取最热 contributing member，`sample_count` 记录出现该 target
  的代表 invocation 数；share 分母为全部代表 invocation 的 GPU kernel duration 总和；
- 一份 `kid-runtime-config/v2` 只 profile 一个 backend 和一个 `test_cmd`；多 backend 由上层串行执行多份配置；
- 强制 eager/禁用 CUDA Graph，避免 replay 绕过 Python capture；
- 多 backend 的排序与后续 workspace 选择不改变单次 KID 归因逻辑。

---

## 4. source_locate 最新设计

### 4.1 类型：一个自主 Agent + CLI 工具

source_locate 不再是 Layer 1 CLI → Layer 2 Agent → Layer 3 CLI 的分层流水线。

它是一个自主 Agent，负责从输入到四层定位完成的整个任务。确定性程序只是 Agent 可调用的工具：

```text
source_locate Agent
├── interface locator CLI/helper
├── inspect/search candidate helper
└── decisions finalize/evaluate helper

外层工作流
└── source extractor CLI
```

工具只产候选或做机械工作，不拥有最终语义决策。

### 4.2 输入

| 来源 | 输入 | 用途 |
| --- | --- | --- |
| locate CLI | KID schema 副本 + `locate_candidates.interface_definition` | semantic interface、call_site 和 Python interface 候选 |
| resolve-third-party | `third_party_manifest.json`、`missing_repos.md` | 可用的 third-party repo 路径和缺失状态 |
| 用户/工作区 | sglang、sgl-kernel、third-party 源码 | Agent 阅读和搜索 |

`archetype/provider` 可以作为提示，但 source_locate 在两者缺失时也必须能够工作。

### 4.3 四层定位目标

- **interface_definition**：semantic low-level Python 接口定义；
- **kernel_impl**：从 host launcher 到核心 kernel 的实现调用链，允许跨仓、多文件、多级模板；
- **py_cpp_binding**：Python↔C++/FFI/JIT binding，允许多文件；纯 Python/DSL 时可
  `not_applicable`；
- **kernel_header**：与实现相关的 `.h/.cuh`；header-impl 合一或 DSL 时可
  `not_applicable`。

### 4.4 Agent workflow

对每个 semantic target：

1. 从 `interface + runtime_event.call_site` 出发定位 `interface_definition`；
2. 阅读接口实现，沿真实调用链向下追踪；
3. 找到 Python→native/JIT/FFI 边界并填写 `py_cpp_binding`；
4. 继续展开 host launcher、模板实例化、device helper 和核心 kernel，填写 `kernel_impl`；
5. 定位相关 header，填写 `kernel_header`；
6. 跨仓时使用 import、CMake/FetchContent、package metadata 和 manifest；
7. 对四层分别判断 `resolved/not_applicable/missed/best_effort`；
8. 写 `source-locate-agent-decisions/v1` decisions，逐 hit 保留 symbol/reason；
9. 调私有 finalize helper 检查 repo、文件、行号和层结构，生成完整 `source_locations` 与
   `ref/locate_agent_notes.md`；
10. 在 extract 之前结束。外层工作流随后显式调用 extract CLI。

source_locate Agent 对四层结果负唯一责任。即使 interface locator CLI 或 symbol helper 返回候选，
是否采纳、如何排序 hits、是否继续追踪仍由 Agent 决定。

### 4.5 interface locator CLI/helper

原“Layer 1”收缩为普通工具，只做确定性候选定位，例如：

```bash
python -m framework_engineer.source_location.cli locate \
  --schema <kid-v2-schema.json> \
  --manifest <third_party_manifest.json> \
  --sglang-repo-root <sglang-repo> \
  --out <located-candidates.json>
```

它不按 archetype/provider 分派，不定位 `py_cpp_binding/kernel_impl/kernel_header`，也不写
`source_locations`。CLI 在 schema 副本上增加临时 `locate_candidates.interface_definition`；
Agent 可以接受、修正或放弃候选，并在最终输出中删除该临时字段。

### 4.6 candidate search helper

已实现 provider-agnostic 的私有 `agent_helper inspect-target/search`，供 Agent：

- 从 `torch.ops.<namespace>.<op>` 搜索 `TORCH_LIBRARY`、`m.def/m.impl`；
- 搜索 `PYBIND11_MODULE`、pybind `m.def`；
- 搜索 `load_jit`、`load_inline`、`build_and_load`、`gen_jit_spec`；
- 根据 interface/symbol 在已知 repo 中运行 `rg`；
- 根据 import/module/file path 映射 repo root。

helper 只返回候选，不写 `source_locations`，因此不需要 FlashInfer/DeepGEMM 等 provider 专用
注册表。`search` 提供 `literal/registration/loader/build` 四种模式；以后若某种模式重复且稳定，
可以优化 helper，但不改变 Agent 的所有权。

私有 `finalize` 读取带 rationale/symbol/reason 的 decisions，校验后剥离分析字段、自动计算
`repo_hint`、删除 `locate_candidates` 并生成 notes。`evaluate` 要求 Golden 核心 hits 按序出现，
允许额外且有 reason 的 helper hits。这些命令不加入公开 `source_location.cli`。

### 4.7 extract CLI

现有 `framework_engineer/source_location/extractor.py` 保留为无语义后处理工具：

```bash
python -m framework_engineer.source_location.cli extract \
  --schema <located-schema.json> \
  --workspace-out <dir>
```

职责：

- 按 `source_locations.layers.<layer>.hits[]` 复制整文件；
- 计算 definition end line 并生成 `read_hints.txt`；
- 对 `not_applicable/missed` 生成占位说明；
- 回填 `kernel_sources_dir`；
- 不修改 Agent 的定位结论，不做源码语义判断。

extract 由 source_locate Agent 之后的外层工作流调用，不再称为 Layer 3。

### 4.8 source_locations 最小结构

目标结构只保留定位结果，不保留旧分层交接状态：

```json
"source_locations": {
  "layers": {
    "interface_definition": {
      "status": "resolved",
      "hits": [{"file": ".../sampling.py", "def_line": 1579}],
      "repo_hint": null
    },
    "kernel_impl": {
      "status": "resolved",
      "hits": [
        {"file": ".../sampling.cu", "def_line": 277},
        {"file": ".../sampling.cuh", "def_line": 1606}
      ],
      "repo_hint": null
    },
    "py_cpp_binding": {
      "status": "resolved",
      "hits": [{"file": ".../binding.cu", "def_line": 54}],
      "repo_hint": null
    },
    "kernel_header": {
      "status": "not_applicable",
      "hits": [],
      "repo_hint": null
    }
  }
}
```

不再需要：

- `source_locations.archetype`；
- `source_locations.source` 聚合优先级；
- 每层的 `source=locate_layer1/locate_layer2_agent`；
- `needs_agent`，因为 source_locate 本身就是 Agent。

### 4.9 输出

```text
<workspace>/
  source_locate_decisions.json           # Agent 内部工作合同；非下游固定契约
  decomposition_<backend>.schema.json    # 已写入四层 source_locations；无 kernel_sources_dir
  ref/locate_agent_notes.md              # 证据、未定位项和人工建议；非下游固定契约

# 随后的外层 extract 输出
  kernel_sources/<low_level_id>/
    interface_definition.py
    py_cpp_binding/
    kernel_impl/
    kernel_header/
    read_hints.txt
```

主要 golden 统一收敛在 `example_kernels/source_locate_golden` 的 testcase workspace：

- `input/all_backends/decomposition.kid.schema.json`：KID semantic resolution 后、无 `source_locations`；
- `workspaces/all_backends/locate/locate_candidates.schema.json`：locate CLI 输出的临时 interface 候选；
- `workspaces/all_backends/agent/located.schema.json`：source_locate Agent 完成四层定位后；
- `workspaces/all_backends/extract/decomposition.extracted.schema.json`：extract CLI 完成物料抽取后；
- `workspaces/all_backends/extract/kernel_sources/`：最终源码和 read hints。

---

## 5. KID ↔ source_locate ↔ 下游契约

```text
single-backend config directory
                    │
                    ▼
               KID Agent
  ├─ Runtime Capture CLI
  │  high NVTX + common-interface capture
  │  + Python stacks + Nsight correlation
  └─ Semantic Resolution
     semantic interface + call_site + provider 兜底
     + GPU duration 聚合/热点排序
                    │
                    ▼
 decomposition_<backend>.schema.json
          （无 source_locations）
                    │
                    ▼
              locate CLI
       interface definition candidates
                    │
       ┌────────────┴────────────┐
       │                         │
third_party_manifest       source repositories
       │                         │
       └────────────┬────────────┘
                    ▼
          source_locate Agent
    自主定位四层，写 decisions
                    │
                    ▼
        finalize helper + locate notes
                    │
                    ▼
           enriched located schema
                    │
                    ▼
             extract CLI
                    │
                    ▼
 kernel_sources/ + kernel_sources_dir
                    │
                    ▼
 snapshot / problem_translate / task_pack
```

契约要点：

1. KID 负责 target 语义边界和运行时耗时；source_locate 不重新选择 target。
2. source_locate 负责全部四层源码定位；KID 不产 `source_locations`。
3. `archetype/provider` 不控制 source_locate；缺失不构成阻塞。
4. Agent 分析报告放 `ref/`，不膨胀正式 schema。
5. extract CLI 只消费四层 hits，不理解 capture 类型或 provider。
6. problem_translate 以 semantic target 为对象，消费 source_locate 结果、snapshot 和 UT。

---

## 6. 落地改造清单

### 6.1 KID

- [x] 完成独立 Nsight Systems PoC：NVTX high/execution、CUDA correlation、SQLite 解析、一个
  execution capture 多 kernel launch 均已跑通；PoC 已移除显式 low-level decorator，通过
  PyTorch dispatcher/Triton launcher 自动捕获并保存 high→execution 调用链；实现位于
  `framework_engineer/kernel_interface_decomposer/nsys_poc.py`。
- [x] 在 H20 PoC 中用 11 个 SGLang 常见后端 case 验证并固化七类 execution capture registry；
  `capture_registry.py` 版本为 `kid-execution-capture/v2`。
- [x] 根据 PoC 结果 finalize `CAPTURE_MECHANISMS.md` 和 golden schema 的 `archetype/provider`
  口径；完整两阶段 golden 位于 `example_kernels/nsys_poc_kid_golden/`。
- [x] 将 high-level code-frame instrumentation（`high_call_id`、边界 frame identity）接入正式 Runtime Capture CLI。
- [x] 将 high→common-interface 完整 Python frame 链和逐边 `call_site_to_next` 接入正式 runtime events。
- [x] 使用 CUDA Runtime/Driver correlation id 连接 GPU activity，覆盖 NVTX pop 后执行的 kernel。
- [x] 构造按 invocation 去重的动态 capture 树，供 Semantic Resolver Agent 使用。
- [ ] 实现 Semantic Resolver Agent：选择 semantic frontier、确定 interface/call_site、provider 兜底、
  execution/kernel 聚合。
- [x] 将 Runtime `archetype` 从旧 F0–F8 含义迁移为七类 capture mechanism。
- [ ] `binding_provider` 改为可选 `provider`，删除固定 enum 和 locator 注册语义。
- [ ] 保持最终 schema 精简；完整调用树只留在 raw events/ref。
- [x] 支持单-backend 配置、默认 `unique_decomposition` 与 `all/last_n/single` invocation 选择，
  以及 eager/CUDA Graph 门禁；多 backend 由上层串行配置。
- [ ] 评估现有 flashinfer `gen_jit_spec` 等 patch：只能作为可选 runtime evidence，不能再作为
  source_locate 正确性的必需条件。

### 6.2 source_locate

- [x] 实现 source_locate Skill 与入口 Prompt，使其对四层定位负唯一责任，并在 extract 前结束。
- [x] 将现有 locator 收缩为 `locate` interface 候选 CLI；删除 archetype/provider 分派。
- [x] 不再实现 Layer 1 的 `py_cpp_binding` provider registry。
- [x] 将 symbol/registration/JIT/build 搜索实现成 Agent 可调用的通用候选 helper；不直接
  写 schema。
- [x] 实现 `source-locate-agent-decisions/v1`、finalize 和 evaluate 私有 helper；保证 KID 字段不变、
  自动生成 notes，并支持 Golden 核心链评测。
- [x] 增加 source_locations 内部 validator，检查文件存在、def_line 合法、四层状态一致。
- [x] 保留并复用现有 extractor CLI、range completion 和 `read_hints.txt` 逻辑。
- [x] 简化 `source_locations` contract，移除 `needs_agent` 和 layer1/layer2 provenance。
- [x] 更新 source-location golden：删除 `to_fill_after_layer1.json`，增加候选/Agent/extract 三阶段产物。

### 6.3 其他文档和下游

- [ ] 更新 `locate_source_locations_standard.md`：从三层职责改为 Agent + tools。
- [ ] 更新 `framework_engineer_design_v2.md` 中 KID/source_locate、F0–F8 和
  `binding_provider` 的旧描述。
- [ ] 检查 extractor/problem_translate/task_pack 对 `needs_agent/source/archetype` 的读取并迁移。

---

## 7. 测试计划

### 7.1 KID capture 与语义解析

1. synthetic：`high → low → execution`，Agent 选择 low。
2. transparent wrapper：`high → mid → low → execution`，Agent 跳过 mid。
3. split：`high → mid → low1/low2 → execution1/2`，Agent 选择两个 low。
4. fused semantic：多个 execution 共同组成一个有明确 ABI/UT 的父接口，Agent 选择父接口。
5. direct builtin/C API：无 Python callee frame时，使用 synthetic execution leaf + high callsite。
6. nested capture：外层 custom op 与内部 Triton/JIT 都保留，kernel 仅归最内层 capture，耗时不重复。
7. 单 semantic target launch 两个 GPU kernel：两个 kernel duration 均正确聚合。
8. NVTX 已 pop、kernel 后执行：correlation 仍能正确归因。
9. provider：可从 path/package 确定时填写，无法确定时合法为空。
10. 多 invocation/multi-backend：warmup 排除、相同拆分收敛为末次代表、prefill/decode 不同拆分分别
    保留，以及独立配置/schema 正确。

### 7.2 source_locate

1. interface locator CLI 给出候选，Agent验证或修正。
2. sgl-kernel built-in：Agent定位 wrapper、binding、CUDA implementation/header。
3. FetchContent/cross-repo：Agent沿 CMake/注册跳到 third-party repo。
4. Triton/DSL：binding/header 标 `not_applicable`，kernel_impl 命中 Python kernel。
5. FlashInfer/DeepGEMM：不依赖 provider 专用 CLI，Agent通过接口和调用链完成绑定与实现定位。
6. generated/JIT code：能定位模板和生成入口；无法穷尽时标 `best_effort` 并写 notes。
7. extract：四层文件、占位、read hints 和 `kernel_sources_dir` 正确。
8. decisions/finalize：证据字段剥离、合法 repo root、行号、状态和 KID projection 校验正确。
9. Golden evaluator：核心 hits 必须按序出现，允许带 reason 的额外 helper hits。

### 7.3 端到端

```text
resolve-third-party
→ KID Agent（Runtime Capture + Semantic Resolution）
→ source_locate Agent
→ extract CLI
→ problem_translate
```

在真实 SGLang eager 推理上验证至少一条 PyTorch、Triton、sgl-kernel 和 third-party 路径。

---

## 8. 已废弃的旧设计

以下内容不再是当前契约：

- KID 是“纯 CLI，无 Agent”；
- 从 execution stack 固定回溯到 high-level 的下一层作为 semantic target；
- `archetype` 表示 F0–F8 源码/交付形态；
- `F2|F3` provisional，再由 locate finalize；
- `binding_provider` 是固定 enum 和 binding locator registry key；
- source locate 按 archetype/provider 分派；
- locate Layer 1 CLI 填 `interface_definition + py_cpp_binding`；
- locate Layer 2 Agent 只补 missed；
- extract 被称为 locate Layer 3；
- `to_fill_after_layer1.json` 是正式流水线中间产物；
- `source_locations.source` 的 layer1/layer2 聚合优先级和 `needs_agent`。

旧 F0–F8 调研仍可作为历史 backend/source 研究资料，但不再定义 KID schema 的 `archetype`。

---

## 9. 一句话总结

**KID 用 common-interface capture 和 Nsight correlation 得到 execution 事实，再由 Semantic
Resolver Agent 从 high→execution 调用树中选择 semantic low-level target；source_locate 是独立的
自主 Agent，借助 locator 与私有 inspect/search/finalize helper 完成全部四层源码定位，并在
extract 前结束。最终 schema 只保留下游真正消费的 target、capture 类别、可选 provider、耗时、
call_site 和定位结果，所有分析中间信息留在 raw events/ref。**

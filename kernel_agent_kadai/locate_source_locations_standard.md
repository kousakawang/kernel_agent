source\_locations 填写标准（历史源码案例参考）

> **2026-07-19 迁移说明**：本文中的 F0–F8、Layer 1/2/3、`needs_agent/source` 和
> archetype null-rule 已废弃，不再定义生产契约。当前契约以
> [KID_and_locate_source_desgin_v2.md](file:///Users/bytedance/Desktop/infra_agent/kernel_agent/kernel_agent_kadai/KID_and_locate_source_desgin_v2.md)
> §4–§5 为准：`locate` CLI 只写临时 `locate_candidates`，source_locate Agent 对四层
> `source_locations` 负唯一责任，`extract` 只做文件复制、range completion、read hints 和
> `kernel_sources_dir` 回填。本文其余内容仅保留为各仓库源码调用链案例，后续会按新 Agent
> 工作流重写。Agent 的当前可执行规范见
> `framework_engineer/skills/source_locate.md` 和
> `framework_engineer/prompts/start_source_locate.md`；用户只提供一个
> `source-locate-agent-config/v1`，Agent 自主完成 locate、四层定位、finalize、extract 和最终
> workspace 校验。
>
> 日期：2026-07-14

***

## 0. 一句话总纲

> **对每个 low\_level\_target（= 被 KID 捕获、选中的那个 GPU kernel），定位「它自己的 python 定义 / py↔cpp 绑定 / 对应头文件 / 完整实现调用链」四层，每层只填「定义起始行** **`def_line`」，范围结束行留给 Layer 3 的 CLI 自动补。**

***

## 1. 三条不变量（先钉死原则，后面全部由此推导）

1. **只填定义行，不填范围**。每个 hit 只写 `{file, def_line}`（定义所在的那一行）。**不写 end / line\_end**。
   - 理由：end 边界的判定（函数体到哪结束）是机械动作，不该占用 agent 的判断力；agent 只干「找到定义在哪」这件真正需要智能的事。
   - end 由 **Layer 3 的 range-completion CLI** 按文件类型自动补，写进 `read_hints.txt`（见 §6）。**不回写 schema**。
2. **interface = low\_level\_target 本身，不往上看一层**。
   - 理由：snapshot 里 dump 的是 low\_level\_target **最直接的输入输出**；translate\_problem 只需搞懂这份直接 IO。再往上一层（原始数据怎么转成 kernel 输入）未必有用——那段逻辑常在更上层，多看一层反而引入噪声。
   - 所以 `interface_definition` = **low\_level\_target 自己的定义**，不是它上面那个 launcher 函数。
3. **Layer 3 不改 schema**。Layer 3（extract）只做两件事：把定位到的源码**拷贝**成物料文件、把行号范围**抄**进 `read_hints.txt`。它不修改 `source_locations`。

这里的 `def_line` 更准确地说是“定位单元的阅读范围起点”：Python 普通 interface 指向 concrete
`def/async def`，有 concrete implementation 时跳过 `@overload` stub；Triton/CuTe kernel 指向最外层
相关 decorator；C++/CUDA template 指向连续模板前缀的首个 `template<...>`；注册、导出和 loader
指向实际产生桥接语义的调用或宏。

***

## 2. 四层定版（语义 + 单文件/目录）

| 层（schema 键）            | 语义                                   | 形态      | hits 数                  |
| ---------------------- | ------------------------------------ | ------- | ----------------------- |
| `interface_definition` | **low\_level\_target 本身**的 python 定义 | **单文件** | 恰好 1                    |
| `py_cpp_binding`       | py↔核心 c++ 实现的桥接（AOT / JIT 两种模式）      | **目录**  | ≥ 1（可空=not\_applicable） |
| `kernel_header`        | 与 `kernel_impl` 源文件**一一对应**的头文件      | **目录**  | ≥ 1（可空=not\_applicable） |
| `kernel_impl`          | 算子实现的**完整调用链**（不止最终 kernel）          | **目录**  | ≥ 1，**按调用顺序**           |

**「单文件」vs「目录」的落地含义**：

- 单文件层（仅 `interface_definition`）：`hits` 数组**只允许 1 个** hit。多于 1 个 = 定位有歧义 → 标 `ambiguous`，交人工。
- 目录层（`py_cpp_binding` / `kernel_header` / `kernel_impl`）：`hits` 数组**允许多个** hit，Layer 3 会把它们各自拷贝、放进 `kernel_sources/<id>/<layer>/` **子目录**里的多个文件；`read_hints.txt` 里该层按 `hits` 顺序列多行。
  - **有序目录层**（`kernel_impl` / `py_cpp_binding`）：文件按 `hits` 顺序编号 `<n>_<basename>`（如 `1_activation.cu`、`2_activation.cuh`），因为顺序有语义（kernel_impl=调用链、py_cpp_binding=py→cpp 桥接次序）。
  - **无序目录层**（`kernel_header`）：文件用 `<basename>` 不编号，因为它与 `kernel_impl` 的源文件**一一对应**，无独立顺序。

***

## 3. 逐层填写标准

### 3.1 `interface_definition`（单文件，1 个 def\_line）

**填**：low\_level\_target 这个接口自身定义所在的 python 文件 + 定义起始行。

- **triton / cuteDSL（F1/F6）**：填**那个 kernel 函数自己**的定义。`def_line` 指向**第一个装饰器行**（`@triton.heuristics` / `@triton.autotune` / `@triton.jit` 里最靠上的那个），以便把 autotune 配置一并纳入范围。
- **其余有 python 接口的形态（F2/F3/F4/F5/F7/F8）**：填**你 import 进来、在 sglang 代码里直接调用的那个 python 接口**的 `def`。
  - 例：`torch.ops.sgl_kernel.silu_and_mul` 的直接 python 封装是 `sgl_kernel/elementwise.py` 里的 `def silu_and_mul`（它内部调 `torch.ops.sgl_kernel.silu_and_mul.default`）。填这个 `def`，**不是** runtime 底层的 `torch.ops...`。
- **F0（pytorch\_native）**：`not_applicable`（停在 torch/aten API，无自有源）。

> 判定口径：「运行时捕获到的接口，其**最近一层 python 源码定义处**」。对 F2 的 silu\_and\_mul，这一层就是 `elementwise.py:258`。

### 3.2 `py_cpp_binding`（**目录**，≥1 个 hit，按 py→cpp 次序）

**填**：把这个 python 接口桥接到核心 c++ 实现的**桥接文件**。**AOT 和 JIT 是两种 binding 模式，形态差别很大——所以本层是目录，允许多文件、多格式**：

- **AOT binding**（单文件常见）：静态注册。定位 `csrc/*_extension.cc`（或等价文件）里 `m.def("<op>", ...)` / `m.impl("<op>", &<symbol>)` 那一处，`def_line` 指向 `m.def` 行。
- **JIT binding**（常需多文件）：运行期生成。桥接往往**跨越 python 与 c++ 两侧**，两处都要纳入，**按 py→cpp 次序**列 hit：
  - **python 侧**：把 `.cu`/`.cpp` 转成可执行 module 的那处调用——`gen_xxx_module().build_and_load()`（flashinfer）/ `load_jit(...)`（sglang JIT）/ pybind `m.def(...)`（deep\_gemm）。
  - **c++ 侧**：TVM-FFI / pybind 的**导出声明文件**——如 flashinfer 的 `csrc/*_binding.cu`（`TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)` + 前向声明）。
  - 例（F5 flashinfer sampling）：hit①`sampling.py:68`（`gen_sampling_module().build_and_load()`）→ hit②`csrc/flashinfer_sampling_binding.cu:54`（FFI 声明+导出）。
- **形态决定的差异**：sglang JIT 的 binding 是**纯 python**（`load_jit` 行，无 c++ 侧）；sgl-kernel AOT 是**纯 `.cc`**；flashinfer JIT 是 **`.py` + `.cu` 两文件**——目录层天然容纳这些差异。

**约束（重要边界，目录化不放行这条）**：一个 binding **过程涉及多文件（允许）** ≠ 一个接口 **fan-out 到多个不同 c++ 实现（分解失败）**。

- sgl-kernel / sglang JIT **保证「一个 python 接口 ↔ 一个核心 c++ 实现」**——即便 binding 目录里有多个文件，它们描述的是**同一个** binding 过程的不同侧。
- 若 third-party 接口在 python 侧 **fan-out 成多步**（多个 python 各自 binding 到不同 c++ 实现）→ 这属于**分解失败**（一个 low\_level\_target 里塞了多个基础目标）。**不在本标准覆盖范围**：Layer 2 agent 遇到应**报错并请求人工介入**，不要强行把多个不相干实现塞进 binding。
- **F0 / F1 / F6**：`not_applicable`（无 py↔cpp 绑定）。

### 3.3 `kernel_header`（目录，多 def\_line，与 `kernel_impl` 源文件一一对应）

**填**：**只填 `kernel_impl` 里各 c++/cuda 源文件对应的头文件**，不引入其他东西。规范 c++ 工程里源文件与声明头一一对应，逐个填。

- 每个头文件一个 hit，`def_line` 指向该函数/kernel 的**声明行**。
- **边界（本层最单纯，别越界）**：header 只跟 `kernel_impl` 的源文件配对。凡是 py↔cpp 桥接性质的声明文件（如 flashinfer 的 `*_binding.cu` FFI 导出）——那是 **`py_cpp_binding` 层**的东西，**不放这里**。
- **唯一例外**：真正的 `__global__` kernel 若**直接定义在** **`.cuh`** **里**（如 flashinfer 的 `act_and_mul_kernel` 定义在 `activation.cuh`，没有独立声明头），则它**没有对应的 header 项**——不为它硬造一个。header-impl 合一的 `.cuh` 属 `kernel_impl`，本层留 `not_applicable`。
- **F0 / F1 / F6**：`not_applicable`（无 c++ 头）。

### 3.4 `kernel_impl`（目录，多 def\_line，按调用顺序）

**填**：算子实现的**完整调用链**，不只是最终的 `__global__` kernel。

**为什么是"链"而非单个 kernel**：translate\_problem 要翻译的是整个 low\_level\_target 的逻辑，它**不止** **`__global__`**——对输入的转换、数据 layout 调整、dtype dispatch 等 host 侧逻辑同样是 low\_level 的一部分，必须一并暴露。

- **按调用顺序**列 hit：从**入口 host 函数**（做 dispatch/layout/launch 的那个）开始，到**真正的** **`__global__`** **device kernel**结束。
- `def_line` 各指向对应函数/kernel 的定义行。
- **launch 与 kernel 分离是常态**（不止 cu，cuteDSL 亦然）：host launcher 和 `__global__` 常在不同文件、甚至不同仓库——全部按序纳入本层。
- **triton / DSL（F1/F6）**：kernel 自身即实现。`kernel_impl` 含该 kernel（及它调用的 `@triton.jit` device helper，若有）。此时它与 `interface_definition` 指向同一 kernel，属正常冗余。
- **F8（downloaded\_cubin）**：无源 → `missed`（`kernel_impl` 目录留空，附风险说明）。

***

## 4. 按 archetype 的填写矩阵

| archetype (code)             | interface\_definition        | py\_cpp\_binding（目录）           | kernel\_header              | kernel\_impl                     |
| ---------------------------- | ---------------------------- | ------------------------------- | --------------------------- | -------------------------------- |
| `pytorch_native` (F0)        | not\_applicable              | not\_applicable                 | not\_applicable             | not\_applicable                  |
| `sglang_triton` (F1)         | triton kernel def（首装饰器行）     | not\_applicable                 | not\_applicable             | 该 triton kernel（+device helper）  |
| `sgl_kernel_builtin` (F2)    | sgl\_kernel py 接口 def        | `*_extension.cc` 的 `m.def/impl`（1 文件） | host 函数声明头（≥1）              | host launcher → `__global__`（按序） |
| `sgl_kernel_thirdparty` (F3) | 同 F2                         | 同 F2                            | clone 内对应头（≥1）              | 同 F2，impl 落在 clone 仓             |
| `sglang_jit` (F4)            | sglang.jit\_kernel py 接口 def | `load_jit(...)` 行（纯 py, 1 文件）    | 常 n/a（.cuh header-impl 合一） | `sources[]` 里 `.cu/.cpp` 调用链     |
| `thirdparty_aot` (F5)        | 三方 py 接口 def                 | py `build_and_load` + c++ `*_binding.cu`（**2 文件**） | 常 n/a（.cuh header-impl 合一） | clone `csrc/*.cu` → `.cuh` 调用链   |
| `thirdparty_triton_dsl` (F6) | 三方 triton/DSL def            | not\_applicable                 | not\_applicable             | 该 DSL kernel（launch/impl 若分离则按序） |
| `thirdparty_cpp_jit` (F7)    | 三方 py 接口 def                 | py `build_and_load` + c++ `*_binding.cu`（**2 文件**） | 常 n/a（.cuh header-impl 合一） | `sources[]` 里 `.cu/.cpp` 调用链     |
| `downloaded_cubin` (F8)      | 加载 cubin 的 py 接口 def         | not\_applicable                 | not\_applicable             | missed（无源）                       |

> **archetype 锚点 vs kernel\_impl 跨仓**：`archetype` 锚定「python 接口从哪 import 来」（= KID 运行时观测到的），**与** **`kernel_impl`** **真正落在哪个仓无关**。典型：`silu_and_mul` 标 `F2`（接口来自 sgl-kernel），但真正的 `__global__` 在 flashinfer——这是**允许且正常**的，`kernel_impl` 可跨到另一个仓库。

> **flashinfer（F5/F7）三文件模式**：`*_binding.cu`（FFI 声明+导出）/ `csrc/*.cu`（host impl）/ `include/**/*.cuh`（kernel 模板）。其中 `*_binding.cu` 是 **binding** 层（跟 py 的 `build_and_load` 一起，共 2 个 binding hit），**不是** `kernel_header`；host impl + kernel 模板进 `kernel_impl`；`.cuh` header-impl 合一 → `kernel_header` 通常 `not_applicable`。

***

## 5. 两个完整实例（照抄模板）

### 5.1 F1 triton — `chunk_gated_delta_rule_fwd_kkt_solve_kernel`

low\_level\_target 是那个 triton kernel 本身（不是 launcher `chunk_gated_delta_rule_fwd_intra`）。

- `interface_definition`：`.../fla/chunk_fwd.py`，`def_line=24`（首装饰器 `@triton.heuristics`，含 autotune 配置）。**单文件 1 hit**。
- `kernel_impl`：目录，`.../fla/chunk_fwd.py`，`def_line=40`（`def chunk_..._kernel`）。此 kernel 自包含，故仅 1 hit。
- `py_cpp_binding`：`not_applicable`。
- `kernel_header`：`not_applicable`。

### 5.2 F2 sgl\_kernel — `torch.ops.sgl_kernel.silu_and_mul`（跨仓到 flashinfer）

完整调用链：`elementwise.py:silu_and_mul` → `torch.ops.sgl_kernel.silu_and_mul` →（binding）→ host `silu_and_mul(activation.cu)` →`<<<>>>`→ `__global__ act_and_mul_kernel(activation.cuh)`。

- `interface_definition`：`sgl-kernel/python/sgl_kernel/elementwise.py`，`def_line=258`（`def silu_and_mul`）。**单文件 1 hit**。
- `py_cpp_binding`：目录，1 hit——`sgl-kernel/csrc/common_extension.cc`，`def_line=76`（`m.def("silu_and_mul", ...)`；`m.impl` 在 77）。AOT 单文件 binding。
- `kernel_header`：目录，1 hit——`sgl-kernel/include/sgl_kernel_ops.h`，`def_line=139`（host 函数 `void silu_and_mul(...)` 的声明）。（`act_and_mul_kernel` 直接定义在 `.cuh`，无独立头，不列。）
- `kernel_impl`：目录，**2 hit，按调用序**：
  1. `sgl-kernel/csrc/elementwise/activation.cu`，`def_line=85`（host launcher `silu_and_mul`：shape/dtype dispatch + 启动）
  2. `flashinfer/include/flashinfer/activation.cuh`，`def_line=28`（真正的 `__global__ act_and_mul_kernel`）

> 对照旧的错误填法：曾把 `kernel_impl` 只填 `activation.cu:85`（host launcher）——**错**。真正的 kernel 在 flashinfer 的 `activation.cuh:28`，必须纳入；且 launcher 也要作为链的一环保留。

### 5.3 F5 flashinfer — `top_k_top_p_sampling_from_probs`（binding 目录多文件）

flashinfer AOT，展示 **binding 目录跨 py/c++ 两侧** + **`.cuh` header-impl 合一导致 kernel\_header = n/a**。

- `interface_definition`：`flashinfer/flashinfer/sampling.py`，`def_line=1579`（`def top_k_top_p_sampling_from_probs`）。**单文件 1 hit**。
- `py_cpp_binding`：目录，**2 hit，按 py→cpp 序**：
  1. `flashinfer/flashinfer/sampling.py`，`def_line=68`（`gen_sampling_module().build_and_load()`，py 侧把 C++ 变 module 的桥）
  2. `flashinfer/csrc/flashinfer_sampling_binding.cu`，`def_line=54`（C++ 侧 TVM-FFI 前向声明 + `TVM_FFI_DLL_EXPORT_TYPED_FUNC` 导出）
- `kernel_header`：`not_applicable`。sampling 的 host/kernel 都在 `.cuh`（header-impl 合一），且 FFI 声明已归 binding，本层无独立头可填。
- `kernel_impl`：目录，**3 hit，按调用序**：
  1. `flashinfer/csrc/sampling.cu`，`def_line=277`（TVM-FFI host 入口）
  2. `flashinfer/include/flashinfer/sampling.cuh`，`def_line=1606`（host dispatch 模板）
  3. `flashinfer/include/flashinfer/sampling.cuh`，`def_line=1192`（真正的 `__global__ TopKTopPSamplingFromProbKernel`）

***

## 6. `read_hints.txt` 与 end\_line 补全 CLI（Layer 3 规约）

Layer 2 只给 `def_line`；**范围 end 由 Layer 3 的 range-completion CLI 自动补**，仅写进 `read_hints.txt`。

**CLI 实现口径**（按文件类型两类）：

- **python（`.py`）**：从 `def_line` 起，按**缩进/AST** 找到该函数体自然结束的行。
- **c++/cuda（`.cpp`/`.cu`/`.cc`/`.h`/`.cuh`/`.hpp`）**：从 `def_line` 起，按**花括号配对**（或声明的 `;`）找到该定义结束的行。

**`read_hints.txt`** **格式**（每层一段；目录层按 `hits` 顺序多行，有序目录层文件带 `<n>_` 前缀）。以 F5 flashinfer sampling 为例：

```
interface_definition.py: read lines 1579-1738  (from .../sampling.py)
kernel_impl/1_sampling.cu:  read lines 277-325   (from .../csrc/sampling.cu)
kernel_impl/2_sampling.cuh: read lines 1606-1633 (from .../sampling.cuh)
kernel_impl/3_sampling.cuh: read lines 1192-1321 (from .../sampling.cuh)
py_cpp_binding/1_sampling.py:                read lines 68-68  (from .../sampling.py)
py_cpp_binding/2_flashinfer_sampling_binding.cu: read lines 54-60 (from .../flashinfer_sampling_binding.cu)
kernel_header/: N/A (not applicable for thirdparty_aot)
```

***

## 7. status / needs\_agent / source（沿用现有语义）

- **status**：`resolved`（定到，hit 齐）/ `not_applicable`（形态决定该层无源）/ `ambiguous`（多候选，单文件层出现>1 即是）/ `not_found`（没定到，给 `repo_hint`）/ `missed`（agent 也定不了，人工兜底）。
- **needs\_agent**：任一层为 `ambiguous`/`not_found`/`<FILL>`/`missed` → `true`；四层全 `resolved`/`not_applicable` → `false`。
- **source（逐层）**：该层**最后更新者**——CLI 定的 `locate_layer1`、agent 补的 `locate_layer2_agent`、人工填的 `manual`、dry-run 骨架 `dry_run`。顶层 `source_locations.source` 为派生聚合（`locate_layer2_agent` > `manual` > `locate_layer1` > `dry_run`）。

***

## 8. 新版 schema 字段形状

每层 hit 从 `{file, line_start, line_end}` 收敛为 `{file, def_line}`；目录层 `hits` 可 ≥1。

```json
"source_locations": {
  "archetype": "sgl_kernel_builtin",
  "archetype_code": "F2",
  "source": "locate_layer2_agent",
  "needs_agent": false,
  "layers": {
    "interface_definition": {
      "status": "resolved",
      "hits": [ { "file": ".../elementwise.py", "def_line": 258 } ],
      "repo_hint": "/sgl-workspace/sglang/sgl-kernel",
      "source": "locate_layer1"
    },
    "py_cpp_binding": {
      "status": "resolved",
      "hits": [ { "file": ".../common_extension.cc", "def_line": 76 } ],
      "repo_hint": "/sgl-workspace/sglang/sgl-kernel",
      "source": "locate_layer1"
    },
    "kernel_header": {
      "status": "resolved",
      "hits": [ { "file": ".../sgl_kernel_ops.h", "def_line": 139 } ],
      "repo_hint": "/sgl-workspace/sglang/sgl-kernel",
      "source": "locate_layer1"
    },
    "kernel_impl": {
      "status": "resolved",
      "hits": [
        { "file": ".../csrc/elementwise/activation.cu", "def_line": 85 },
        { "file": ".../flashinfer/include/flashinfer/activation.cuh", "def_line": 28 }
      ],
      "repo_hint": null,
      "source": "locate_layer2_agent"
    }
  }
}
```

> 单文件层（仅 `interface_definition`）：`hits` 恰好 1，多于 1 判 `ambiguous`。
> 目录层（`py_cpp_binding` / `kernel_header` / `kernel_impl`）：`hits` ≥ 1，Layer 3 抽成同名子目录下的多文件；有序目录层（`kernel_impl` / `py_cpp_binding`）文件带 `<n>_` 前缀，`kernel_impl` 严格按调用序、`py_cpp_binding` 按 py→cpp 序。

***

## 9. Layer 1 / 2 / 3 职责划分（谁定位什么）

locate 分三层，**定位的智能活集中在 Layer 2**，Layer 1/3 都是确定性工具：

| Layer | 类型 | 定什么 | 不定什么 |
| --- | --- | --- | --- |
| **Layer 1** | 确定性 CLI | 只做**可 100% 可靠**的：① `interface_definition`（KID 已给 call_site，最近一层 python def 是固定推导）；② **已注册 `binding_provider`** 的 binding（按 provider 注册，一套规则一次性拿全该 provider 的所有 binding hit）。 | **不碰** `kernel_impl` / `kernel_header`（哪怕 triton）——调用链展开、多级实现判断都不是固定规则能做对的。 |
| **Layer 2** | agent | **算子实现定位的主力**。读 Layer 1 结果，按自己对代码的理解，**尽力复现算子调用链上的核心逻辑**，把 `kernel_impl`（调用链）+ `kernel_header`（与 impl 一一对应）填好；以及**未注册 provider** 的 binding（已注册 provider 的 binding 由 Layer 1 一次性拿全）。 | 模版/重载/宏展开/ifelse 分支这类静态啃不动的，标 best-effort，不强求穷尽。 |
| **Layer 3** | 确定性 CLI | 只做**搬运**：按 `source_locations` 的 `{file, def_line}` 拷贝整文件、用 range-completion 补 end line 写进 `read_hints.txt`、回填 `kernel_sources_dir`。 | 不改 `source_locations`、不做任何语义判断。 |

**定位是尽力而为，不是硬门槛**：`kernel_impl` 能定到核心 kernel 就定，调用链上的 device helper（如 flashinfer triton 的 `scale_and_clamp`）由 agent 判断值不值得纳入，**不设硬规则**。定不全不阻断——因为 `schema.json` 里逐层的 `def_line`（一手导航线索）才是 locate 的核心产物，下游 translate_problem 看得到原仓库、可据行号自行探索；`kernel_sources/` 的 extract 文件是给看不到原仓库的 kernel_engineer 的**兜底参考**，允许残缺。

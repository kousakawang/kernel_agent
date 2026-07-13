# UT_TODO — 真实算子改造清单（KID 插桩捕获 UT）

> 背景：现有 [test_instrumentation_capture.py](./test_instrumentation_capture.py) 用 fake module 验证插桩**记账逻辑**，测不出「KID 的假设是否契合真实库结构」。本清单把它改造成 **在真实 GPU 环境、用真实算子** 跑的 UT，直接验证各类 kernel launch 捕捉是否真的成功。
>
> 用户已定各 case 的真实算子（见下）。**本文件只记 TODO，不改代码**——等更要紧的事做完再回来落地。
>
> 运行前提（改造后）：**必须在有 GPU、装了 sglang / sgl_kernel / triton / flashinfer 的环境**跑。真实算子/库缺失的 case 自动 `skipTest`，不 fake 兜底。日期：2026-07-13。

---

## 通用改造原则

1. **不再造 fake**：import 真实模块 → `ri._instrument_module(真实 module)` → 真实调用一次（喂真实 device tensor）→ 读 `events.jsonl` 断言。
2. **GPU-only**：`setUp` 里 `if not torch.cuda.is_available(): skipTest(...)`；真实库 import 失败也 `skipTest`。保留每 case 的 `dump()` 打印供人工核对 log。
3. **保留全局态复位**（`_INSTALLED/_CALL_COUNTER/_CURRENT/_CONFIG`）与临时 `output_dir`（沿用现有 `_CaptureBase`）。
4. **wrapper 只在 target ctx 内记录**：非 target 类（3a/4/6）的真实调用要包在 `self.target_ctx()` 里，或先 wrap 一个真实 target 再在其体内调用。
5. **覆盖度局限（贯穿所有非 triton case 的备注）**：跑通一个算子 example **不保证**同类其他算子都能捕获——真实算子的注册/导出/调用形态各异。每个 case 的断言只证明「该具体算子这条路径能捕获」，不外推。这条要写进每个真实 case 的 docstring。

---

## 逐 case TODO

### Case 1 — target wrap（任意入口函数）
- **算子**：随便写一个自有入口函数当 target（用户：无所谓）。
- **改造**：临时 py 写一个函数，设 `_CONFIG["target"]` 指向它 → `_instrument_module` → 调用。
- **断言**：`_kid_target_wrapped` 挂上；events 有 `target_wrapped`（无 `target_wrap_failed`）+ `target_begin/end`。
- **状态**：与真实库无关，现有 fake 版逻辑基本可直接留用（可不 GPU）。

### Case 2 — triton kernel launch（F1/F6 同段 patch）
- **算子**：自写一个 triton 空 kernel 或 vector-add（用户：kernel 逻辑不重要）。
- **改造**：真实 `import triton; @triton.jit def add_kernel(...)`；`_instrument_module(triton.runtime.jit)`；在 `target_ctx()` 内真实 `add_kernel[grid](...)` launch 一次（喂 cuda tensor）。
- **断言**：`triton.runtime.jit.JITFunction._kid_getitem_patched` 挂上；events 有 `wrap category=triton_dsl` + `implementation.source_files=[本测试文件路径]` + `definition_line=真实行号`。
- **价值**：最硬的一类；新版 triton 若改了 `__getitem__` 启动路径，这里直接暴露。

### Case 3a — sgl_kernel AOT 算子（F2/F3）✅ 已确认
- **算子**：**`import sgl_kernel; sgl_kernel.gelu_and_mul(x)`**（装好的 sgl_kernel 包里的 AOT 算子，编进 `.so`；**不是** `sglang.jit_kernel.activation.gelu_and_mul`，也不是 `srt/layers/elementwise` 的 triton 版）。
- **输入**：`x = torch.randn(8, 8, 128, dtype=torch.float16, device="cuda")`（用户指定）。
- **改造**：`import sgl_kernel` → `_instrument_module(sgl_kernel)` → `target_ctx()` 内调 `sgl_kernel.gelu_and_mul(x)`。
- **断言**：events 有 `wrap category=sgl_kernel`，`implementation.source_files` 为空（AOT，源交 locate 静态补）。
- **待确认/风险**：`gelu_and_mul` 是否是 `sgl_kernel` 模块里 `owner==module.__name__` 的普通函数——[_wrap_module_functions](../runtime_instrumentation.py#L336-L349) 只对满足该门槛的可调用套壳。若它其实是 `torch.ops.sgl_kernel.gelu_and_mul` 的再导出 / OpOverloadPacket，**当前 patch 可能抓不到** → 那样本 case 会「无 wrap 事件」，正好暴露真实盲点，需在断言里显式区分「捕获成功」vs「暴露盲点」并打印。

### Case 3b — torch.ops.sgl_kernel.* 直调 — SKIP（用户决定）
- 用户核实：这些基本是老代码 / CPU 代码，GPU 用不到。**不测**，仅在此备注留痕。

### Case 4 — torch 原生（F0）+ ⚠️ 已知局限
- **算子**：`torch.nn.functional.linear`（用户认可）。
- **改造**：`import torch.nn.functional as F` → `_instrument_module(F)` → `target_ctx()` 内 `F.linear(x, w)`（cuda tensor）。
- **断言**：`F.linear._kid_wrapper_wrapped` 挂上；events 有 `wrap category=pytorch_native`。
- **⚠️ 必须在 docstring 写明的局限**（回答用户疑问 4）：[_patch_torch_functional](../runtime_instrumentation.py#L352-L381) 是**写死白名单**（linear/conv1d/conv2d/layer_norm/rms_norm/scaled_dot_product_attention/silu/gelu/softmax/embedding）。**捕获到 `F.linear` ≠ 能捕获所有 torch 原生算子发起**。当前抓不到：
  1. 不在白名单的 `F.*`（如 `F.dropout`）；
  2. 不走 functional 的 `torch.matmul` / `a @ b` / `tensor.softmax()`；
  3. 直接 `torch.ops.aten.*`。
  → 这是 KID 升级要考虑的（扩白名单 or 换 `__torch_function__`/TorchDispatch 底层 hook），**不属本 UT 能解决**。case 4 只证明「白名单机制本身 work」。

### Case 5 — sglang-owned JIT（F4）
- **算子**：`from sglang.jit_kernel.norm import rmsnorm`（参考 [test_rmsnorm.py:26](../../../../sglang/python/sglang/jit_kernel/tests/test_rmsnorm.py#L26)）。
- **输入**：`input=torch.randn(bs, hidden, device="cuda", dtype=fp16)`, `weight=torch.randn(hidden, ...)`；如 `rmsnorm(input, weight, out=..., eps=1e-6)`。
- **改造**：`_instrument_module(sglang.jit_kernel.utils)` 使 `load_jit` 被 patch → 首次调 `rmsnorm(...)` 触发 JIT 加载。
- **断言**：events 有 `jit_module_loaded` + `source_files`（cpp+cuda）+ `wrappers_by_export`。
- **注意**：`load_jit` 直接 `_record_event`，**无需 target ctx**；但要确保 rmsnorm 首次调用时真的走 `sglang.jit_kernel.utils.load_jit`（若已被别处提前触发过 JIT 编译/缓存，可能不再走 load_jit）。

### Case 6 — 第三方前缀套壳（flashinfer）+ ⚠️ 语义边界
- **算子**：`from flashinfer.norm import rmsnorm`（参考 [test_rmsnorm.py:38](../../../../sglang/python/sglang/jit_kernel/tests/test_rmsnorm.py#L38)）。
- **改造**：`_CONFIG["resolution"]["third_party_prefixes"]` 里配 `"flashinfer"` → `_instrument_module(flashinfer.norm)` → `target_ctx()` 内调 `rmsnorm(...)`。
- **断言**：events 有 `wrap category=third_party`（对 `flashinfer.norm.rmsnorm` 这个 **python 入口函数**套壳）。
- **⚠️ 语义边界（写进 docstring）**：第 6 类 patch 只对第三方模块的**普通 python 函数**套壳，**不是** F7 的 JIT 源捕捉（`flashinfer.jit.core.gen_jit_spec` 的 patch **尚未实现**，见设计文档 §4.5，是升级目标）。所以 case 6 证明「第三方前缀套壳 work」，**不代表**抓到了 flashinfer 底层 JIT kernel 源。
- **依赖 `owner==module.__name__` 门槛**：同 3a 风险，若 `flashinfer.norm.rmsnorm` 是再导出，可能不被套 → 需区分「捕获」vs「暴露盲点」。

---

## 汇总表

| # | 捕捉类型 | 真实算子（用户已定） | 关键断言 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | target wrap | 自写入口函数 | `target_wrapped`+begin/end | 与库无关 |
| 2 | triton launch | 自写空/vector-add triton kernel | `_kid_getitem_patched`+`triton_dsl`+source_files | 最硬 |
| 3a | sgl_kernel AOT | `sgl_kernel.gelu_and_mul`, (8,8,128) fp16 | `wrap sgl_kernel`, source_files 空 | 查 owner 门槛；可能暴露盲点 |
| 3b | torch.ops.sgl_kernel | — | — | **SKIP**（GPU 用不到） |
| 4 | torch F0 | `F.linear` | `wrap pytorch_native` | ⚠️ 白名单局限，非全覆盖 |
| 5 | sglang JIT | `sglang.jit_kernel.norm.rmsnorm` | `jit_module_loaded`+source_files+wrappers | 首调才走 load_jit |
| 6 | 第三方前缀 | `flashinfer.norm.rmsnorm` | `wrap third_party` | ⚠️ 非 F7 JIT 捕捉；owner 门槛 |

## 落地前提
- 改造后需在真实 GPU 环境跑；缺 GPU/库的 case `skipTest`。
- F7（`gen_jit_spec`）仍是升级 TDD 靶子，不在本轮真实化范围。
- 每个非 triton case 的 docstring 写明「单算子过 ≠ 同类全覆盖」的局限。

# KID → source_locate 对接合同

> 本文定义 KID 最终产物与 `source_locate` Agent 之间的正式交接边界。
> 对接实现、测试和评审应以本文、最新 golden 及 artifact validator 为准。

## 1. 流水线与责任边界

```text
KID Runtime Capture CLI
  └─ execution captures + Python stacks + Nsight kernel correlation
                         │
                         ▼
KID Semantic Resolver Agent
  └─ semantic interface + call_site + runtime metrics
                         │
                         ▼
output/<backend>/decomposition.schema.json
                         │
          third_party_manifest + source repositories
                         │
                         ▼
source_locate Agent
  └─ interface_definition / py_cpp_binding / kernel_header / kernel_impl
                         │
                         ▼
enriched schema + locate notes → extract CLI → kernel_sources/
```

KID 负责确定 low-level target 的语义边界及运行时热点数据；`source_locate` 负责该
target 的全部源码定位。`source_locate` 不重新选择或拆分 KID 已确定的 semantic target，
KID 也不提前填写源码文件、symbol 或四层定位结果。

## 2. 正式交接产物

每个 backend 独立生成：

```text
output/<backend>/decomposition.schema.json
```

文件是 `kernel-interface-decomposition/v2` 的数据实例，不是 JSON Schema 元文件。
`source_locate` 只把它作为正式 KID 输入；Runtime Capture 的 JSONL、SQLite、完整
Python stack 和 Semantic Resolver notes 都不是稳定对接接口。

最新完整样例：

```text
kernel_agent/example_kernels/nsys_poc_kid_golden/
  output/nsys_poc/decomposition.schema.json
```

## 3. 顶层字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | string | 固定为 `kernel-interface-decomposition/v2`。 |
| `backend_name` | string | 该文件对应的单个 backend/run 名称。 |
| `target` | object | 用户指定的 high-level target：`interface/file/line`。 |
| `coverage_report` | object | 所选 high-level invocation 的 GPU 时间覆盖情况。 |
| `kernels` | array | 已按 GPU 热点降序排列的 semantic low-level targets。 |

`coverage_report.per_invocation[]` 保存 `call_id`、stage、`covered_us`、
`total_gpu_us`、coverage 和未归因 kernel ids。coverage 不足表示 KID 可能漏捕获，
应先处理 KID 结果，不能由 `source_locate` 猜测缺失 target。

## 4. `kernels[]` 字段合同

| 字段 | 必需 | source_locate 如何使用 |
|---|---:|---|
| `rank` | 是 | 热点顺序；不修改。 |
| `low_level_id` | 是 | 稳定唯一 ID；作为四层定位和抽取目录的关联键。 |
| `kernel.raw_name` | 是 | Nsight 中最热的代表 kernel，可作为实现搜索证据。 |
| `kernel.normalized_name` | 是 | 便于人和 Agent 阅读的稳定名称。 |
| `interface` | 是 | KID 已确定的 semantic Python 接口，是源码定位的主要起点。 |
| `archetype` | 是 | execution capture 类别，固定枚举；只能作为背景提示。 |
| `provider` | 否 | 可选源码仓库/包提示；允许 `null`，不能作为硬分派条件。 |
| `metrics.duration_us` | 是 | 该 semantic target 下所有关联 GPU kernel duration 之和。 |
| `metrics.share_in_invocation` | 是 | 在当前 high-level invocation 中的 GPU 时间占比。 |
| `measurement` | 是 | 指标、聚合方式和样本数；不修改。 |
| `runtime_event.call_site` | 是 | 推理调用链中调用 semantic interface 的源码位置。 |
| `runtime_event.attribution` | 是 | KID semantic 归因方法和置信度。 |

重要语义：`kernel.raw_name` 只是代表 kernel。一个 semantic target 可以关联多个 GPU
kernel，此时 `metrics.duration_us` 是所有相关 kernel duration 的聚合值，不等于代表
kernel 自身的 duration。

## 5. 固定 `archetype` 枚举

`archetype` 只回答：主要 execution 是通过哪种 common interface 被捕获的。它不描述
源码仓库、AOT/JIT 交付形态，也不决定 `source_locate` 的代码路径。

| 值 | 捕获边界 | 常见覆盖 |
|---|---|---|
| `pytorch_dispatch` | `TorchDispatchMode.__torch_dispatch__` | ATen、`torch.library`、`torch.ops` 扩展算子 |
| `triton_launch` | Triton `JITFunction/Autotuner/Heuristics` launcher | SGLang 或 third-party Triton kernel |
| `cute_dsl_launch` | `cutlass.cute.compile` 返回 callable | CuTe DSL kernel |
| `tilelang_launch` | `tilelang.JITKernel.__call__` | TileLang JIT kernel |
| `tvm_ffi_call` | `tvm_ffi.module.Module` export | SGLang JIT、FlashInfer AOT/JIT export |
| `inductor_launch` | Inductor `CachingAutotuner.run` | `torch.compile` 生成的 Triton kernel |
| `python_binding` | 已登记的 Python-visible extension export | DeepGEMM 等绕过 PyTorch dispatcher 的绑定 |

这是固定枚举。增加 capture mechanism 时必须同时更新：

1. `capture_registry.py`；
2. `CAPTURE_MECHANISMS.md`；
3. Runtime/Semantic validator；
4. golden 和对应测试。

## 6. `provider` 的解读

`provider` 是可选自由字符串，表示实现源码最可能所属的仓库或包，例如 `sglang`、
`sgl-kernel`、`sgl-attn`、`flashinfer`、`deepgemm`。

- 它不表示编译器：SGLang 的 Triton/CuTe/TileLang target 仍可写 `sglang`。
- runtime namespace 与源码仓库可能不同，例如 `sgl_kernel` 导出的 attention 实现可能属于
  `sgl-attn`。
- `provider=null` 不构成定位阻塞；Agent 必须仍能从 `interface`、call site、仓库和 manifest
  自主查找。
- 不建设 provider→locator 的硬编码分派表。

## 7. `call_site` 与源码定位的区别

`runtime_event.call_site` 属于 KID：它表示运行时调用 semantic interface 的位置。例如：

```json
{
  "interface": "deep_gemm.bf16_gemm_nt",
  "archetype": "python_binding",
  "provider": "deepgemm",
  "runtime_event": {
    "call_site": {
      "file": "/abs/path/nsys_poc.py",
      "line": 933
    },
    "attribution": {
      "method": "python_stack+execution_capture+cuda_correlation_id",
      "confidence": "high"
    }
  }
}
```

该位置不是 `deep_gemm.bf16_gemm_nt` 的定义位置，也不是 C++/CUDA kernel 的位置。
`source_locate` 应以 semantic `interface` 为主要目标，以 call site 的 import、alias、对象来源
和调用上下文为证据，再定位四层源码。

## 8. source_locate 必须完成的四层结果

`source_locate` 对以下结果负唯一责任：

- `interface_definition`：semantic low-level target 自身的 Python 定义；
- `py_cpp_binding`：Python 到 C++/CUDA/FFI 实现的桥接链；
- `kernel_header`：与实现相关的声明/header；
- `kernel_impl`：host dispatch/launcher 到 device kernel 的实现调用链。

Agent 可以调用 `framework_engineer/source_location` 下的 locator、validator 和 extract CLI，
也可以自主搜索与阅读源码。CLI 返回的是工具结果或候选，最终四层语义判断仍由 Agent 负责。

定位结果允许多文件和跨仓库。找不到或不能完整静态展开时应记录 `missed`/best-effort 状态及
证据，不得改写 KID 的 `interface` 来掩盖定位失败。

## 9. KID 最终产物禁止包含的内容

以下内容不得出现在 `decomposition.schema.json`：

- `implementation`、`source_files`、`symbols`；
- `source_locations`、`kernel_sources_dir`；
- Python stack、execution/capture ids；
- semantic 候选、选择理由、alternative、Agent notes；
- 旧字段 `archetype_code`、`binding_provider`、`dry_run`。

前两类由 `source_locate`/extract 后续补充；运行时证据留在 `cli_log/`；自由分析过程留在
`ref/`，避免主 schema 随内部实现膨胀。

## 10. source_locate 的输入与输出约束

输入：

- KID `decomposition.schema.json`；
- `third_party_manifest.json` 和缺失仓库说明；
- SGLang、sgl-kernel 及 third-party 源码；
- 可选的原仓库搜索、build metadata 和 runtime kernel 名证据。

输出：

- 保持 KID 字段不变的 enriched schema；
- 每个 `low_level_id` 的四层 `source_locations`；
- 自由格式 `ref/locate_agent_notes.md`；
- 调用 extract 后生成的 `kernel_sources/` 与 `kernel_sources_dir`。

source_locate 不修改 `rank`、`interface`、`archetype`、`provider`、kernel 名称、metrics、
measurement、call site、attribution 或 coverage。如果发现 semantic target 本身错误，应报告给
KID/人工重新解析，不能在 locate 阶段静默替换。

## 11. 验证与验收

KID handoff 产物先运行：

```bash
PYTHONPATH=kernel_agent python3 -m \
  framework_engineer.kernel_interface_decomposer.artifact_validator \
  kernel_agent/example_kernels/nsys_poc_kid_golden
```

validator 会检查固定字段、archetype、排序、coverage、代表 kernel、call site 和 Runtime
Capture 的一致性，并禁止源码定位字段进入 KID 最终产物。

source_locate 的验收至少包括：

1. 每个 `low_level_id` 都有明确四层状态；
2. `interface_definition` 对应 KID 的 semantic `interface`；
3. 多 hit 的 binding/implementation 顺序能表达真实调用链；
4. KID 原字段没有被修改；
5. notes 记录歧义、跨仓证据和 best-effort 缺口；
6. extract 能依据定位结果生成 `kernel_sources/`。

## 12. 参考资料

当前合同与实例：

- `kernel_agent/example_kernels/nsys_poc_kid_golden/output/nsys_poc/decomposition.schema.json`
- `kernel_agent/example_kernels/nsys_poc_kid_golden/ARTIFACT_GUIDE.md`
- `kernel_agent/example_kernels/nsys_poc_kid_golden/README.md`
- `kernel_agent/framework_engineer/kernel_interface_decomposer/artifact_validator.py`

capture/archetype：

- `kernel_agent/framework_engineer/kernel_interface_decomposer/CAPTURE_MECHANISMS.md`
- `kernel_agent/framework_engineer/kernel_interface_decomposer/capture_registry.py`

总体架构与开发状态：

- `kernel_agent/kernel_agent_kadai/KID_and_locate_source_desgin_v2.md`
- `kernel_agent/remain_core_dev.md`

source_location 工具：

- `kernel_agent/framework_engineer/source_location/`
- `kernel_agent/framework_engineer/source_location/example/third_party_manifest.json`

`locate_source_locations_standard.md`、`step0_5_handoff_contract.md` 和
`source_location/example/README.md` 仍包含旧 F0–F8 或 Layer 1/2/3 设计。它们可以参考四层定位
案例，但不能覆盖本文的最新职责和字段合同。

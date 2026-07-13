# Dry-run：Step 0.5 → Step 1 交付链验证

在**无 GPU、无真实 profiling**的前提下，验证「经人工介入后，整条交付链（KID → locate → Layer 3 抽取）能把 Step 1 所需的全部产物跑通」。

它不跑真实 KID / locate，而是生成与真实产物**结构完全一致**的骨架，只把「真实 agent 也定位不到、需人工判断」的字段留成 `<FILL: ...>` 占位；人工填好后，第三步调用**真实的 Layer 3 CLI** 完成源码抽取。

> 依据：[step0_5_handoff_contract.md](../../kernel_agent_kadai/step0_5_handoff_contract.md)、[KID_and_locate_source_desgin_v2.md](../../kernel_agent_kadai/KID_and_locate_source_desgin_v2.md) §5。

---

## 三步总览

| 步骤 | 命令 | 干什么 | 人工填什么 |
| --- | --- | --- | --- |
| ① kid | `dry_run.cli kid` | 每 backend 生成 `decomposition_<backend>.schema.json` 骨架 | 每个 kernel 的 `interface` + `archetype`（+ `low_level_id`） |
| ② locate | `dry_run.cli locate` | 给每个 kernel 补 `source_locations` 骨架（按 archetype 套 null 规则） | 定位不到的层的 `file` / `line_start` / `line_end` |
| ③ extract | `dry_run.cli extract` | passthrough 调**真实 L3 CLI**，切片抽出四层源码 | 无（全自动） |

每步都会打印**新生成文件的绝对路径**和**需要人工填写的行号**。

---

## 前置

- 用装了本仓库的 Python（`framework_engineer` 可导入即可，dry-run 本身不依赖 GPU/torch）。
- 一份 dry-run config，见样例 [../configs/dry_run.example.py](../configs/dry_run.example.py)。config 与真实（V2）KID **同构**：多 backend `service_cmds` + 统一 high_level `target`。

关键 config 字段：

```python
service_cmds = [{"backend_name": "triton", "cmd": "..."}, {"backend_name": "flashinfer", "cmd": "..."}]
target = {"file": ".../radix_linear_attention.py", "line": 78}   # 分解起点(high_level)
output_root = "/tmp/kid_dry_run_out"                              # dry-run 产物根
kernels_per_backend = 3                                          # 每 backend 生成几个 kernel 槽
```

---

## 步骤 ① kid — 生成 schema 骨架

```bash
python3 -m framework_engineer.dry_run.cli kid --config framework_engineer/configs/dry_run.example.py
```

- 每个 backend 生成 `<output_root>/workspaces/<backend>/decomposition_<backend>.schema.json`。
- 每个 schema 含 `kernels_per_backend` 个 kernel 槽。

**你要做的**：对每个真实关心的 kernel 槽，填这几个字段（其余槽可整段删掉 = 选择性放弃该 kernel）：

- `low_level_id`：该 low_level 的稳定 id（会用作 `kernel_sources/<id>/` 子目录名）
- `interface`：运行时接口名（如 `torch.ops.sgl_kernel.gelu_and_mul` / triton fn 名）
- `archetype` + `archetype_code`：明文类别名 + 对应 F 代号（见下表）

> `metrics` / `runtime_event.wrapper` 等标了「可选」的 `<FILL>` **可以直接删整行**，不影响后续。

---

## 步骤 ② locate — 补 source_locations 骨架

```bash
python3 -m framework_engineer.dry_run.cli locate --workspace <output_root>/workspaces
# 或对单个 schema：--schema <path>
```

- **闸门**：若 schema 里 `interface`/`archetype`/`low_level_id` 还是 `<FILL>`，直接报错（rc=2）并列出行号——先回步骤 ① 填完。
- 通过后，按每个 kernel 已填的 `archetype` **自动套 null 规则**生成四层：
  - 形态决定不适用的层（如 `sglang_triton` 的 py_cpp_binding / kernel_header）→ **自动 `not_applicable`，无需填**。
  - 其余适用层 → **`missed` + file/line 占位**（模拟"agent 定位不到，交人工"）。
- 额外产出 `locate_report.json` + `locate_agent_notes.md`。

**你要做的**：对每个 `missed` 层，填 `hits[0].file` / `line_start` / `line_end`（指向真实 sglang 源码，或 `third_party_cache/` 里 clone 的文件），并把该层 `status` 改成 `resolved`。定位不到又想放弃的层，保留 `missed`（见步骤 ③ 的 `--allow-empty`）。

---

## 步骤 ③ extract — 真实抽取（调真实 L3 CLI）

```bash
python3 -m framework_engineer.dry_run.cli extract --workspace <output_root>/workspaces
# 放弃部分层、允许占位放行：加 --allow-empty
```

- **硬停闸门**：若**必填层**（`interface_definition` / `kernel_impl`）仍是 `missed` 或含 `<FILL>`，直接停（rc=2）并列出待补清单——回步骤 ② 填 file/line。
- 通过后，对每个 kernel 生成 `<workspace>/<backend>/kernel_sources/<id>/`：
  - `resolved` 层 → 按 file+行号切片（前后各 200 行 padding）抽成 `interface_definition.py` / `kernel_impl.{py,cu,cpp}` / …
  - `not_applicable` 层 → 空文件 + 注释
  - `missed` 层（仅 `--allow-empty`）→ 占位空文件 + 注释
  - `read_hints.txt`：每层一行（`read lines X-Y` / `N/A` / `MISSING`）
- 回填 `kernel_sources_dir` 到 schema。

---

## 产物结构（跑完三步）

```
<output_root>/workspaces/<backend>/
├── decomposition_<backend>.schema.json    # ①生成 → ②补 source_locations → ③回填 kernel_sources_dir
├── locate_report.json                     # ②
├── locate_agent_notes.md                  # ②
└── kernel_sources/<low_level_id>/         # ③
    ├── interface_definition.py
    ├── kernel_impl.{py,cu,cpp}
    ├── py_cpp_binding.cc                   # 或空文件+注释
    ├── kernel_header.h                     # 或空文件+注释
    └── read_hints.txt
```

`kernel_sources/<id>/` 的结构即对接 Step 2 `import-kernel-sources-to-taskpack` 所需。

---

## archetype 明文类别名对照

产物/配置里**只用明文名**（禁裸 `F*` 代号）；`archetype_code` 仅作附属。

| 明文 `archetype` | code | 含义 | 四层适用性 |
| --- | --- | --- | --- |
| `pytorch_native` | F0 | torch/aten/cuBLAS API | 仅 a；b/c/d 不适用 |
| `sglang_triton` | F1 | sglang 自带 triton | a/b；c/d 不适用 |
| `sgl_kernel_builtin` | F2 | sgl-kernel 内实现 (AOT) | 四层俱全 |
| `sgl_kernel_thirdparty` | F3 | sgl-kernel FetchContent 三方 | 四层俱全 |
| `sglang_jit` | F4 | sglang-owned JIT | 四层俱全 |
| `thirdparty_aot` | F5 | 三方 C++/cuda AOT | 四层俱全 |
| `thirdparty_triton_dsl` | F6 | 三方 triton/cuteDSL | a/b；c/d 不适用 |
| `thirdparty_cpp_jit` | F7 | 三方 C++ JIT (flashinfer/deep_gemm) | 四层俱全 |
| `downloaded_cubin` | F8 | 下载预编译 cubin (无源) | a；b 无源；c/d 不适用 |

---

## 返回码

- `0`：成功。
- `2`：闸门拦截或硬停（locate 前 KID 字段没填完 / extract 前必填层没定位）——按打印的行号补齐后重跑。

## 常见问题

- **填错 archetype 想重来？** 改 schema 里的 `archetype` 后重跑 `locate`——它会按新形态重生成 `source_locations`（覆盖式）。
- **填了不存在的路径/文件，或行号非法（如 end < start）会怎样？** extract 会**当作"没填"处理**——不会静默放行：
  - 若发生在**必填层**（`interface_definition` / `kernel_impl`）→ **硬停 rc=2**，清单里标出原因（如 `file not found: /x/y.py`）。回步骤 ② 填真实有效的路径。
  - 若发生在**非必填层**（如 py_cpp_binding）→ 生成占位文件 + `read_hints` 标 `MISSING (file not found: ...)`，extract 继续（rc=0）。
  - 所以**用户填写时必须保证 file 真实存在、行号有效**；dry-run 不校验 config 里 third-party 目录，靠这一层兜住"填错"。
- **某层实在定位不到？** 保留 `missed`，用 `extract --allow-empty` 放行，会生成占位空文件 + `MISSING` 提示（= 用户已知风险）。人工绝不需要手动建/搬这些文件——CLI 全包了。
- **只想验一个 kernel？** 步骤 ① 里把多余 kernel 槽整段删掉即可。

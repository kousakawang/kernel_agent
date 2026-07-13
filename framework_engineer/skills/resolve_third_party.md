# Skill: resolve-third-party

Step 0.5 的第一个 skill。目标：**确定 sglang 运行时依赖的所有 GPU 推理相关 third-party 仓库的正确版本，并把它们按正确版本 clone 到本地缓存**，产出一份 `third_party_manifest.json`（`name → 正确版本本地路径`）+（若有失败项）`missing_repos.md`。

> **形态说明：当前以 CLI 为主，不需要 agent 编排。** 这个问题已被完全确定性化——「明确仓库 → 找版本线索 → 版本映射成 ref → clone」四步全是查表/读取/规则/执行，没有需要智能判断的环节。正常路径就是跑一条 CLI 命令。agent 只在**规则未覆盖的新情况**（新库、tag 命名变化、`version_mismatch`、clone 失败归因）时作为兜底介入，去把新规则补进 `registry.py` / 映射——本文档同时是给「未来维护 registry 的人（或 agent）」看的运维手册。

本 skill **不做**源码到 kernel 的精确定位——那是第二个 skill `locate-kernel-source` 的职责。本 skill 只保证「正确版本的仓库源码出现在磁盘上」。

## 职责边界

- **做**：明确需要的仓库、定版本（两桶）、把每个库的**正确版本 git 源码 clone 到 `(name, version)` 缓存**、如实记录成功/失败。
- **不做**：解决 clone 失败（网络/tag 缺失）；重编 sgl-kernel；改 sglang 源码；装新包；把 clone 链接进运行环境；精确定位 kernel 源码。

## 快速使用（CLI 为主）

在**装了 sglang/flashinfer 的目标 python 环境**里（否则 Bucket A 的 importlib 全查不到）：

```bash
# ① dry-run：定版本 + 写 manifest（含可复跑 clone 命令），不联网
python -m framework_engineer.third_party_solver.cli resolve --config <cfg> --dry-run
# 复核 manifest：sgl_kernel_version_mismatch 是否 false、各库 version/ref 是否合理

# ② 正式跑：执行 clone（进度打印到 stderr，JSON 小结打印到 stdout）
python -m framework_engineer.third_party_solver.cli resolve --config <cfg>
```

代理：config 里写 `https_proxy` 或加 `--https-proxy`。输入配置见 [configs/resolve_third_party.example.py](../configs/resolve_third_party.example.py)：`service_cmds`（必填）/ `sglang_repo_root`（必填，含 `sgl-kernel/`）/ `third_party_cache`（必填）/ `output_root`（必填）/ `workload_cmds`·`explicit_paths`·`extra_env`·`https_proxy`（可选）。

---

## 核心三步（本 skill 的方法论）

### 步骤 1 — 明确所有需要的仓库（固定 universe）

GPU 推理相关的算子/通信库是**有限集**，硬编码在 [`registry.py`](../third_party_solver/registry.py) 的 `UNIVERSE`，不做运行时发现。每条登记 `name / archetype(F*) / version_source / url / url_kind / on_default_path / backend_flags / ref_template`。

- `on_default_path=True`：即使启动命令不显式指明也会被用到的库（如 flashinfer / deep_gemm / cutlass / mscclpp）。**必须包含**，否则漏依赖。
- `backend_flags`：只用于给 manifest 注解 `triggered_by`，**绝不用来裁剪 clone 范围**——一律对整个 source-bearing universe 定版本 + clone（宁多勿漏）。
- **故意不在 universe 的**：`fla` / `causal_conv1d`（sglang/sgl-kernel 自带，F1/F2，不是外部仓库）、`triton_kernels`（随包纯 python，非编译进 `.so`）。它们由 `locate-kernel-source` 就地定位。
- **F8（下载的 cubin，如 `flashinfer_cubin`）**：无源码，永远 `failed`，不 clone。

### 步骤 2 — 找版本线索（严格二分：install 好的 vs sgl-kernel 编译配置）

每个库的"正确版本"来源分两桶，**必须区分**，因为它们的版本线索在不同地方：

**Bucket A — 独立 pip 包（运行时 import 的）→ 线索在 `pip`/`importlib`**
适用 `flashinfer` / `deep_gemm` / `flash_attn_4` 等。
```python
importlib.metadata.version(dist_name)   # 以“装好的”为准，不读 pyproject 声明（可能是范围）
```
> `dist_name` ≠ import 名。已核实：`flashinfer`→`flashinfer_python`、`deep_gemm`→`sgl-deep-gemm`、`flash_attn_4`→`flash-attn-4`（import 名 `flash_attn`）。查法：`importlib.metadata.packages_distributions()` 反查，或 `pip show <dist>`。

**Bucket B — 编进 sgl-kernel `.so` 的（无独立 pip 版本）→ 线索在 sgl-kernel 构建配置**
适用 `flash_attn(sgl-attn)` / `flash_mla` / `cutlass` / `mscclpp` / `flashinfer_embedded`。
```
1. importlib.metadata.version("sglang-kernel")  # 运行时真实版本，锚点（分发名 sglang-kernel）
2. 用它对齐 sgl-kernel 源码树：校验 sgl-kernel/python/sgl_kernel/version.py == 已装版本
   （不一致 → manifest 标 version_mismatch=true，不自动 checkout）
3. 读该源码树的 FetchContent pin（cmake_pins.py 解析）：
     CMakeLists.txt        -> sgl-attn / flashinfer(norm) / cutlass / mscclpp
     cmake/flashmla.cmake  -> FlashMLA
   pin 里的 URL 直接写明 sgl fork 还是官方（sgl-project/... vs NVIDIA/...）
```
> ⚠️ 断层：装好的 `sgl_kernel` wheel **不含 CMakeLists**，所以 fetch pin 必须从**对应版本的源码树**读，不能从 wheel 读。这就是为什么 config 要 `sglang_repo_root`。

### 步骤 3 — 版本线索 → clone ref 的映射规则

拿到的"版本线索"有三种形态，映射成 git `ref` 的规则各不同（由 registry 的 `ref_template` + [`_format_ref`](../third_party_solver/version_resolver.py) 承载）：

| 线索形态 | 来源 | 映射成 ref | 例子 |
| --- | --- | --- | --- |
| **commit hash** | Bucket B 的 CMake pin | **直接用**（不加工），cache 目录名取前 12 位 | `bcf72ccc6816...` → fetch/checkout 该 commit |
| **规范 release tag** | Bucket A，pip 版本恰好等于 tag | 默认模板 `v{version}` | `0.1.2` → `v0.1.2`（deep_gemm、大多数库） |
| **非规范 tag（需变换）** | Bucket A，pip 版本 ≠ tag 拼法 | 自定义 `ref_template` + 占位符变换 | FA4：pip `4.0.0b17` → tag `fa4-v4.0.0.beta17` |

**变换规则的表达**（`ref_template` 可用占位符）：
- `{version}`：原始 pip 版本（如 `4.0.0b17`）。
- `{pep440_beta}`：把 PEP440 的 `bN` 归一成 `.betaN`（如 `4.0.0b17`→`4.0.0.beta17`）。
- FA4 因此写成 `ref_template="fa4-v{pep440_beta}"` → 自动得到 `fa4-v4.0.0.beta17`；环境升级到 beta18 会自动跟着变，不会静默拿旧版。
- `ref_template=None`：无可用 tag 时 clone 默认分支（丢版本精度，最后手段）。

> **加新库/新规则的正确姿势**：先 `git ls-remote --tags <url>` **实际查**该库的 tag 命名（不要凭"beta 大概没 tag"猜——deep_gemm 和 FA4 都是查了才发现有 tag），再决定 `ref_template`。若现有占位符表达不了新的命名变换，在 `_format_ref` 加一个占位符即可。

---

## install 好的 vs clone 下来的：目录差异 & 为什么统一 clone

即使某库已 pip 装好，也**一律按正确版本 clone git 源码**，不用它的 site-packages 目录。因为 **wheel ≠ git 源码树**：

| 内容 | git 仓库（clone 后） | pip wheel（installed） |
| --- | --- | --- |
| kernel 源码（`.py` DSL / triton） | 有 | **有**（纯 python 逐字打包） |
| JIT 用的 C++ 源 `.cu/.cuh` | 顶层 `csrc/` | 挪到包内 **`<pkg>/data/csrc/`**（路径不同！） |
| tests / benchmarks / examples | 顶层 `tests/`、`benchmarks/` | **通常被剥掉** |
| 3rdparty 子模块 | 顶层 `3rdparty/` | 可能裁掉 |
| Bucket B 库（fa/flash_mla/cutlass…）的源 | 完整 | **不作为独立包存在**；同名包多是无关库 |

三条后果,决定了必须统一 clone：
- **JIT 溯源路径会错**：下游读 `gen_jit_spec(sources=[FLASHINFER_CSRC_DIR/...])`，installed 里指向 `data/csrc/`、clone 里指向顶层 `csrc/`。统一成 git 布局，下游只处理一种。
- **拿不到 tests**：`problem_translate` 做 L4 参考要抄 tests，wheel 里没有。
- **同名不同物**：`cutlass`→`nvidia_cutlass_dsl`、`flash_attn`→FA4——版本内容都不对。

代价是多占磁盘（实测整套 ~350M），相对模型可忽略。仅 **P1（用户 `explicit_paths`）** 和 **P2（sgl-kernel 内嵌 git 树）** 会跳过 clone。

### JIT kernel 的源码去哪找（感知 install/clone 差异）

JIT 不生成新源码，它 nvcc/DSL 编译一组**已提交在仓库里的文件**，`~/.cache/.../cached_ops` 里只有编译产物（无源）。溯源永远读 `gen_xxx_module()` 的 `gen_jit_spec(sources=[...])` 列表：
- 路径锚点 `FLASHINFER_CSRC_DIR` 等——**installed 指 `<pkg>/data/csrc/`，clone 指顶层 `csrc/`**，这是 install/clone 的关键差异。本 skill 统一 clone 后，下游一律按 clone 的顶层布局找。
- 走下载 cubin 的 op（F8）→ 无源，`failed`。
（这套定位是 `locate-kernel-source` 的机制②，本 skill 只负责把正确版本源码放到磁盘。）

## Agent 兜底（仅规则外）

正常路径无需 agent。仅当 CLI 产出异常时介入，且**只做规则维护，不改运行环境**：
- `version_mismatch=true`：sgl-kernel 源码树版本 ≠ 装的版本 → 判断是否让用户 checkout 匹配版本的源码树，或提示核对；**不盲目 clone**。
- 某默认路径库 `package not installed`：提示环境不完整。
- 新库 / tag 命名变化：`git ls-remote` 查证后往 `registry.py` 加条目或调 `ref_template`。
- clone 失败：只确认已如实记录（`clone_failed` + `clone_command`），**不重试/不换镜像**。

## 输出

- `<output_root>/third_party_manifest.json`：每条 repo 记录 `name / archetype / version / version_source / clone_source / resolution / local_path / url / ref / triggered_by / on_default_path / version_mismatch / status / clone_command / evidence`。顶层含 sgl-kernel 版本对齐信息 + `failed` 汇总。
- `<output_root>/missing_repos.md`：仅当有 `clone_failed` / `failed` 项时生成，逐条给原因 + 可复跑 clone 命令。
- `<third_party_cache>/<name>/<version>/`：clone 下来的完整 git 源码树（已有源码的 wheel 不复用，见上）。

## 完成标准

- manifest 中所有 `status: ok` 项的 `local_path` 可读且是**完整 git 源码树**（非 wheel、非纯编译产物）。
- 所有 Bucket B 项的 `ref` == sgl-kernel CMake pin；`clone_source` 正确区分 `sgl_fork` / `official`。
- 无源库（F8）落在 `failed`；clone 失败库落在 `clone_failed` 且带可复跑命令。
- `version_mismatch` 已复核并向用户说明（若为 true）。

# Skill: resolve-third-party

Step 0.5 的第一个 skill。目标：**确定 sglang 运行时依赖的所有 GPU 推理相关 third-party 仓库的正确版本，并把环境里缺源码的仓库 clone 到本地缓存**，产出一份 `third_party_manifest.json`（`name → 正确版本本地路径`）+（若有失败项）`missing_repos.md`。

本 skill **不做**源码到 kernel 的精确定位——那是第二个 skill `locate-kernel-source` 的职责。本 skill 只保证「正确版本的仓库源码出现在磁盘上」。

## 职责边界

- **做**：定版本（两桶）、判断环境里是否已有源码、缺的 clone 到 `(name, version)` 缓存、如实记录成功/失败。
- **不做**：解决 clone 失败（网络/tag 缺失）；重编 sgl-kernel；改 sglang 源码；装新包；把 clone 链接进运行环境；精确定位 kernel 源码。

## 输入（config）

用户提供一个 `.py`（或 json/yaml）配置，参考 [configs/resolve_third_party.example.py](../configs/resolve_third_party.example.py)：

- `service_cmds`（必填）：N 条 `{"backend_name", "cmd"}`，每条对应一条 backend 启动路径。
- `sglang_repo_root`（必填）：sglang 源码根，**必须含 `sgl-kernel/` 源码树**（读 CMake pin 用）。
- `third_party_cache`（必填）：clone 落地根，内部按 `(name, version)` 分目录。
- `output_root`（必填）：manifest + missing_repos.md 落地目录。
- `workload_cmds` / `explicit_paths` / `extra_env`（可选）。

## 底层机制（helper 已固化，agent 不必重造）

确定性逻辑全部在 `framework_engineer/third_party_solver/` CLI 里：

- **固定 universe**（`registry.py`）：GPU 推理相关库是有限集，硬编码。`on_default_path` 标注「即使启动命令不显式指明也会用」的库；`backend_flags` 只用于注解 `triggered_by`，**绝不裁剪 clone 范围**（漏掉默认路径库是大忌）。
- **版本二分**（`version_resolver.py`）：
  - Bucket A（独立 pip 包，如 flashinfer/fla/deep_gemm）：`importlib.metadata.version(dist_name)`，以**装好的**为准。
  - Bucket B（编进 sgl-kernel `.so` 的 fa/flash_mla/cutlass/…）：读 sgl-kernel 源码树的 `FetchContent` pin（`cmake_pins.py`）。**先校验源码树版本 == 已装 `sgl_kernel` 版本**，不一致只在 manifest 标 `version_mismatch: true`，**不自动 checkout**。
- **clone 三态**（`cloner.py` + `manifest.py`）：`ok`（本地路径）/ `clone_failed`（空路径 + 可复跑 `clone_command`）/ `failed`（无源如 F8，或定位不到）。clone 失败**不重试、不解决**。

## Agent 工作流程

1. **Preflight**：确认 config 可加载、`sglang_repo_root/sgl-kernel` 存在、`third_party_cache` 可写。在**装了 sglang/flashinfer 的目标 python 环境**里运行（否则 Bucket A importlib 全查不到）。

2. **先 dry-run 复核版本**：
   ```bash
   python -m framework_engineer.third_party_solver.cli resolve --config <cfg> --dry-run
   ```
   读产出的 `third_party_manifest.json`，重点检查：
   - `sgl_kernel_version_mismatch`：若为 `true`，说明用户给的 sgl-kernel 源码树版本 ≠ 环境里装的 `sgl_kernel`，Bucket B 的 pin 可能对不上运行的 `.so`。此时**判断**是否让用户 checkout 匹配 tag 的 sgl-kernel 源码，或提示用户核对——**不要盲目往下 clone**。
   - Bucket A 项版本是否都解析出来（`status != failed` 且 `version` 非空）；若某默认路径库 `package not installed`，提示用户环境不完整。

3. **正式 resolve（执行 clone）**：
   ```bash
   python -m framework_engineer.third_party_solver.cli resolve --config <cfg>
   ```
   universe 全量 clone（有源码的都拉），不按 flag 裁剪。

4. **失败只记录不解决**：CLI 已把 clone 失败写成 `status: clone_failed` + `clone_command`、`local_path` 留空，并生成 `missing_repos.md`。agent **不尝试修复**（不换镜像、不重试），只需确认这些项如实落进了 manifest，并在给用户的总结里点名。

5. **收尾**：复核每个 `status: ok` 项的 `local_path` 真实可读；把 `missing_repos.md` 里的待办转达用户。

## 输出

- `<output_root>/third_party_manifest.json`：每条 repo 记录 `name / archetype / version / version_source / clone_source / resolution / local_path / url / ref / triggered_by / on_default_path / version_mismatch / status / clone_command / evidence`。顶层含 sgl-kernel 版本对齐信息 + `failed` 汇总。
- `<output_root>/missing_repos.md`：仅当有 `clone_failed` / `failed` 项时生成，逐条给原因 + 可复跑 clone 命令。
- `<third_party_cache>/<name>/<version>/`：clone 下来的仓库（已有源码的库不落这里，只在 manifest 记录既有路径）。

## 完成标准

- manifest 中所有 `status: ok` 项的 `local_path` 可读且是源码树（非纯编译产物）。
- 所有 Bucket B 项的 `ref` == sgl-kernel CMake pin；`clone_source` 正确区分 `sgl_fork` / `official`。
- 无源库（F8）落在 `failed`；clone 失败库落在 `clone_failed` 且带可复跑命令。
- `version_mismatch` 已复核并向用户说明（若为 true）。

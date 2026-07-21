# resolve-third-party：单独运行指南（README）

面向使用者。说明**需要准备哪些输入**、**如何启动这个 agent 单独跑**、以及**跑完得到什么**。

- 想让 **agent 单独跑** → 用入口 prompt：[resolve_third_party_agent_entry.md](./resolve_third_party_agent_entry.md)
- 想了解**方法论 / 运维细节**（怎么定版本、加新库） → [resolve_third_party.md](./resolve_third_party.md)
- 只想**自己敲 CLI**（不走 agent）→ 见下方「方式 B」。

---

## 这个 skill 干什么

kernel-agent 流水线 **Step 0.5 第一步**：确定 sglang 运行时依赖的 GPU 推理相关 third-party 仓库的**正确版本**，按版本 clone 到本地缓存，产出：

- `third_party_manifest.json` —— `name → 正确版本本地路径`（下游 `source-locate` 的输入）。
- `missing_repos.md` —— 仅当有失败项时生成，逐条给原因 + 可复跑 clone 命令。

它**只**保证「正确版本的源码出现在磁盘上」，不做 kernel 源码定位。

---

## 一、需要准备的输入

### 1. 运行环境（关键）

必须在**已安装 sglang / flashinfer 等的目标 Python 环境**里运行。原因：一部分库（Bucket A，如 flashinfer / deep_gemm）的版本要靠 `importlib.metadata.version(...)` 读「已装版本」，环境不对就全查不到。

### 2. 一份 config 文件

**参考模板（务必先看）**：[`kernel_agent/framework_engineer/configs/resolve_third_party.example.py`](../configs/resolve_third_party.example.py)

复制它改成你的值即可：

```bash
cp kernel_agent/framework_engineer/configs/resolve_third_party.example.py my_resolve_cfg.py
# 然后编辑 my_resolve_cfg.py，把下表的必填字段改成你环境里的真实值
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | :---: | --- |
| `service_cmds` | ✅ | list，每条 `{"backend_name": ..., "cmd": ...}`，是一条 sglang 启动命令（可给多条 backend 路径）。 |
| `sglang_repo_root` | ✅ | sglang checkout 根目录，**必须包含 `sgl-kernel/` 源码树**（Bucket B 库靠它读 CMake pin 定版本）。 |
| `third_party_cache` | ✅ | clone 目标目录（内部按 `(name, version)` 建子目录，可复用）。 |
| `output_root` | ✅ | `third_party_manifest.json` 和 `missing_repos.md` 的落盘目录。 |
| `workload_cmds` | ⬜ | 压测/测试命令，仅 agent 需要运行时确认时用。 |
| `explicit_paths` | ⬜ | `{name: 本地路径}`，用户手动指定、覆盖自动定位（P1）。 |
| `extra_env` | ⬜ | 额外环境变量，如 `{"PYTHONPATH": ...}`。 |
| `https_proxy` | ⬜ | 所有 git clone 用的代理（也可命令行 `--https-proxy` 传）。 |

一份**最小可跑**示例（只填必填项，完整版见上面的模板文件）：

```python
# my_resolve_cfg.py
service_cmds = [
    {
        "backend_name": "flashinfer",
        "cmd": (
            "python3 -m sglang.launch_server --model-path /data/models/Qwen3.5-9B/ "
            "--host 127.0.0.1 --port 8080 --attention-backend fa3 "
            "--tensor-parallel-size 1 --disable-radix-cache"
        ),
    },
]
sglang_repo_root = "/sgl-workspace/sglang"      # 必须含 sgl-kernel/
third_party_cache = "/data/third_party_cache"   # clone 落盘处
output_root = "/data/step0_5_out"               # manifest 落盘处

# 可选
# https_proxy = "100.68.160.173:3128"
# extra_env = {"PYTHONPATH": "/sgl-workspace/sglang/python"}
```

> config 也支持 `.json` 或 YAML 子集；`.py` 为推荐格式（与框架其余配置一致）。

---

## 二、如何启动

### 方式 A：让 agent 单独跑（推荐）

把入口 prompt [resolve_third_party_agent_entry.md](./resolve_third_party_agent_entry.md) 整段作为**首条 prompt** 投喂给一个独立 agent，并告诉它你的 config 路径。agent 会自动：dry-run → 复核 manifest → 正式 clone → 汇报结果，并在 `version_mismatch`、新库等异常时停下问你。

启动示例（把 `<cfg>` 换成你的 config 绝对路径）：

```
你的任务见 skills/resolve_third_party_agent_entry.md。
config 路径：<cfg>
请开始。
```

### 方式 B：自己敲 CLI（不走 agent）

正常路径本就是两条命令，你也可以手动跑：

```bash
# ① dry-run：定版本 + 写 manifest（含可复跑 clone 命令），不联网
python -m framework_engineer.third_party_solver.cli resolve --config <cfg> --dry-run

# 复核 <output_root>/third_party_manifest.json：
#   - 顶层 sgl_kernel_version_mismatch 是否为 false
#   - 各库 version / ref 是否合理（Bucket B 的 ref 应等于 sgl-kernel CMake pin）

# ② 正式跑：执行 clone
python -m framework_engineer.third_party_solver.cli resolve --config <cfg>
```

可选参数：
- `--https-proxy <host:port>`：覆盖 config 里的 `https_proxy`。
- `--clone-timeout <秒>`：单个 clone 超时（默认 600）。

约定：进度打到 **stderr**，JSON 小结打到 **stdout**。返回码 `0` = 配置/环境 OK（clone 失败不算 CLI 失败，记在 manifest 里）；`2` = 硬错误（如配置缺失、`sgl-kernel source tree not found`）。

---

## 三、跑完得到什么

```
<output_root>/
├── third_party_manifest.json   # 权威产物：每个库的 name/version/ref/local_path/status/...
└── missing_repos.md            # 仅当有 clone_failed / failed 项时生成

<third_party_cache>/<name>/<version>/   # 每个成功库的完整 git 源码树
```

**验收要点**：
- `status: ok` 的项，`local_path` 可读且是**完整 git 源码树**（不是 wheel、不是纯编译产物）。
- 无源库（F8，如 `flashinfer_cubin`）落在 `failed`——符合预期。
- clone 失败的库落在 `clone_failed`，`missing_repos.md` 里有可复跑命令。
- 若 `sgl_kernel_version_mismatch=true`：说明你的 sgl-kernel 源码树版本 ≠ 已装版本，需 checkout 匹配版本后重跑（agent 会提示，不会盲目 clone）。

---

## 四、常见问题

- **Bucket A 库版本全查不到 / `package not installed`**：环境不对，确认在装了 sglang/flashinfer 的那个 Python 环境里跑。
- **clone 超时/网络失败**：配 `https_proxy` 或加 `--https-proxy`；失败项已记进 manifest，可用其中的 `clone_command` 手动复跑。
- **加了新库、或某库 tag 命名变了**：先 `git ls-remote --tags <url>` 查真实 tag，再按 [resolve_third_party.md](./resolve_third_party.md)「步骤 3」往 `registry.py` 补条目 / 调 `ref_template`。

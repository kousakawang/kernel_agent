# Source Locate Agent 使用说明

Source Locate 在 KID 已确定 semantic low-level target 后，为每个 target 定位四层源码，并抽取成可供
下游独立阅读的源码包：

- `interface_definition`：semantic Python interface；
- `py_cpp_binding`：Python→C++/FFI/JIT 桥接链；
- `kernel_impl`：host launcher→dispatch/template→核心 device kernel；
- `kernel_header`：独立声明/API header。

用户入口统一为：

```text
一个 source_locate_config.json
  + framework_engineer/prompts/start_source_locate.md
```

用户不需要手动依次执行 locate、启动中间 Agent、finalize 和 extract。宿主 Agent 读取入口 Prompt 与
配置后，会自主完成完整 workflow。

## 1. 快速启动

准备配置后，在能够访问本仓库和源码目录的宿主 Agent 中提交：

```text
请阅读并严格执行：
  <kernel_agent>/framework_engineer/prompts/start_source_locate.md

配置文件：
  /absolute/path/to/source_locate_config.json
```

入口 Prompt 会让 Agent 完成：

```text
config/KID/源码根预检
  → locate CLI
  → interface candidates
  → Agent 阅读并追踪四层源码
  → decisions + finalize
  → located schema + notes
  → extract CLI
  → extracted schema + kernel_sources
  → 完整 workspace 校验
```

## 2. 配置文件

配置合同为 `source-locate-agent-config/v1`：

```json
{
  "schema_version": "source-locate-agent-config/v1",
  "testcase_id": "all_backends",
  "kid_schema": "../../input/all_backends/decomposition.kid.schema.json",
  "third_party_manifest": "third_party_manifest.json",
  "sglang_repo_root": "/absolute/path/to/sglang",
  "workspace": "../../workspaces/all_backends"
}
```

相对路径以配置文件所在目录解析。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 固定为 `source-locate-agent-config/v1` |
| `testcase_id` | 当前 testcase 的安全目录名，只允许字母、数字、点、下划线和连字符 |
| `kid_schema` | KID Semantic Resolver 的最终 V3 schema |
| `third_party_manifest` | 允许读取并写成正式 hit 的第三方源码仓清单 |
| `sglang_repo_root` | SGLang 完整源码根；其中的 `sgl-kernel/` 会自动加入搜索范围 |
| `workspace` | 当前 testcase 独占的输出目录，不能位于任何被搜索源码仓内部 |

配置只声明真正的外部输入和 workspace，不逐项配置中间文件名。`prepare-run` 会确定性派生所有阶段
路径，从而保证不同 testcase 不会互相串目录。

当前可运行示例：

```text
example_kernels/source_locate_golden/config/all_backends/source_locate_config.json
```

## 3. 用户需要准备的依赖

### 软件

- Python 3；source-location 实现只依赖 Python 标准库；
- `rg`（ripgrep），供限定源码根的搜索 helper 使用；
- 能阅读 Prompt/Skill、访问本地文件并运行 shell 命令的宿主 Agent。

不需要 GPU、SGLang 服务、编译环境，也不需要安装或修改被定位的 Python package。

### KID V3 schema

输入必须是 `kernel-interface-decomposition/v3`，每个 kernel 至少具有：

- 唯一、安全的 `low_level_id`；
- 已确定的 semantic `interface`；
- `runtime_event.call_site.file/line`；
- `archetype` 与可空的 `provider`；
- 尚未出现 `locate_candidates`、`source_locations` 或 `kernel_sources_dir`。

Source Locate 必须原样保留 rank、interface、provider、archetype、kernel、metrics、coverage 和 call
site。即使 Agent 认为 semantic target 有误，也只能在 notes 中报告。

### Third-party manifest

只有 manifest 中 `status=ok` 且 `local_path` 存在的完整源码树会进入搜索范围：

```json
{
  "schema_version": 1,
  "repos": [
    {
      "name": "flashinfer",
      "status": "ok",
      "local_path": "/absolute/path/to/flashinfer"
    }
  ]
}
```

`local_path` 应指向完整 git 源码树，而不是 site-packages、wheel、`.so` 或编译缓存。pip 安装信息只
用于 third-party resolver 确定版本；Source Locate 使用 manifest 中的源码仓。

## 4. 标准 workspace

给定 `workspace=/path/to/workspaces/<testcase>`，Agent 固定生成：

```text
workspaces/<testcase>/
├── locate/
│   └── locate_candidates.schema.json
├── agent/
│   ├── source_locate_decisions.json
│   ├── located.schema.json
│   └── ref/locate_agent_notes.md
└── extract/
    ├── decomposition.extracted.schema.json
    └── kernel_sources/
        └── <low_level_id>/
            ├── interface_definition.py
            ├── py_cpp_binding/
            ├── kernel_impl/
            ├── kernel_header/
            └── read_hints.txt
```

新增 testcase 时使用相同名字的 config/input/workspace，不与其他 testcase 共用 workspace。

## 5. Agent 内部 workflow

以下步骤由入口 Prompt 驱动，不要求用户手工执行。

### Prepare

私有 `prepare-run` 读取配置并：

- 校验配置字段和相对路径；
- 校验 KID schema、manifest 与 SGLang root；
- 拒绝写入任何源码根内部的 workspace；
- 创建固定阶段目录；
- 返回全部绝对路径、target 列表和允许的搜索根。

### Locate

Agent 调用公开 `locate` CLI。它先解析 call-site import，再解析全限定 interface；支持 alias、relative
import、re-export 和 class method，不做全仓 leaf-name fallback。输出只是 interface candidates，
`resolved/ambiguous/not_found` 尚不是最终四层结论。

### Source analysis

Agent 对每个 target 调用私有 `inspect-target/search`，打开源码验证真实调用边，写入
`source-locate-agent-decisions/v1`：

- interface candidate 是否可信；
- Python→native/JIT/FFI 边界；
- host→dispatch/template→device kernel 调用顺序；
- 是否存在独立声明 header；
- gap 与人工 follow-up。

`provider/archetype` 只能提供阅读线索，不能用作分派或定位结论。

### Finalize

私有 `finalize` 校验 decisions、源码路径和行号，自动计算 `repo_hint`，删除临时 candidates，并生成：

- `agent/located.schema.json`；
- `agent/ref/locate_agent_notes.md`。

located schema 不包含 `kernel_sources_dir`。

### Extract

Agent 复制 located schema，再调用公开 `extract` CLI。extract 计算 definition end line、复制完整源文件、
生成缺失层占位和 `read_hints.txt`，并只在副本中增加 `kernel_sources_dir`。

### Validate

私有 `validate-run` 对完整 workspace 做最终校验：

- KID-owned fields 在所有 schema 中完全一致；
- located layers 与 decisions 一致；
- extracted schema 只比 located schema 多 `kernel_sources_dir`；
- 每个 target 的四层目录、占位和 `read_hints.txt` 存在；
- 所有正式 hit 仍位于允许的源码根。

只有 `validate-run` 返回 `ok=true`，Agent 才能宣布完成。

## 6. 四层最终状态

- `resolved`：真实调用关系已由源码证据闭合，必须有 hits；
- `best_effort`：已有可靠 hits，但动态 dispatch、生成代码或模板分支无法完整静态展开；
- `missed`：该层应存在，但允许 roots 中没有可靠 hit；
- `not_applicable`：源码证明该层确实不存在。

`interface_definition` 和 `kernel_impl` 不能使用 `not_applicable`。任何 `best_effort/missed` 都必须在
decisions 的 gaps 中解释；出现 `missed` 时必须填写 `manual_followup`。

## 7. 产物消费者

| 产物 | 主要用途 |
| --- | --- |
| candidate schema | Agent 验证 interface 的起点 |
| decisions JSON | finalize、审计和问题排查 |
| located schema | extract 前的正式四层结果 |
| locate notes | 人工查看证据、缺口和补仓建议 |
| extracted schema | problem_translate/workspace 下游索引 |
| `kernel_sources/` | 脱离原始仓库后仍可阅读的源码物料 |
| `read_hints.txt` | 每份完整文件中应重点阅读的 definition 范围 |

## 8. 常见失败

- `prepare-run` 失败：修正 config、KID schema、manifest 或 workspace 路径后重新启动；
- locate `not_found`：合法中间状态，Agent 继续搜索，缺仓时最终写 `missed`；
- manifest repo 被跳过：检查 `status=ok` 和 `local_path`；
- finalize 拒绝路径：把正确完整源码仓加入 manifest，不使用 site-packages/编译缓存；
- finalize 报行号越界：源码版本漂移，重新打开定义并修正 decisions；
- extract 失败：修正 located schema 对应的 decisions/source path，已有输出不会在预检失败时被清理；
- validate-run 失败：不得发布部分结果，修正根因并重跑对应阶段。

## 9. 开发与回归测试

两个公开 CLI 仍只有 `locate` 和 `extract`。`prepare-run`、`inspect-target`、`search`、`finalize`、
`evaluate` 和 `validate-run` 都是入口 Prompt 使用的私有 helper。

完整 golden：

```text
example_kernels/source_locate_golden/config/all_backends/source_locate_config.json
example_kernels/source_locate_golden/input/all_backends/decomposition.kid.schema.json
example_kernels/source_locate_golden/workspaces/all_backends/
```

运行 CPU 回归测试：

```bash
python3 -m unittest \
  framework_engineer.tests.test_source_location_agent_config \
  framework_engineer.tests.test_source_location_agent_helper \
  framework_engineer.tests.test_source_location_locate \
  framework_engineer.tests.test_source_location_extract
```

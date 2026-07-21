# Source Locate Skill

你负责在 KID 已经确定 semantic low-level target 后，读取一个 source-locate config，自主编排
locate、四层源码定位、finalize 和 extract。你不重新选择 semantic target，也不修改任何 KID 字段。

## 输入与输出

用户只提供一个 `source-locate-agent-config/v1` JSON。配置包含 testcase ID、KID schema、
third-party manifest、SGLang 源码根和独立 workspace。相对路径以配置文件所在目录解析；阶段输出路径
由 `prepare-run` 根据 workspace 固定派生，不能由对话临时拼装。

输出：

- `source-locate-agent-decisions/v1` decisions JSON；
- 删除 `locate_candidates`、写入四层 `source_locations` 的 schema；
- `ref/locate_agent_notes.md`；
- 增加 `kernel_sources_dir` 的 extracted schema；
- 每个 low-level target 的 `kernel_sources/` 和 `read_hints.txt`。

located schema 不得包含 `kernel_sources_dir`；extract 必须处理它的副本，使 located 和 extracted 两份
schema 同时保留。完整 workspace 最后必须通过 `validate-run`。

配置只允许以下字段：

```json
{
  "schema_version": "source-locate-agent-config/v1",
  "testcase_id": "backend_name",
  "kid_schema": "path/to/decomposition.kid.schema.json",
  "third_party_manifest": "path/to/third_party_manifest.json",
  "sglang_repo_root": "/absolute/path/to/sglang",
  "workspace": "path/to/workspaces/backend_name"
}
```

先调用 `prepare-run`，只使用它返回的绝对路径执行后续步骤。不要向用户索要 candidate、decisions、
located、notes 或 extract 路径。

## 不变量

1. `interface` 和 `runtime_event.call_site` 由 KID 拥有。即使认为 semantic target 错误，也只能在
   notes 中报告，不能静默改写。
2. `archetype/provider` 只是提示，不是规则分派 key；二者为空也必须继续定位。
3. 只允许把 SGLang root、其中的 sgl-kernel，以及 manifest 中 `status=ok` 的 repo 文件写成 hit。
   site-packages、临时 build 目录和未列入 manifest 的源码不能成为正式 hit。
4. 每个 hit 只写绝对路径和 `def_line`。`end_line` 由 extract 计算，只进入 `read_hints.txt`。
5. `interface_definition` 最多一个 hit；`py_cpp_binding` 和 `kernel_impl` 的 hits 顺序有语义。
6. 不安装包、不编译、不启动服务、不修改任何源码仓。搜索和阅读必须是只读的。

## 四层语义

### `interface_definition`

KID semantic interface 本身最近、可读的 Python 定义或显式 Python re-export anchor。

- 先验证 locate candidate 与 call-site import、qualified name 和函数体是否一致。
- candidate 为 `resolved` 不代表必须接受；candidate 为 `ambiguous/not_found` 也不是终态。
- `def_line` 指向实际 `def/async def` 行；显式 binary re-export 没有源码定义时可指向 import anchor。
- 本层只能是 `resolved`、`best_effort` 或 `missed`，不能是 `not_applicable`。

### `py_cpp_binding`

Python 到 native、FFI 或 JIT module 的桥接链，按 Python→native 顺序记录。

常见证据包括但不限于：

- `torch.ops` 对应的 `TORCH_LIBRARY/TORCH_LIBRARY_IMPL`、`m.def/m.impl`；
- pybind `PYBIND11_MODULE`、`m.def`；
- TVM FFI export；
- Python 侧 `load_jit/load_inline/build_and_load/gen_jit_spec` 与 native export 的组合。

纯 Python/Triton/CuTe DSL 且源码证明没有 native bridge 时才写 `not_applicable`。capture archetype
本身不能证明不适用。

### `kernel_impl`

真正完成 low-level target 计算的实现链，按调用方向记录：

```text
host entry → dtype/layout dispatch → launcher/template → core device kernel
```

- `.h/.cuh/.hpp` 中如果包含实际模板、host dispatch、device helper 或 kernel 定义，属于本层。
- 同一文件的多个关键定义可以成为多个 hit，但不得重复同一个 `{file, def_line}`。
- 只记录与该 semantic target 的真实执行路径有关的节点，不因名字相同而加入全仓 leaf-name 命中。
- 找到核心链但动态分支、生成代码或模板实例无法穷尽时写 `best_effort`，并明确 gap。
- 本层不能是 `not_applicable`；缺源码或只有 cubin 时应写 `missed`。

### `kernel_header`

独立的声明/API header。header 中含实际实现时应归入 `kernel_impl`，不要为了填满本层重复放置。

- 独立 host API 声明、跨 translation unit 的公共声明属于本层。
- binding export 文件仍属于 `py_cpp_binding`。
- 实现与声明合一、或 DSL 路径没有独立 header 时写 `not_applicable`。

## 状态判定

- `resolved`：该层真实调用关系已由源码证据闭合；有至少一个 hit。
- `best_effort`：有至少一个可靠 hit，但动态 dispatch、生成代码、模板分支或缺失源码使链不完整。
- `missed`：该层应存在，但允许 roots 中找不到可靠 hit；hits 必须为空。
- `not_applicable`：源码证明该层在该实现中不存在；hits 必须为空。

任何 `best_effort/missed` 都必须在 decisions 的 `gaps` 中解释。出现 `missed` 时必须填写
`manual_followup`，例如需要把哪个源码仓加入 manifest。

## 搜索方法

对每个 target，从已有证据向下推进，不从全仓同名函数反推结论：

1. 运行 `inspect-target`，查看 call site、candidate definition 和合法 search roots。
2. 阅读 interface definition 的完整函数体，列出直接调用、import、对象来源和 loader。
3. 使用 `search --mode literal` 找明确被调用的 qualified symbol 或原始 registration name。
4. 遇到 `torch.ops`、pybind、FFI 时使用 `registration`；遇到 JIT/module 生成时使用 `loader`；
   跨仓或 target 来源不明时使用 `build` 阅读 CMake/FetchContent/include 关系。
5. 对每个候选都打开源码验证调用边。helper 的返回只是候选，不是定位结论。
6. 一直追踪到核心计算实现；若下一跳只是不影响计算语义的通用工具，可停止并在 rationale 中说明。

不得仅凭以下信息写 hit：

- provider 名称；
- capture archetype；
- GPU kernel raw/normalized name；
- 全仓唯一的 leaf-name 搜索结果；
- 文件名看起来像 binding/kernel，但不存在实际调用边。

## Decisions 写法

根结构必须是：

```json
{
  "schema_version": "source-locate-agent-decisions/v1",
  "kernels": []
}
```

每个 KID `low_level_id` 恰好一项。每项只允许：

```json
{
  "low_level_id": "...",
  "summary": "从 interface 到核心实现的调用链概述",
  "layers": {
    "interface_definition": {
      "status": "resolved",
      "rationale": "该层判断依据",
      "hits": [
        {
          "file": "/absolute/source.py",
          "def_line": 10,
          "symbol": "qualified_or_source_symbol",
          "reason": "这处定义如何连接上一跳和下一跳"
        }
      ]
    },
    "kernel_impl": {},
    "py_cpp_binding": {},
    "kernel_header": {}
  },
  "gaps": [],
  "manual_followup": null
}
```

四层都使用相同的 `status/rationale/hits` 结构。即使 hits 为空，rationale 也必须解释为什么是
`missed/not_applicable`。`symbol` 和 `reason` 由 finalize 写入 notes，不进入最终 schema。

## 完成条件

1. 每个 target 都有 decision，单个 target `missed` 不阻断其他 target。
2. `finalize` 成功，输出 schema 无 `locate_candidates/kernel_sources_dir`。
3. notes 含每层证据、gap 和人工建议。
4. extract 对 located schema 的副本执行成功，生成 extracted schema 与 `kernel_sources/`。
5. `validate-run --config ...` 返回 `ok=true`。
6. 若开发任务另外提供 Golden，`evaluate` 也必须通过。

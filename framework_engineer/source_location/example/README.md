# Source Locate 两个 CLI 本机示例

完整的用户准备、Agent 启动和产物说明见上一级
[`README.md`](../README.md)。本文只保留当前开发机路径下的最短命令示例。

`locate` 只产生 Python interface 候选，`extract` 只消费 source_locate Agent 已确认的
四层 `source_locations`。两者都不按 `archetype/provider` 分派。

本目录的 `third_party_manifest.json` 提供当前示例使用的 FlashInfer、DeepGEMM 和
sgl-attn 本地源码路径；SGLang 根目录通过 `--sglang-repo-root` 显式传入。

## locate

在 `kernel_agent` 仓库根目录执行：

```bash
python3 -m framework_engineer.source_location.cli locate \
  --schema example_kernels/source_locate_golden/input/all_backends/decomposition.kid.schema.json \
  --manifest example_kernels/source_locate_golden/config/all_backends/third_party_manifest.json \
  --sglang-repo-root /Users/bytedance/Desktop/infra_agent/sglang \
  --out /tmp/locate_candidates.schema.json
```

`--out` 必填且不能等于 `--schema`。输出保持所有 KID 字段不变，只在每个 kernel 下增加
临时的 `locate_candidates.interface_definition`。唯一候选、歧义和未找到分别标记为
`resolved`、`ambiguous`、`not_found`；它们仍需 Agent 验证。

仓库内对应 golden 是
`example_kernels/source_locate_golden/workspaces/all_backends/locate/locate_candidates.schema.json`。

## source_locate Agent

Agent 入口 Prompt 和定位标准分别位于：

```text
framework_engineer/prompts/start_source_locate.md
framework_engineer/skills/source_locate.md
```

Agent 对每个 target 调用私有的 `inspect-target/search` helper，阅读并验证真实调用链，然后写入
`source-locate-agent-decisions/v1` decisions。最终通过 helper 合并结果：

```bash
python3 -m framework_engineer.source_location.agent_helper finalize \
  --schema /tmp/locate_candidates.schema.json \
  --decisions /tmp/source_locate_decisions.json \
  --manifest example_kernels/source_locate_golden/config/all_backends/third_party_manifest.json \
  --sglang-repo-root /Users/bytedance/Desktop/infra_agent/sglang \
  --out /tmp/located.schema.json \
  --notes-out /tmp/ref/locate_agent_notes.md
```

`finalize` 会删除临时 candidates、校验所有 hit 位于允许的 repo、检查行号、自动计算
`repo_hint`，并保证 KID 字段不变。私有 helper 不属于公开 CLI；查看其完整用法可执行：

```bash
python3 -m framework_engineer.source_location.agent_helper --help
```

Agent 到生成 located schema 和 notes 为止，不调用 extract。

## extract

Agent 确认四层结果并移除 `locate_candidates` 后执行：

```bash
cp example_kernels/source_locate_golden/workspaces/all_backends/agent/located.schema.json \
  /tmp/extracted.schema.json
python3 -m framework_engineer.source_location.cli extract \
  --schema /tmp/extracted.schema.json \
  --workspace-out /tmp/source_locate_workspace
```

命令创建 `kernel_sources/<low_level_id>/`、复制四层整文件、把计算出的 definition
结束行写入 `read_hints.txt`，并在 schema 中回填 `kernel_sources_dir`。`missed` 和
`not_applicable` 会生成说明占位；无效路径或行号会在清理旧输出前失败。

## 路径说明

manifest 使用当前开发机的绝对路径。仓库移动后需要同步修改其中的 `local_path` 和
`--sglang-repo-root`。`status != ok` 或路径不存在的 manifest 项不会进入搜索范围。

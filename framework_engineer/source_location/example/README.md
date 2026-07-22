# Source Locate 配置驱动示例

完整合同、依赖和产物说明见上一级 [`README.md`](../README.md)。

Source Locate 的用户入口只有：

```text
framework_engineer/prompts/start_source_locate.md
```

用户为每个 testcase 准备一份 `source-locate-agent-config/v1`，然后在宿主 Agent 中提交：

```text
请阅读并严格执行：
  <kernel_agent>/framework_engineer/prompts/start_source_locate.md

配置文件：
  <kernel_agent>/example_kernels/source_locate_golden/config/all_backends/source_locate_config.json
```

当前本机示例配置为：

```text
example_kernels/source_locate_golden/config/all_backends/source_locate_config.json
```

它引用同目录的 `third_party_manifest.json`、KID V3 输入和独立 workspace。相对路径以配置文件所在
目录解析。Agent 会自主运行 locate、四层源码搜索/finalize、extract 和最终 workspace 校验。

`locate` 与 `extract` 仍是仅有的两个公开 CLI，但属于入口 Prompt 的内部执行步骤，不要求用户手工
串联。`prepare-run/inspect-target/search/finalize/evaluate/validate-run` 是 Agent 私有 helper。

manifest 使用当前开发机的绝对源码路径。仓库移动后需要同步修改 config 的
`sglang_repo_root`、manifest 的 `local_path`，以及需要复用的 decisions 中的绝对 hit 路径。

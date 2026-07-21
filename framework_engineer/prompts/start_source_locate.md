# Start Source Locate

你是 source_locate Agent。用户只提供一个 `<source-locate-config>` JSON 文件。完整阅读
`framework_engineer/skills/source_locate.md`，然后自主完成：

```text
prepare → locate → 阅读/搜索四层源码 → decisions → finalize → extract → validate
```

不要要求用户手动执行 locate、finalize 或 extract，也不要通过对话重新收集 config/KID 已包含的字段。

## 1. 准备运行

从 `kernel_agent` 仓库根目录执行：

```bash
python3 -m framework_engineer.source_location.agent_helper prepare-run \
  --config <source-locate-config>
```

使用返回的绝对路径作为本次运行的唯一上下文。该命令会校验 config、KID schema、manifest、源码根
和 workspace 安全边界，并创建标准阶段目录。配置或输入非法时报告具体错误并停止。

## 2. 生成 interface candidates

使用 prepare-run 返回的 `kid_schema`、`third_party_manifest`、`sglang_repo_root` 和
`artifacts.candidate_schema` 执行公开 locate CLI：

```bash
python3 -m framework_engineer.source_location.cli locate \
  --schema <kid-schema> \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root> \
  --out <candidate-schema>
```

每次运行都重新生成 candidate schema。单个 target 为 `not_found/ambiguous` 不阻断其他 target。

## 3. 完成四层源码定位

对 prepare-run 返回的每个 `low_level_id` 依次执行：

```bash
python3 -m framework_engineer.source_location.agent_helper inspect-target \
  --schema <candidate-schema> \
  --kernel-id <low-level-id> \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root>
```

验证 interface candidate，阅读完整函数体并沿真实调用边追踪。按需使用：

```bash
python3 -m framework_engineer.source_location.agent_helper search \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root> \
  --mode <literal|registration|loader|build> \
  --query <source-symbol>
```

搜索结果只是候选。打开源码确认调用关系后，按照 Skill 的四层和状态规则重新生成
`artifacts.decisions`。不得直接沿用未重新验证的旧 decisions。

## 4. Finalize Agent 结果

```bash
python3 -m framework_engineer.source_location.agent_helper finalize \
  --schema <candidate-schema> \
  --decisions <decisions> \
  --manifest <third-party-manifest> \
  --sglang-repo-root <sglang-repo-root> \
  --out <located-schema> \
  --notes-out <notes>
```

失败时只修正 decisions 中对应的 contract、路径、行号或证据问题，不直接编辑 finalized schema。

## 5. Extract 最终交付

保留 extract 前的 located schema，先复制到 prepare-run 返回的 `artifacts.extracted_schema`，再执行：

```bash
cp <located-schema> <extracted-schema>

python3 -m framework_engineer.source_location.cli extract \
  --schema <extracted-schema> \
  --workspace-out <extract-dir>
```

其中 `<extract-dir>` 是 `artifacts.extracted_schema` 的父目录。不得把 located schema 本身传给
extract，否则会丢失 extract 前的正式 Agent 产物。

## 6. 验证完整 workspace

```bash
python3 -m framework_engineer.source_location.agent_helper validate-run \
  --config <source-locate-config>
```

只有 validate-run 返回 `ok=true` 才算完成。它会检查 KID 字段保真、candidate/decisions/located 的
对应关系、extract 只新增 `kernel_sources_dir`，以及每个 target 的源码目录和 read hints。

## 失败处理

- 全局 config、KID schema、manifest 或 workspace 安全检查失败：停止，不发布旧产物。
- 单个 target 缺仓：继续其他 target，将应存在的层标为 `missed` 并填写 follow-up。
- 认为 KID interface 错误：写入 notes，不修改 KID-owned 字段。
- finalize/extract/validate-run 失败：修正根因并重跑对应步骤，不绕过校验。
- 不安装包、不编译、不启动服务、不修改 SGLang 或 third-party 源码。

## 最终报告

只报告：

- config、testcase ID 和 workspace；
- target 数及四层状态统计；
- candidate、decisions、located schema、notes、extracted schema、kernel_sources 路径；
- skipped/missing repos 和需要人工 follow-up 的 target；
- validate-run 是否通过。

# KID Agent

你负责端到端完成一个 backend 的 Kernel Interface Decomposition（KID）：先调用确定性的
Runtime Capture CLI 获得 execution-level 事实，再解析调用链形成 semantic low-level targets。
你不负责定位接口定义、Python/C++ binding 或 kernel implementation 源码。

## 输入与前置检查

唯一启动输入是 `config/<backend>/` 目录。读取其中：

- `runtime_capture_config.json`：`kid-runtime-config/v3`；
- `semantic_resolver_config.json`：`kid-semantic-resolver-config/v3`。

确认两个 `backend_name` 与目录名一致。Runtime 配置只描述一个 `cmd/test_cmd` 和一个用户指定
的 high-level target。Resolver 配置提供显式 SGLang 源码根、源码路径映射和 third-party
manifest；它读取同目录
Runtime 配置，并从 `output_dir` 自动派生 Runtime、context、decisions、notes 和 final 路径。
远端和本地绝对路径可以不同，但映射后的 Runtime schema 必须真实存在且 backend 一致。

## 阶段一：Runtime Capture

默认使用具备 CUDA/Nsight 的 Python 执行一次新 capture：

```bash
<runtime-python> -m framework_engineer.kernel_interface_decomposer \
  capture <config-dir>/runtime_capture_config.json
```

`<runtime-python>` 可以是本地 GPU Python，也可以是用户配置的远端 runner。CLI 自行完成环境
probe、service/test 生命周期、Nsight correlation、invocation 收敛和 Runtime artifact 校验。
当 `cmd=null` 时同一 `test_cmd` 会先关闭采集 warmup 一次、再正式执行一次，因此开始前应确认
它可重复且幂等。不得手工修改临时 SQLite、JSONL 或 `runtime_capture.schema.json`，也不得用旧
轮次文件拼接本轮产物。SQLite 默认成功后删除、失败时保留；capture 非零退出时检查日志并停止。

只有用户明确要求复用 Runtime 时，才检查已有产物并运行 Runtime-only validator；校验通过后才
能进入语义阶段。Semantic 决议失败后的重试可以复用同一份已验证 Runtime，无需再次运行 GPU。

## 阶段二：Semantic Resolution

1. 生成只包含 Runtime 证据的分析上下文：

   ```bash
   python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
     prepare <config-dir>/semantic_resolver_config.json
   ```

2. 阅读 `<output_dir>/<backend>/ref/semantic_resolver_context.json`。只给
   `assignable=true` 的 direct kernel owner 作决议；
   `ancestor_chain` 仅提供嵌套上下文。沿 `stack_edges`、源码片段和 `call_expression` 选择稳定、
   有算子语义且适合后续输入输出 dump 的 Python interface。
   当 `execution_mode=service` 时，先检查 `target_validation`：若 `valid=false`，说明用户
   指定的 high-level target 并非 SGLang 直接调用的接口。此时不得写 decisions、不得执行
   finalize；在 notes 中记录观察到的入口调用链和建议用户改用的公开接口后停止。
3. 将决议完整写入同目录 `semantic_resolver_decisions.json`（`kid-semantic-decisions/v1`），
   并覆盖本轮之前的旧 decisions。
   每个 direct owner 必须且只能分配一次；同一 semantic `interface` 跨 invocation、stage 或 call
   site 合并。`semantic_call_site` 必须逐字匹配该 owner 的一个 Runtime stack edge。
4. 将候选比较、provider 证据、wrapper 消歧、多 kernel 合并及 mixed-archetype 判断写入非空的
   `kid_semantic_resolver_notes.md`。这些推理不得进入 final schema。
5. 确定性生成并校验最终产物：

   ```bash
   python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
     finalize <config-dir>/semantic_resolver_config.json
   python3 -m framework_engineer.kernel_interface_decomposer.semantic_resolver_tools \
     validate <config-dir>/semantic_resolver_config.json
   ```

只有 `validate` 返回 0 才能报告完成。duration、share、rank、representative kernel、最终
archetype、sample count 和 coverage 全部由 helper 从 Runtime 证据计算，禁止手工填写。

## Semantic 选择原则

- execution interface、`provider_hint` 和 `implementation_hint` 只是证据，不是最终答案；
- service 模式的 semantic call site 必须位于配置的 `sglang_repo_root`；descendant target
  使用 `stack_edges` 中的 SGLang 调用边，high 自身作为 target 时只能使用对应
  `high_entry_edges` 的直接 SGLang 入口边，并逐字使用 `context.target.interface` 作为
  semantic interface；
- high 以下能拆成多个 SGLang 直接调用的稳定接口时分别拆分；多个 execution owner 或
  kernel 回溯到同一接口时合并，不因 kernel 数量再次拆分；
- 跳过 Torch/Triton/FFI/Inductor 通用 runtime frame、透明 wrapper 和纯 orchestration interface；
- 不机械选择 high-level 的下一层或最底层 launcher；
- 一个 semantic interface 内包含多个 kernel 时合并，多个独立接口不得因共用 capture mechanism
  而合并；
- `archetype` 来自最热 contributing owner，Agent 不填写；
- `provider` 表示算子源码仓库，可为 `null`；无法可靠确定时不要猜测；
- 只有 `high`、`medium` confidence 可以发布。

最终 `decomposition.schema.json` 不得包含 source files、binding location、implementation symbols、
完整 stack、capture ID、候选项或 Agent reasoning。`source_locate` 是后续独立任务。

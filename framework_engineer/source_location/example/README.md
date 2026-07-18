# Locate Layer-1 CLI 本机示例

本目录提供一个可直接运行的 `third_party_manifest.json`。其中只保留当前
Layer-1 示例会使用、并且在本机确实存在的源码仓库：

- `flashinfer`：`/Users/bytedance/Desktop/infra_agent/flashinfer`
- `deep_gemm`：`/Users/bytedance/Desktop/infra_agent/DeepGEMM`
- `flash_attn`（sgl-attn）：`/Users/bytedance/Desktop/infra_agent/sgl-attn`

sglang 仓库不放进 manifest，而是通过 CLI 的 `--sglang-repo-root` 单独传入。

## 运行

在 `kernel_agent` 仓库根目录执行：

```bash
cd /Users/bytedance/Desktop/infra_agent/kernel_agent

python3 -m framework_engineer.source_location.cli locate \
  --schema example_kernels/to_fill_kid.json \
  --manifest framework_engineer/source_location/example/third_party_manifest.json \
  --sglang-repo-root /Users/bytedance/Desktop/infra_agent/sglang \
  --out /tmp/to_fill_after_layer1.json
```

命令成功时返回码为 `0`，并生成：

- enrichment schema：`/tmp/to_fill_after_layer1.json`
- 定位报告：`/tmp/ref/locate_report.json`

可以使用下面的命令查看结果：

```bash
python3 -m json.tool /tmp/to_fill_after_layer1.json
python3 -m json.tool /tmp/ref/locate_report.json
```

如果省略 `--out`，CLI 会原子更新 `--schema` 指向的文件。建议第一次运行时保留
`--out`，确认输出后再决定是否原地更新。

## 路径说明

这个 manifest 使用当前开发机的绝对路径。如果仓库被移动，请同步修改
`third_party_manifest.json` 中的 `local_path`，以及命令中的
`--sglang-repo-root`。manifest 中 `status` 不是 `ok` 或者 `local_path` 不存在的
仓库不会进入搜索范围。

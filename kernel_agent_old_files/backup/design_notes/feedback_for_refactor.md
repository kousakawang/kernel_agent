1. 优化目标里下面的三个配置仍然都需要手动指定：
target_file = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/fla/chunk_fwd.py"
target_line = 339  # Preferred: line inside the target function or on its `def`.
function_name = "chunk_gated_delta_rule_fwd_intra"  # Optional when target_line is set.
按照我们的设定，只需要target_file和target_line就够了。

2. 【严重问题】generate-harness CLI现在是不可用的
我把task_pack的结果放在kernel_agent/qwen35_gdn_extend_task_pack里了，你可以自行确认下。
我感觉当前生成的UT和benchmark都是没有原始实现的？

3. 其他（除了validate-task-pack需要跑通UT和benchmark没有测），我验证了是初步可以跑通的
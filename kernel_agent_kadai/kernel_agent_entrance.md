### 任务目标
定义kernel-agent的入口形式。

### 实际期望的落地形态：

framework-agent根据启动和测试命令从框架分理出热点算子的实现和UT/benchmark及相关数据(task-pack) -> kernel-agent拿到这组数据进行优化 -> framework-agent把算子回接回框架，测试在测试场景的收益（吞吐，TTFT/TPOT等）

### 当前的矛盾点：
kernel-agent的入口如何制定：

方案A： 采取当前的方案，找到实际框架里调用的kernel，围绕调用kernel的接口做输入输出/snapshot,构建UT和benchmark ->把这些打包给kernel-agent

优点： kernel-agent看到的算子/输入输出就是框架里实际跑的case
      基于已有实现的优化更直接
      framework测构造起来稳定，只要分离出当前框架的实现就可以

缺点： kernel-agent缺乏对原始问题的描述。看到的算子可能已经经过了一些计算逻辑上的优化（比如近似计算）
      kernel-agent无法在算子融合等方向上有较好的发挥。
      kernel-agent要接受不同类型的原始算子的实现，可能是pytorch/triton/cuteDSL/cuda/cutlass，而输出也需要考虑不同的实现路径。

方案B： 把算子的实现统一为pytorch的实现，然后基于pytorch构造UT和benchmark

优点： 对原始的实现有清晰的认知，可发挥空间更大包括算子融合，近似计算等

缺点： 需要重新开发框架直接可用的基于pytorch的算子后端 （要基于原始模型的实现重新实现，不是基于当前的triton等转换，需要保证实现和原始模型是一致的）
      用什么开发相对缺乏指导性，一般来说从容易开发的到开发困难的逐步尝试
      相比框架默认的实现的收益不明确，可能接回框架还是负收益。
      
      
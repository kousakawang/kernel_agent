## 主题：
原始接口拆解 + 提供pytorch example 功能的讨论

## 当前面临什么问题：
* 当前的目的： 通过一个framework_engineer agent 构造出优化模型推理时的用到的kernel的kernel_engineer agent 所需要的task_pack (例如kernel_agent/backup/examples)

* 当前的设计（如何定义问题规模）：当前会把用户在配置里写入的接口作为基础优化对象。如果是包含一个核心kernel的单一接口（low_level target）会支持把它作为优化对象。如果是一个包含多个核心kernel的接口（high_level target） 会通过CLI工具对其先进行拆解，拆解完后的每一个target(low_level target)作为优化的对象。 先不考虑算子融合。

* 当前task_pack里包含什么：核心包括snapshot（用于跑UT和benchmark的输入输出数据和参考性能数据），original_impl:原始的框架实现，env_probe： 用于检测当前环境里有哪些可用的开发工具，以及一些说明文档和固定格式文件

* 当前面临的问题：kernel_engineer 能参考的仅仅包含原始的框架实现，对于有的kernel，很难把问题描述清楚。需要一份pytorch原始实现作为参考来理解原始要实现的计算/通信问题。

* 当前打算落地的解决方案：从transformer里找一份原始的模型forward的实现（基于pytorch实现的推理），放到task_pack里作为参考。

* 解决方案的问题：因为transformers和sglang对于一个相同模块的推理的实现的切分粒度不一样，存在一些无法找到对应pytorch实现的算子。如果仅仅把整个模块的transformers的推理代码给到kernel_engineer, 它未必可以很好的理解问题规模。例如：对于GEMM/GroupGEMM/Norm这种 我们很容易把kernel需要解决的问题描述标准化，而对于chunk_gated_delta_rule_fwd_intra这种接口，乃至一些cuteDSL这类更复杂的实现，kernel_enginner可能会理解不到问题的本质，或者被原本的实现限制住了思路。

## 设计上需要重新讨论的问题：
前提： 
我觉得其实用high_level/low_level作为target分类并不严格。
实际上应该是 high_level / low_level + 问题定义明确（有明确的pytorch或者python实现）/问题定义模糊（只有cuda/DSL kernel可以参考），你觉得是否合理？

基本上有下面几条结论：
high_level(一整个模块)的推理过程是问题定义明确的 （尽管有radix- attention等不对齐的部分，整个计算流程一般可以在transformers里找到对应的pytorch原声实现作为参考）

low_level的推理过程可以分为明确和不明确：
明确的代表： GEMM/groupGEMM/Rmsnorm/rope这种， 我只用简短的语言就可以把问题定义清楚，原始的pytorch实现也很容易。

不明确的代表：chunk_gated_delta_rule_fwd_intra这种拆分出来的triton-kernel

我觉得：
1. 我们当前这种以low_level target作为基本优化单元，后续再考虑做融合的模式适合问题定义明确的kernel？

2. 总是以low_level target 作为基本优化对象的做法未必是正确的？（需要讨论）

3. 假定用户指定一个特定high_level目标：我们有哪些路径来做优化？
3-1: 像现在这样拆解成low_level target，基于low_level优化，同时把transformers里整个模块的推理代码作为参考。（后续可能会在得到若干个low_level target后插入fusion_planeer做问题合并）

3-2: 先判断模块里是否包含问题描述不明确的kernel（如何判断？是不是很难工具化只能靠AI）,如果不包含按照3-1，否则把整个模块作为优化对象，提供模块的sglang原始实现信息和transformers的实现作为参考

3-3: 3-2的复杂版：当包含问题描述不明确的kernel的时候，不是单纯把整个high_level target作为优化对象，而是结合transformers的实现和对模型推理的理解做拆分。拆分出来若干个问题描述明确的mid_level

3-4: 其他你觉得合适的处理办法是？

4. 假定用户指定一个特定的low_level target, 且问题定义不明确，我们要如何做？
这是最坏的case，没有周旋的余地。我们需要这个worst cade来考虑兜底方案。
我认为的兜底方案里，task_pack需要提供：
1. 原始实现 + 性能数据和输入输出golden
2. transformers的模块参考实现
你觉得呢？



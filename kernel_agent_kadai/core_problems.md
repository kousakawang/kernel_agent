只支持low_level or 从high-level 分解到 low-level存在什么问题？

问题一: 部分实现比较零散，缺乏原始的pytorch问题描述，kernel-engineer 根据原始的的kernel实现进行优化，优化思路可能受限。

问题二: 算子划分粒度层面无法再做优化，比如无法做新的算子融合，无法做算法层面的一些优化（快速近似计算路径等）


解决方案：

问题一:
思路1: 对于原始的每个实现，额外添加一层python/pytorch翻译agent，该agent不要求性能，只求实现一个pytorch/基础python的参考实现。（包括参考transformers等库的实现，能把实现直接抽取出来也可以）
思路2: 其实是思路1的放宽版本，先基于思路1做，当无法提供原始实现的pytorch/基础python参考版本时，能给出包含原始实现的上层模块的基础实现版本，然后大致定位出逻辑是这个版本的哪几行代码
思路3: （也就是之前说的mode2），当我们无法在指定的接口获得优化收益时，索性放弃逐个low-level的算子优化，转而去做mode2的优化

问题二：
思路（当前）：
其实就是当low_level target无法取得优化时，另外一种优化模式。同问题一的思路3。

输入是比较大的模块在transformers里的原始实现。
输出是对原始实现的替代。
按照当前的设想，需要做的事情是：
1. 找到原始需要优化的high_level模块在transformers里的对应实现（或者把原始的high-level适当分块，找到每个分块里transformers对应的实现）
2. 对原始transformers的实现做改造，让其核心资源（kvcache，indexer等）的表达形式和sglang一致。同时写UT和原始transformers的实现比较，确保pass。后续会根据选择有分歧。
   1. 路线1: 当需要和从原始框架的模块保存下来的数据做精度对齐时，需要同时把原始框架里的实现里的参数到transformers的参数做翻译，除了kvcache等已经适配了，一些其他的instance-tensor的表达形式也可有可能有区别需要翻译。保证两者跑的是一个相同的计算case。
   2. 路线2: 不需要和原始的框架保存下来的数据做精度对齐时，则跳过这一步。
 
3. 根据现有的改造后的transformers的实现做优化。
4. 把优化回接到框架里，核心资源以外有不一致的地方，框架做适配。

最大优势： 可以重新审视一个大模块的算子划分，但是这里的优化空间真的值得吗？以及重新划分粒度后，框架新增开发量和是否通用。最关键的是，上面的mode2的四步，每一步都有不小的工作量和不确定性。


思路（新增）：
1. 实际上，当我们当前要优化的实现找不到比较好的transformers实现作为参考时，可以尝试换条路径。（一般sglang对同一layer都有多条实现路径，flashinfer等的路径更容易和transformers对上）尝试找到这个模块的一条forward路径，它的实现大致能和transformers的实现路径对齐。（算子划分粒度上）同时又有框架的原生实现（接口等都不需要重新适配）。在这条路径上做优化，有比较完整的transformers参考。


关于mode2的更新：
说实话我有点倾向于完全取消mode2，改作下面两件事情。

1. 对于问题1的解决（low level算子语义不清晰的问题）
新增problem_translate agent（skill）,它的作用就是为一个high_level分解出来的每个low_level的target添加问题理解说明（或者配套），是对我们之前说的分为3个level的增强版。它会增加一个level。
在这套新的设计下，

level4（最优case）: 能生成基于pytorch/或者基础python实现的参考接口，能通过snapshot跑通UT，证明精度是对齐的。
level3 (次优case): 虽然无法生成完全一致的UT，但是可以找到计算逻辑上等价的transformers表达。（参考op_mapping_qwen3_5_gdn_sglang_transformers.md）
level2 (normal case)： 找不到一致的表达，但是能找到包含该low_level的transformers的高层模块的代码(但是不多，一般transformers对算子的分割力度会更细)
level1 (just try mode)：只有原始实现，没有参考实现

a.
对于用户直接指定low_level target，基本流程不变，只是多了level3。（为什么要新增一个level我后面会说明，主要还是为了high_level target的分解）

b.
对于用户指定high_level taget，
使用这个agent时，我会：
不仅仅跑默认的backend路径对于GDN而言就是triton，而是把硬件支持的路径（主要是针对attn的区别，一般会有2～3种）都跑一遍。
对于每个路径下high_level target 拆解出来的low_level target,我都会尝试去问题定义。
最后我会用一个公式去计算哪个路径对于kernel_engineer来说是最理想的。
这个公式可以理解为:
sum(lv_[i]_weight * l_[i]_kernel_time_cost/ total_kernel_time_cost) for i = 1,2,3,4
也就是说，给每个level的low_level_target 一个权重 lv4 = 4,lv3 = 3, lv2=2, lv1=1, 再看每个level的low_level在整体的模块里的耗时占比。用耗时占比 * 权重 得到一个 最终的kernel_engineer_prefer_score 我们选择这个score最高的route给kernel-agent优先优化。
当然，其余路径也不是会丢弃，当我们优先选择的的路径优化效果不理想时，我们会依次降级路径。

也就是说： 把某个high_level target拆分出来的low_level target语义不清晰的问题，通过切换路径来找出定义更清晰的low_level target。 从而缓解原始的痛点。此外，切换路径的另外的好处时，当切换到flash-infer/flash-attn等通用算子库时，往往接口本身需要兼顾各种框架，语义会天然比定制的triton-kernel之类清晰。且他们在原始的仓库里天然会有注释和UT可以参考(big point)。


2. 对于问题2： 算子划分粒度层面无法再做优化，问题描述的表达形式受限

多路径尝试的做法，也可以缓解该问题。
切换多路径来优化 = 更多的问题描述方式。
并且sglang/vLLM 作为专业的推理框架，本身就存在对计算接近最优的划分方式。扔掉这种参考自己重新去造轮子，ROI很低。


总结下来，也就是说我倾向于放弃mode2， 转而新增一个problem_translate agent（skill），强化对low_level target分级的功能。并且在整个work_flow上增加多路径尝试分解和记分步骤。（仅针对high_level target）
这样，我们可以放弃mode2这个逻辑不统一且复杂的路径，把整个工具的设计尽量统一。



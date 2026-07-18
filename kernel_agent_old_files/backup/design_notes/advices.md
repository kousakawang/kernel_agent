我又重复阅读了一下framework_engineer下面的内容，感觉有几个点需要修改一下：
### 1  
关于入口prompt，我建议提供一个标准化用户输入配置文件 （对应你的kernel_agent/framework_engineer/prompts/start_phase1_validation.md文件的line17-44） 用户直接把必要的信息填好，然后启动agent就可以，这里没必要再交互对话了。只有当用户给的信息不满足大于等于必填输入时，agent才给出反馈。

### 2 
kernel_agent/framework_engineer/skills/qwen35_linear_core_task_pack.md里的部分信息还是比较旧的，而且它里面的信息其他prompts里都有了，感觉好像没啥存在的必要了，是不是要把它和其他文档里对它的引用都删除掉

### 3
prompts里的两个文档的一些建议：
文档1： kernel_agent/framework_engineer/prompts/framework_engineer.md
1. 关于角色定位，可以再准确一点，严格来说，我们是对框架里需要优化的算子/接口做配套工程，使得需要优化的算子/接口和框架解耦开来，能让kernel_engineer无感知框架的前提下做算子/接口性能优化

2. 用户 Gate
任务启动前必须检查：
用户提供的服务启动命令必须可直接运行。
用户提供的 workload/test 命令必须可直接运行。
优化目标至少明确到某个 module forward，或明确到一个/多个 kernel/core 接口。
如果启动命令、workload 或优化目标不满足要求，输出明确错误并中断。Framework Engineer 在 Phase 1 没有义务修复用户服务脚本、数据集或环境问题。

这一段可以直接给出用工具check的方法，虽然另外一个文档里已经写了

3. Phase 1 八步流程: 
这里少了task_pack的初始化？

4. 职责边界：
建议保留负责即可，并且把接收 KernelDeliveryPackage 或 FrameworkChangeRequest。这个我们还没涉及到的部分标注为待实现避免误导

5. 完成标准：
我觉得 “并清楚知道要优化什么、怎么测、目标收益是什么时，Phase 1 框架侧交付才算完成。”这句话不妥当，framework_engineer不需要知道目标收益，其实唯一的标准就是执行validate-task-pack CLI 能pass。
所以我建议增强validate-task-pack， 把包括对task_pack是否包含必须内容的检查也包含进去，也就是说这个CLI需要包含：
a. 环境的确认
b. UT和benchmark的确认
c. task_pack 包含必须文件的确认

文档2: kernel_agent/framework_engineer/prompts/start_phase1_validation.md
这个文件基本没啥问题，唯一的改进建议是：
执行步骤（from line 57）
这里每个步骤执行完后看什么，确认什么，什么时候继续，什么时候中断的描述可以更详细一点

### 4 
mutable_arg_paths: <如果目标接口会原地修改输入，例如 kwargs.ssm_states> 这个配置，我建议不要。
因为framework_engineer 无法判断，用户也无法准确判断哪些接口会修改参数，最保险的做法是，执行完之后把输入也一并保存下来，correctness验证比较输入输出

### 5
我建议把framework_engineer的目录和kernel_agent的目录隔离，相互不可见。唯一的交互是约定好好的pack（避免阅读不需要的代码增加上下文负担）。这样我们framework_engineer 在使用时外部的第一层目录是framework_engineer 对应的python文件接口的路径之类要改下
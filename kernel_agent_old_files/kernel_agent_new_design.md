### 目标
设计一个可以根据其他agnet提供的目录，实现算子优化task的Kernel_agent工程

### 当前现状
前置工作已基本完成，/Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent/backup/examples
是一个参考，它是kernel_agent工作的入口，里面包含了kernel_agent需要的相关文件，包括UT，benchmark，当前的参考实现（baseline），可用开发环境检测（triton/cuda/cuteDSL等）以及一些其他说明

### 你的任务
结合当前的可参考工作目录，以及/Users/bytedance/Desktop/remote_dev_project/model_ana/kernel_agent/backup/reference_repos 这里的一些参考仓库或者其他你可以搜索到的仓库，帮我做一个kernel_agent的工程设计

要求明确以下几点：（可以先调研参考仓库是怎么做的，给一个简单的说明））
1. 基本agent工程的目录结构要如何设计，包括系统prompt，skill，知识库等
一般来说：
系统prompt包含对职责和现状的描述
skill包含一些profiling和log解析工具的可执行脚本/文件和使用说明

知识库包含对应硬件的spec信息，对应语言（cuteDSL等）的编程文档和参考代码。
但是这样的话，输入内容是不是太多了，其他仓库是怎么设计的？

2. agent工作的的流程设计
整个工作流程是怎样的？
比如：
如何制定优化目标（当用户没给出的时候）
有多种开发语言可选择时，如何选择，优先接入开源仓库的实现测试性能吗？
实现完后，如何tuning。
如何制定对优化后的kernel的评价体系？
停止工作的标准是什么？



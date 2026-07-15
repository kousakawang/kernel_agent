1. KID:kernel_agent/framework_engineer/kernel_interface_decomposer

职责：根据用户给的high_level_tagret 和启动及测试命令，把high_level_target里执行的符合我们要求的F0-F8 low_level_target的调用找出来并且分类。根据low_level_target的耗时给热点low_level_target排序
参考文档：
输入：用户给的配置文件（包含启动，测试命令，high_level相关的信息）
输出：schema.json(第一阶段，定位low_level_target的耗时和调用位置)

TODO：
1. 明确捕获机制，做UT （nsys or python代码直接分析？）
2. 明确耗时测试机制 （打nvtx tag 还是low_level_target加装饰器直接测试耗时）
3. 明确输出文件格式（参考当前文件）
4. 功能升级开发
5. 需要在GPU跑，用户测试。



2. locate layer-1 (CLI):kernel_agent/framework_engineer/source_location 
职责：根据KID提供的low_level_target,把基础的源码定位信息填写到相同的schema.json里
参考文档：
输入：third-party-solver提供的输出。（third_party_manifest.json，missing_repos.md），KID执行完的schema.json，sglang源码，third-party源码
输出：更新完后的schema.json和ref下面的locate_report.json

TODO:
1. 整理需要固化成CLI的场景
2. 改造当前实现，先在shcema.json里添加字段，把能填写上的填好（参考dry-run locate阶段）
3. 用当前的example_kernels下面的case直接测试（可以不依赖gpu直接测试）


3. locate layer-2 (agent):
职责：继续补全schema.json,把没有填好的信息补全。
参考文档：
输入：third-party-solver提供的输出。（third_party_manifest.json，missing_repos.md），KID执行完的schema.json，sglang源码，third-party源码
输出：更新完后的schema.json和ref下面的locate_agent_notes.md

TODO:
1. 先根据当前目录下仓库路径更新third_party_manifest.json里的路径
2. 新增agent角色开发（是否需要添加helper？）
3. 用当前的example_kernels下面的case直接测试（可以不依赖gpu直接测试）



4. problem_translate agent:

职责：根据locate执行完的schema.json, 保存下来的snapshot，third-party-solver提供的输出。等信息完成原始low_level_target到torch/原始python的转换。
参考文档：
输入：locate执行完的schema.json, 保存下来的snapshot，third-party-solver提供的输出。sglang源码/3rd-party源码。其他源码（如transformers）
输出：问题定级的json等
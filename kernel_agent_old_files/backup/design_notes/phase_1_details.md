* 我会在这里列出phase1的具体工作流程（我希望的），然后在需要实现/决策的地方特别说明，你可以拿你当前实现的内容和我的需求做比对，做对应的完善/修改。尤其注意【point】前后的内容。

phase1: 
* 目标：
我希望对Qwen3.5里的linear-attention模块进行优化（整个模块）

* 整个工作流的入口：
前提：
我会给出具体的模型的启动命令（包含部署方式，模型路径等），并且给出测试场景。
启动命令类似：
python3 -m sglang.launch_server --model-path /data01/models/Qwen3.5-9B --host 127.0.0.1 --port 8080 --mem-fraction-static 0.7 --cuda-graph-max-bs 128 --tensor-parallel-size 1 
测试场景类似：
python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts 48 --apply-chat-template --random-output-len 1 --random-input-len 0 --image-resolution 1080x1080 --image-format jpeg --image-count 1 --image-content blank --random-range-ratio 1 --max-concurrency 8 --host=127.0.0.1 --port=8070
或者一些其他agent可以直接运行的脚本

step1:
我会对framework engineer下达下面的命令：
根据我的启动命令和测试命令，整理出我对应测试场景下Qwen3.5-9B模型 linar-attention模块算子性能优化的需求，按照【约定好的标准】生成需要交付给kernel-engineer的产物。

step2:
framework engineer接受到命令后，会做如下几件事：
0. 按照我的启动命令启动服务，并且跑测试代码，确认case是可以跑通的（所有的前提），获取testcase的原始性能数据。
1. 阅读模型的推理代码，把对应的模块可以拆成哪几个算子提取出来
2. 根据框架代码构造对应算子的正确性UT，里面由pytorch或者triton实现提供golden，UT需要验证当前算子正确性和优化后算子的正确性，UT里优化后的算子只保留接口，但是测试逻辑要包含优化后的算子，kernel enginner 填完算子的实现后，UT可以直接运行。UT需要读取shape列表来测试。
3. 根据框架代码构造对应算子性能benchmark测试，里面有原始实现的性能测试和优化后的算子的性能测试，优化后的算子的只保留接口，但是测试逻辑要包含优化后的算子，kernel enginner 填完算子的实现后，性能benchmark可以直接运行。性能benchmark需要读取shape列表来测试。
4. 在框架里加入shape收集代码，运行我提供的测试命令，收集出来需要优化的算子shape，放到shape列表里，这个shape列表会被上述的2/3里的测试文件使用

在这里，有几件需要注意的事情：
1. 关于4的shape收集，如果可以根据原始测试文件推断出来算子的输入shape，可以不实际侵入框架加log，但是实际上对于一些动态serving，shape是根据调度结果决定的，所以我建议还是加入shape收集机制，【point1】你觉得这个shape收集机制应该怎么做，是让agent自由发挥吗。另外，需要注意跑shape收集时，需要关闭cuda-graph。然后因为我们不可能让shape列表过于发散，要对收集到的原始shape列表做处理，得到最终shape猎豹。

2. GEMM/MOE这种比较好构造UT和benchmark，有一些算子的输入可能不是单纯的torch.Tensor 而是一些封装好的其他结构（尤其是attention，需要load kvcache）。【point2】你能确认下是否存在这样的情况，并且思考要如何处理这样情况的UT和bencnmark构造吗

step3:
framework engineer调查当前环境，给kernel engineer提供算子实现的候选路径。
kernel engineer不能自己安装环境，而是要在framework enginerr提供的编程语言/DSL/三方库里做算子实现。
一个例子：framework engineer会给出如下信息：
```
你可以基于如下手段实现优化的算子：
DSL：
cuteDSL/Triton/cutile （给出import的脚本，证明可以使用）

三方库：
cuDNN，cublas，cutlass  （给出import脚本或者库的头文件，库文件路径）

最终手段：
手写cuda/PTX 代码 （给出cuda相关头文件，runtime库的路径）
```

step3后，framework engineer的第一步工作就完成了，他会按照“合作合同”，交付给kernel engineer一些产物。
其中包括需要优化的算子的描述，对应的算子的UT和接口，对应的算子性能benchmark和接口，以及需要check的shape列表，可用开发环境等。
这些信息会被收集到一个新的文件里，下一步，我会让kernel engineer阅读这个新的文件，开始做算子开发/优化工作。
【point3】，结合上述描述，你觉得为了完成step1~3里的工作需要为framework engineer准备哪些tool/skill/system prompt/data-base(knowledge)。
合作合同又要怎么写比较合适。

我个人觉得需要：
1. 把收集shape的工作标准化，包括形成最终的shape列表
2. 对一些比较复杂的UT构造做进一步约束，比如输入是kvcacheblock+ block_id table时，UT的输入参数要如何构造
3. benchmark内容的标准化，包括warmup，测试性能是否用cuda-graph
4. 开发环境说明的标准化，哪些DSL/三方库可以用，最好提供明确的证据（可以执行的脚本等）让kernel-engineer能直接验证，避免信息的mismatch。
5. （后面需要），添加profiling，找出热点算子，给算子的重要度排序。

step4: kernel engineer开始干活，他会只把framework给进的信息作为原始输入，尝试开始实现/优化算子。

step5：kernel engineer会根据现状，决定用什么方式开发/优化算子，分析可优化点，做开发/修改计划。

step6: kernel engineer会先跑UT，保证正确性，不正确就修改，直到能改对为止。

step7: kernel engineer会在UTpass后，跑benchmark，包括使用Nsight-compute等工具分析，包括使用auto-tune等工具对一些参数进行自动调优。

step8: kernel engineer会重复5～7，直到达到迭代收敛标准【point1: 迭代收敛标准要如何制作】

step9: kernel enginner会按照“合作合同”给framework engineer一份交付产物，包括算子优化的说明，以及实现UT和benchmark里的接口
       （后面再考虑）如果当前的实现对框架有特殊改动，也一并说明（比如需要swapAB/ 改变权重的排布格式等），同时，这份改动也需要反映在UT里，让framework engineer能直接参考

step9后，【point2】kernel engineer的基础工作完成，结合上述描述，你觉得为了完成step4~9里的工作需要为kernel engineer准备哪些tool/skill/system prompt/data-base(knowledge)。合作合同应该如何实现比较好？
我个人觉得需要：
1. profiling工具，编译器命令的使用说明（获取PTX等时需要）（特别对于非nvidia硬件）nvidia需要吗？（做成skill？）
2. 硬件的spec说明 （做成知识库？）
3. 卡间互联的说明（megamoe/通算融合等开发需要）（虽然也可以通过 nvidia-smi topo --matrix等获取拓扑，但是详细的带宽数据是不是也需要）（做成知识库？）
4. 一些基本的开发原则的标准化，比如如果使用C++开发，需要编译成library，bind出来一个python接口，或者采取loadlibrary等方式在UT/benchmark里调用，因为他们都是python脚本。以及开发的顺序，由简单到困难，先优先尝试成熟的开源库，并且评估优化空间，有空间再继续做后续的复杂开发（使用cuteDSL等）。对于需要tuning的算子（triton等）给出最佳参数。
5. 一个性能优化pointchecklist，把算子性能优化常见的点列出来供check(system prompt)
6. 对应语言/DSL的算子代码示例/开源库的文档等 （知识库）


step10: 我会让framework engineer开始验收优化成果。framework engineer的验证工作开始，他会拿着kernel engineer的产物，把优化后的算子接入进框架，重新测试性能，评估性能收益

step11:（后面再考虑），当offline的kernel优化，拿到框架里测试没有收益/负收益/收益远不达预期时，需要如何处理

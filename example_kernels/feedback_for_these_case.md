1. 功能错误：
   /Users/bytedance/Desktop/infra_agent/kernel_agent/example_kernels/kid_dry_run_out/workspaces/all_backends/kernel_sources/sgl_kernel_fa3_fwd/read_hints.txt line2 这里c++接口的边界定位错误，endline应该是1198


2. 设计上的一些讨论点：
   经过这一轮更真实的验证，（这几个不同backend的算子是接近真实场景的case，实际场景只会更复杂），我有一些想法上的更新。
   a. locate功能一定是**agent强依赖**的，CLI的局限性非常大。实际代码的实现非常复杂，除了部分固定文件的定位（比如sgl_kernel bind的部分，原始接口的定义部分）到了找kernel_impl，哪怕是triton或者cuteDSL这种纯python代码也不是靠CLI可以hold住的。因为CLI无法对多级实现的kernel做判断。它只能适合一个头文件一个cpp实现这种固定的模式。对于代码里的调用栈无法跟踪。 ->所以locate功能layer1我们要做什么要重新思考下，尽量做的薄一点。
   
   b. Bind这一层会有一些更复杂的case。从工程设计角度，可以抽象成c++实现到python实现的bind。但是根据AOT/JIT，不同的JIT方法，它对应的代码形态会差别比较大。也不一定是cc文件（sglang自带的是python）。这次的example_kernels/kid_dry_run_out/workspaces/all_backends/kernel_sources/flashinfer_batch_prefill_pagedcase和flashinfer_top_k_top_p_sampling证明了这一点。
   
   ->所以从描述完整的的c++到python实现的转换过程的角度，bingding这一层我们也应该做成目录，允许不同格式的文件，允许多文件

   c. kernel_header这一层不应该有歧义， 只找包含kernel_impl里涉及到的接口应的头文件，不引入其他东西 -
   
   > 这个最单纯

   d. kernel_impl: 问题最多，最复杂。 既然我们按照定义的F0-F8 级列表来定义low_level target， 就必须要意识到这个python low_level_interface下面其实一定是会有各种复复杂实现的。可能涉及核心算子判断，到核心算子为止的调用链路判断。此外出现在调用链路上的非核心接口的实现文件需要定位出来，给出readhint吗，也是个问题。我注意到你在kernel_agent/example_kernels/kid_dry_run_out/workspaces/all_backends/kernel_sources/flashinfer_triton_rms_norm里给出了scale_and_clamp的实现。所以哪怕是tritonkernel 也并不简单。 cuteDSL更复杂，c++代码的还涉及到模版匹配，接口重载，macro展开。 另外由于我们是静态阅读代码，对于代码里的ifelse分支的判断也会变的很困难。要怎么把这个kernel_iml写出来很难用一套标准化的规则来做。 
   
   -> 我想好好考虑下，layer3（extract） 做完后我们究竟应该交付什么。或者说我们对于extract出来的每个low_level_target的代码产物的作用的期待值需要降低。 这里分为两层： 对于translate_problem agnet,它可以看原始仓库，自己去探索实现，不需要被extact出来的产物限制住。对于kernel_engineer,它看不见原始仓库，所以必须依赖于extract出来的原始实现。
   但是你注意到没： 这里其实有一个隐藏的矛盾。
   当 translate_problem agent拥有完整的仓库的阅览和参考权限，都无法等效的实现出来一个pytorch实现时，kernel_agent根据残缺的本地tack_pack里的kernel实现（如我们上面所说，抽取出来文件可能会漏掉一些相关文件）更加难以做一版优化版的算子实现。
   所以这里**约等于当translate_problem agent翻译失败时，这个算子的优化几乎没必要尝试了**。

   不过我有一个好消息是，对于这种实现极其复杂的low_level_target,我们一般都能找到UT，/Users/bytedance/Desktop/infra_agent/kernel_agent/example_kernels/all_backends_sglang.py 这个文件我加了UT。
   而没有UT的算子往往比较简单。简单triton/torch原生这种。
   这就意味着，其实对kernel_agent而言，locate这一层其实没这么重要。
   
   所以我可能倾向于把locate layer2 作为能找到kernel_impl就找，尽力而为，但是不需要设定过于硬性的要求。
   layer3 extact主要是给kernel_agent多一手资料。而translate_problem主要参考layer2 填完的schema.json。参考从上往下的接口行号在仓库自行探索。但是复杂的接口一般会在UT查找中就把task做完了。




下一步：
当前的dry-run和工具需要做几个修改。具体修改要做的在各个部分已经写好。
先修改这个，然后把当前尚未完成开发的几个部分的职责重新刷新一轮，更新相关的文档。（KID和locate的三层layer）

1. KID： KID现在目的很明确，抓high_level_target 下面 python调用链里我们分类的F0-F8的这些kernel的接口调用。找出有哪些low_level_tagret, 及他们在哪一行被调用。产出第一阶段的schema.json 
【需要你改的是：】
schema文件里
      "runtime_event": {
        "wrapper": {
          "api": "recompute_w_u_fwd",
          "file": "/Users/bytedance/Desktop/infra_agent/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py",
          "line": 111
        },
这个字段我们不再需要，我们只需要写这个接口是在哪个文件哪一行被调用的。wrapper已经被移除了。所有和low_level_target有关的信息都收敛到自身及其以下的code里。dry-run需要修改。
执行完后的文件可以参考to_fill_kid.json（你也要同步改这个）

2. locate:
   1. layer-1 （CLI）:只做最基本的处理，把确定后的逻辑做了，比如interface_definition.py 这一层，以及根据类别，bind文件确定的层。原则上不碰header和kernel_implement(哪怕是triton)。你觉得或者layer1要怎么划分职责
   2. layer-2 (agent)： 参考当前填好的的to_fill_locate.json。对算子实现做定位。按照自己的理解尽量复现算子调用链路上的核心逻辑。把我们说的四层算子相关文件填好。
   【需要你改的是：】binding允许多文件，多格式，做成一个目录。约束kernel_header只放和kernel_impl一一对应的文件。这个dry-run可能没有要修改的，主要修改to_fill_locate.json
   3. layer-3 (extract): 职责不变，补全end line, 拷贝文件。bug你已经修复。
   【需要你改的是：】extract需要支持bind层是目录且可以存放多文件，同步更新read_hints.txt。需要根据更新后的to_fill_locate.json重新生成本地文件


按照work_flow先整理下到正式落地需要的TODO，先把每个步骤单独做好，最后再从工程设计的合理行角度合并配置文件，完善步骤和步骤的交互接口，这样让待开发和测试的TODO显得有条理。

## 当前的workflow：（默认每一步之间会插入人工review，忽略人工插入的部分的描述）

### 用户准备：

先明确对应的模型+硬件有哪几种可以尝试的后端路径，逐个尝试，把可用的服务启动命令记录下来。

注意：

这个也可以做成自动化，不过严格意义上属于另外一个产品 deploy-agent的范畴，kernel-agent里就先不讨论，职责划分上。deploy-agent调优模型部署参数（包括后端选择和分布式，调度参数等），kernel-agent基于确定好的服务启动方式进行优化。实际使用时我们会先使用deploy-agent完成启动命令的优化，再由kernel-agent这一层做算子level的优化。这里我们只讨论kernel-agent就假定最开始拿到的启动命令已经是最优的。



**步骤输出： 启动命令+测试命令**



### 明确/clone third-party ：

TODO：

* resolve-third-party-repos skill的开发：
  这一步我一般会单独执行，把一些定位仓库和代码版本的上下文信息和后续工作隔离开，只保留最终的结果文件

  * 前提： 该skill需要运行在正式的开发环境里
  * 输入： N条启动命令 （内含模型信息和对应的后端信息），当前sglang仓库源码所在的路径，测试命令（可以先单独在一个配置文件里），希望clone 的目的路径（所有的third-party都clone到该路径下）
  * 工作流程（对agent的约束）：agent可以选择查阅仓库线索，可以选择实际运行服务和测试，可以在各种仓库里临时打log来完成对third-party仓库的信息明确。但是实际运行命令打点确认不是必须的，当版本线索明确的时候，也可以直接clone对应的仓库。
  * 禁止事项：破坏当前的运行环境，比如拉取三方库，重新编译sgl-kernel等。clone下来的仓库只需要作为参考，不要编译并链接到当前环境中。
  * agent输出：
    * 需要仓库的路径： 如果仓库还不存在，就clone下来。
    * 一个文件，内部包含仓库名和对应的本地仓库路径和版本，以及一个说明选项（或者叫evidence）简要说明为什么选择当前的仓库和版本。

  

* resolve-third-party-repos skill的测试：
  * 找一个已知case运行，看agnet能否拿到符合预期的三方仓库。



**步骤输出：完备的可查阅third-party仓库 + 对应的说明文件**



### KID分解 :

TODO:

* 当前实现(tools/kernel_interface_decomposer)的增强实现

  * 整体入口的更改：不仅仅接受high_level target, 也允许输入low level target。 当输入low_level target时，只做kernel定义和实现的源码整合工作，不做拆解和耗时占比统计等工作。

  * 整体入口的更改：可以接受多个启动命令，解析生成多个路径下对应的分解结果（仅针对high_level target)（每个分解结果一个文件，避免冲突）
  * 增强对当前输出文件的约束：严格要求所有的分解结果（low_level接口）有对应的定义所在的python行号，调用GPU kernel的地方（python),kernel定义的地方（接口，包括python和c++以及sgl-kernel里的pytorch接口注册代码），kernel定义的地方（源码，包含python(如triton)和c++（如cultass)
  * 新增import-decomposition CLI： 对于上面步骤的输出文件，把对应low_level里定位到的下面四种实现都拷贝到这个low_level target的专属文件夹里，同时在文件夹里附一个txt，给出四个文件需要read的行数范围。并且把生产的文件夹路径回填到上面步骤的输出文件里。这四个文件包括：
    * 包含原始low_level 接口定义的文件（py文件），里面会有kernel launch 行为。
    * 包含py接口和c++接口绑定关系的文件(一般是xx_extension_xx.cc)，如果类似triton/cuteDSL等kernel没有，置空。
    * 包含kernel定义的文件（只针对c++ kernel，定义only，一般为头文件.h)，如果类似triton/cuteDSL等kernel没有，置空。->为什么需要，可能包含注释
    * 包含kernel实现的文件（py or cu  or cpp) kernel的实现，包含c++kernel和cuda kernel
  * 输入： 包含启动命令，测试命令，target (high_level or low_level的)相关文件和行号信息
  * 输出：
    * High_level 目标时：多个文件夹（每个文件夹对应一个path，如果只有一个path的启动命令，就一个文件夹），每个文件夹里又包含当前分解并选中的low_level target的子文件夹，子文件夹里包含的内容如上面import-decomposition CLI里描述。以及一个输出文件(schema文件)（增强后的当前KDI的输出文件），里面包含low_level target的耗时情况，kernel类别，kernel源码对应信息（包括回填的路径）
    * low_level 目标时： 一个文件夹，里面内容和上面相同，只是schema文件里不需要写耗时等信息，只需要写源码信息。

* 当前KID的测试：

  * 人工找一个low_level target和high_level target 分别校验，看结果是不是符合预期的。

**步骤输出： 所有的需要优化的low_level接口的必要信息，和源码资料， 可以理解为N个workspace，每个里面完全独立** 



### 问题未分级task_pack创建

* 针对上面的每一个workspace，生成每一个low_level 接口未做问题分类的task_pack。 ---- > 一个workspace对应一个 K个low_level 接口的task_pack
* 基本流程已经打通（当前framework_engineer里的内容）

TODO:

* 对config文件做改造，支持输入多个low_level(只支持low_level,因为分解的工作已经做了)。
* 加一个CLI，把workspace里的schema文件转换到task_pack生成任务的config文件里（包含启动命令，接口文件，行号等）。
* 剩余的空着的配置人工填写(forward bundary等）。

* 加一个CLI，把workspace里每个low_level子文件夹拷贝到task_pack里，并且把scheme文件里该level_low target对应的耗时占比/源码所在文件路径，行号等信息也一并创建到task_pack的一个新的文件里，并且在文件里新增分级字段，初始默认分级为1。
* 修改validate-task-pack CLI，需要支持四种level的检测，每种level的检测对应不同的内容。其中level4比较特殊，需要passUT。其余三个模式是检测对应的文件是否存在。
* 测试： 找一个上一个step生成的workspace，看看是否能正确生成K个low_level target的task_pack。并且通过validate-task-pack的检测。（因为默认现在都是模式1，所以可以通过检测）

**步骤输出： N个workspace，每个workspace里包含K个未被分类的task_pack。task_pack里对每个low_level的信息已经做到完全收集，包含kernel源码路径，kernel源码文件，耗时占比，分级，snapshot** 



###　problem_translate agent开发

核心新增agent。独立工作，拥有独立上下文，能够更好的完成translate的工作。

TODO：

* agent的开发：
  * 输入： 一个独立的task_pack + 外部信息（用户提供的transformers仓库路径）
  * agnet工作流程（和你写的基本一样）：根据task_pack里提供的信息（仓库位置，kernel类别等），先尝试去找到接口对应的原始UT，并且用该UT生成task_pack内LV4的UT-> 若找不到尝试自己实现lv4的接口->若成功，生成对应的文件，保留UT（lv4的证据）-> 若失败，尝试lv3，去仓库找对应  ->找到生成lv3对应的证据文件，找不到lv2...依次类推。
  * 输出： 对每个task最终的level判定，（写到上面的文件里）该level需要的证据文件（其中level1的证据在上一轮已经具备）。
* agent的测试：
  * 人工check几个实现定好level的 low_level是否能判定正确且生成正确的文件（包括UT）



###  validate-task-pack 

基本和你写的一样，新增对level的check，再次确认一遍task_pack的可用性，TODO上面的步骤已经写了。

### final summary

一个skill（包含简单的CLI），阅读每个workspace（对应一个high_level target)。根据level判定结果和性能占比数据计算公式的score。对work_space进行排序。下一阶段的优化就基于这个顺序优先优化排序高的workspace。

TODO：

skill的创建和基于验证过的task_pack的workspace排序测试。








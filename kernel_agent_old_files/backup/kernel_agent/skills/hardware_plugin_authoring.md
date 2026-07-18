# Skill: Hardware Plugin Authoring

此 skill 用于把新硬件、新 DSL 或新工具链接入 Kernel Agent。

## 新插件需要覆盖

- 硬件 spec：带宽、cache、shared/local memory、向量宽度、矩阵单元。
- 编程模型：线程/block/warp 或厂商等价抽象。
- 编译器：命令、flags、debug 方法。
- profiler：采集命令、关键指标、常见瓶颈解释。
- 示例：至少一个 memory-bound、一个 compute-bound、一个 reduction/fusion 示例。

## 编写原则

- 只写可被 Kernel Agent 直接用于决策的内容。
- 每条优化建议最好附一个适用条件。
- 不要把框架接入知识放在 hardware plugin 中。
- 不要改变 `KernelRequestSpec` 或 `KernelDeliveryPackage` 的核心格式。

## 验收

- Kernel Agent 能根据插件选择实现路径。
- Kernel Agent 能解释 profiler 指标并提出下一轮优化。
- 插件包含 fallback 或“不支持”的明确说明。

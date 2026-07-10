# Future Command Surface

这些命令只定义未来 CLI 形态，第一阶段不实现。

## profile-model

```bash
kernel-agent framework profile-model \
  --framework sglang \
  --model /path/to/model \
  --launch-command launch.sh \
  --workload-command workload.sh \
  --out hotspot_report.md
```

作用：采集或解析模型 profile，生成 `hotspot_report.md`。

## make-kernel-request

```bash
kernel-agent framework make-kernel-request \
  --hotspot-report hotspot_report.md \
  --candidate qwen35_linear_attention_prefill \
  --out kernel_request_spec.yaml \
  --ut-out unit_test_harness.py
```

作用：基于热点候选生成 `KernelRequestSpec` 和 UT 模板。

## review-framework-change

```bash
kernel-agent framework review-framework-change \
  --request framework_change_request.yaml \
  --out framework_change_review.md
```

作用：评估 Kernel Agent 提出的框架配套改造。

## verify-e2e

```bash
kernel-agent framework verify-e2e \
  --delivery kernel_delivery_package.md \
  --launch-command launch_candidate.sh \
  --workload-command workload.sh \
  --out e2e_verification_report.md
```

作用：执行端到端性能/精度验收。

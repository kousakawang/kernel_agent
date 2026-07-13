# Kernel Interface Decomposer

Automates the workflow for decomposing a high-level Python model interface into
GPU kernels, Python wrappers, and implementation source locations.

## Quick Start

Create a config:

```yaml
version: 1
workdir: /path/to/model_ana
output_dir: ./kernel_decompose_out/qwen35_gdn

target:
  file: ./sglang/python/sglang/srt/models/qwen3_5.py
  line: 488

commands:
  service: "python -m sglang.launch_server ..."
  ready:
    type: http
    url: "http://127.0.0.1:30000/health"
    timeout_sec: 300
  test: "python test_request.py"
  stop:
    signal: SIGINT
    grace_sec: 30

selection:
  per_invocation: true
  top_k: 20
  min_duration_us: 0
  min_share_in_invocation: 0.0
  stages: ["prefill", "decode", "mixed", "unknown"]

profiling:
  nsys_bin: nsys
  trace_cuda_graph_nodes: true
  include_python_backtrace: true
  skip_target_invocations: 0
  max_runtime_sec: 1800

resolution:
  source_roots:
    - ./sglang/python
    - ./sglang/sgl-kernel
  third_party_prefixes:
    - flashinfer
    - flash_attn
    - deep_gemm
    - aiter
    - kt_kernel
```

Run:

```bash
python3 -m framework_engineer.kernel_interface_decomposer run config.yaml
```

Analyze an existing trace:

```bash
python3 -m framework_engineer.kernel_interface_decomposer analyze config.yaml --nsys-rep out/profile.nsys-rep
```

Main output:

```text
<output_dir>/decomposition.schema.json
<output_dir>/profile.nsys-rep
<output_dir>/profile.sqlite
<output_dir>/events/events_<pid>.jsonl
<output_dir>/service.log
<output_dir>/test.log
```

## Notes

- The runtime injector is installed through a generated `sitecustomize.py` under
  `<output_dir>/_inject`; no SGLang source files are modified by default.
- Stage labels come from `forward_batch.forward_mode` when available and are
  written into `PYGPU:type=target` and `PYGPU:type=wrap` NVTX ranges.
- Runtime JIT source resolution records `load_jit` `cuda_files`, `cpp_files`,
  wrapper exports, and compile flags when the JIT module is loaded.
- PyTorch native ops intentionally stop at public API / ATen op boundaries.


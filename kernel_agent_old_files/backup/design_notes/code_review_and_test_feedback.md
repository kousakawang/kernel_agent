## cmd_probe_target_calls:
1. 没有以非cuda-graph模式运行服务和测试
2. arg_count": 0  log里的arg_count=0是什么意思


## capture-snapshots：
当前执行报错了,应该是函数签名没对上，需要修正一下？
```
F
======================================================================
FAIL: test_real_sglang_snapshot_task_pack_flow (kernel_agent.framework_engineer.tests.test_real_sglang_phase1.RealSGLangPhase1CliTests.test_real_sglang_snapshot_task_pack_flow)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/sgl-workspace/kernel_agent/framework_engineer/tests/test_real_sglang_phase1.py", line 169, in test_real_sglang_snapshot_task_pack_flow
    capture = self._run_cli(
              ^^^^^^^^^^^^^^
  File "/sgl-workspace/kernel_agent/framework_engineer/tests/test_real_sglang_phase1.py", line 271, in _run_cli
    self.assertEqual(proc.returncode, 0, f"args={args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
AssertionError: 1 != 0 : args=('capture-snapshots', '--task-pack', '/tmp/qwen35_gdn_extend_task_pack', '--service-cmd', 'CUDA_VISIBLE_DEVICES=7 SGLANG_VLM_CACHE_SIZE_MB=0 python3 -m sglang.launch_server --model-path /data01/models/Qwen3.5-9B/ --host 127.0.0.1 --port 8080 --mem-fraction-static 0.7 --cuda-graph-max-bs 128 --tensor-parallel-size 1 --mm-attention-backend fa3 --cuda-graph-bs 128 120 112 104 96 88 80 72 64 56 48 40 32 24 16 8 4 2 1  --disable-radix-cache --disable-cuda-graph --disable-cuda-graph', '--workload-cmd', 'python3 -m sglang.bench_serving --backend sglang-oai-chat --dataset-name image --num-prompts 8 --apply-chat-template --random-output-len 8 --random-input-len 16 --image-resolution 480x720 --image-format jpeg --image-count 1 --image-content random --random-range-ratio 1 --host=127.0.0.1 --port=8080', '--target-file', '/sgl-workspace/sglang/python/sglang/srt/layers/attention/fla/chunk_fwd.py', '--function-name', 'chunk_gated_delta_rule_fwd_intra', '--target-name', 'chunk_gated_delta_rule_fwd_intra', '--signature', 'candidate_extend(q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc)', '--mode', 'extend', '--backend', 'triton', '--layer-id', '', '--max-raw-cases', '32', '--mutable-arg-path', 'kwargs.ssm_states', '--drop-first-arg', '--startup-timeout', '240', '--workload-timeout', '1200', '--health-url', 'http://127.0.0.1:8080/health')
stdout:
{"raw_snapshot_count": 0, "target_name": "chunk_gated_delta_rule_fwd_intra", "workload_returncode": 1}

stderr:


----------------------------------------------------------------------
Ran 1 test in 100.203s

FAILED (failures=1)
```
除了这个报错，还有如下问题：
1. 没有以非cuda-graph模式运行

以及改进建议：
1. 整个模型里的接口调用次数太多了，会dump太多的文件，一般来说算子level的性能优化，我们只需要跑一层transformer layer就够了。
如果只是给接口增加装饰器，每一层都会dump，很快就超过了max_raw_cases的限制，导致我们无法收集到足够多有效的case。这里有两个解决思路，要么在运行模型时，让他只跑一层transformer_layer。（需要修改model的python代码）
要么同一个hash的key值，只dump一次（但是记录hit hash key的次数）。




## probe-env

kernel_agent/framework_engineer/templates/probe_cuda_extension.py
这个文件的实现有问题，会有如下的错误：
FAILED: [code=1] main.o 
c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=kernel_agent_probe_cuda_extension -DTORCH_API_INCLUDE_EXTENSION_H -isystem /usr/local/lib/python3.12/dist-packages/torch/include -isystem /usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /root/.cache/torch_extensions/py312_cu130/kernel_agent_probe_cuda_extension/main.cpp -o main.o 
/root/.cache/torch_extensions/py312_cu130/kernel_agent_probe_cuda_extension/main.cpp: In function ‘void pybind11_init_kernel_agent_probe_cuda_extension(pybind11::module_&)’:
/root/.cache/torch_extensions/py312_cu130/kernel_agent_probe_cuda_extension/main.cpp:4:46: error: ‘add_one’ was not declared in this scope
    4 | m.def("add_one", torch::wrap_pybind_function(add_one), "add_one");

我看了下临时生成的cpp文件，里面没有包含CUDA_SRC里定义的代码
```cpp
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("add_one", torch::wrap_pybind_function(add_one), "add_one");
}
```
需要修复


## select-snapshots
这个没测，要先修改capture-snapshots。
根据capture-snapshots修改后的版本我们再看一下是不是需要修改这部分逻辑，如果我们选择记录对应的key值的命中次数，这部分会比较简单

## generate-harness
没测，需要按照你说的重新实现
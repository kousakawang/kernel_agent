from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor add_one(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one, "add_one");
}
"""


CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void add_one_kernel(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] + 1.0f;
}

torch::Tensor add_one(torch::Tensor x) {
  auto out = torch::empty_like(x);
  int n = x.numel();
  add_one_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
"""


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
    mod = load_inline(
        name="kernel_agent_probe_cuda_extension",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        with_cuda=True,
        verbose=False,
    )
    x = torch.randn(1024, device="cuda")
    out = mod.add_one(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + 1.0)
    print("cuda extension usable: load_inline compile+run ok")


if __name__ == "__main__":
    main()

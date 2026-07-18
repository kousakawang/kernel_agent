from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x, y, out, n: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) + tl.load(y + offs, mask=mask), mask=mask)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    _add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, block=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y)
    print(f"triton usable: version={triton.__version__}")


if __name__ == "__main__":
    main()

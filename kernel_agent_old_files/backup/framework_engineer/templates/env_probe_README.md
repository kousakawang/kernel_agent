# Env Probe Templates

Framework Engineer should copy these probes into `task_pack/env_probe/`, run them, and summarize results in `env_manifest.yaml`.

The goal is not to prove every possible optimization path. The goal is to prove which implementation paths Kernel Engineer may use without installing new dependencies.

## Required Probes

- `probe_triton.py`: import Triton and compile/run a tiny kernel.
- `probe_cutedsl.py`: import CuTe DSL/CUTLASS Python package and, when possible, compile/run a minimal kernel.
- `probe_cuda_extension.py`: compile/run a minimal CUDA extension through PyTorch.
- `probe_ncu.sh`: check Nsight Compute and optionally profile one benchmark case.

## Policy

- `available=true` means the package or binary exists.
- `usable_for_task=true` means the probe ran successfully enough for this task.
- CuTe DSL must not be marked unusable just because one existing framework backend has a stricter hardware-specific path. Judge custom-kernel feasibility by the probe result.

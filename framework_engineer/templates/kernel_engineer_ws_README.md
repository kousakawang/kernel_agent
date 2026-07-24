# kernel_engineer workspace

Kernel Engineer may create, modify, and delete implementation sources, build
scripts, compiled artifacts, profiler outputs, and temporary iteration files in
this directory.

Outside this directory, Kernel Engineer may modify only:

- `task/candidate_impl.py`

`kernel_translate/` and `kernel_source_package/` are read-only references.
Snapshots, reference/original implementations, correctness and benchmark
harnesses, scripts, environment probes, task contracts, the outer README, and
the validator are read-only. Files under `task/docs/` are also read-only
Framework Engineer evidence.

Write iteration notes to `task/kernel_engineer_ws/iteration_log.md`.

# kernel_translate workspace

This directory is the only writable workspace for the `kernel_translate` step.

`kernel_translate` may create translated implementations, analysis notes, helper
code, and intermediate artifacts here. It may read the rest of the task pack,
including `kernel_source_package/` and `snapshots/`, but it must not modify
`candidate_impl.py`, the replay harness, task contract, snapshots, or files
outside this directory.

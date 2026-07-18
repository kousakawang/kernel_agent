# Qwen3.5 Linear Attention Operator Breakdown

## Summary

- Task ID:
- Model:
- Source wrapper:
- Source core kernel:
- Selected target:

## Module Path

Describe how execution reaches the selected core kernel.

```text
BailingMoELinearAttention.forward
  -> ...
  -> GDNAttnBackend.forward_extend
  -> GDNKernelDispatcher.extend
  -> TritonGDNKernel.extend
```

## Candidate Sub-Operators

| Sub-Operator | Source | Inputs | Outputs | Phase 1 Decision | Reason |
| --- | --- | --- | --- | --- | --- |
| conv/qkv preparation |  |  |  | defer |  |
| gating |  |  |  | include as input |  |
| core recurrent/chunk kernel |  |  |  | optimize |  |
| state tracking |  |  |  | defer |  |

## Selected ABI

```python
def candidate(*args, **kwargs):
    ...
```

The concrete args/kwargs tree is defined by selected snapshot samples.

## Framework-Owned Inputs

- q/k/v:
- g/beta:
- ssm_states:
- cache_indices:
- query_start_loc:

## Risks

- 

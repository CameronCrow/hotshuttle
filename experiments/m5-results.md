# M5 results -- end-to-end fan-out

4 workers, 1 physical slot, 8 questions each,
driven from `workers.yaml` through the scheduler. Consecutive turns deliberately
target different workers, so the pool evicts and restores on nearly every dispatch.

- dispatches: **36** in 118s
- re-prefilled: **5833 / 51013 tokens = 11.4%** (bar: < 15%)
- of which the four unavoidable cold seeds account for 100% of their own round
- pool: `{'free': [], 'resident': {0: 'manifest'}, 'stats': {'cold_fills': 4, 'evictions': 35, 'restores': 32}}`
- compactions: `{'pool': 0, 'compiler': 0, 'scheduler': 0, 'manifest': 0}`

## Where each worker ended up

**pool** (hotshuttle/core/pool.py), 1749 tokens resident:

```
>user
Name one edge case its code explicitly handles.<|im_end|>
<|im_start|>assistant
<think>

</think>

When all slots are locked mid-generation, it raises a RuntimeError instead of blocking.<|im_end|>
<|im_start|>user
Summarize its role in five words or fewer.<|im_end|>
<|im_start|>assistant
<think>

</think>

Slot allocator and evictor<|im_end|>
```

**compiler** (hotshuttle/core/compiler.py), 1359 tokens resident:

```
its code explicitly handles.<|im_end|>
<|im_start|>assistant
<think>

</think>

Whitespace changes in the seed or prior progress, which invalidate the cache from that byte on.<|im_end|>
<|im_start|>user
Summarize its role in five words or fewer.<|im_end|>
<|im_start|>assistant
<think>

</think>

Prompt byte-compilation and cache-guarding.<|im_end|>
```

**scheduler** (hotshuttle/core/scheduler.py), 1738 tokens resident:

```
|>assistant
<think>

</think>

worker<|im_end|>
<|im_start|>user
Name one edge case its code explicitly handles.<|im_end|>
<|im_start|>assistant
<think>

</think>

One bad task must not stall the run.<|im_end|>
<|im_start|>user
Summarize its role in five words or fewer.<|im_end|>
<|im_start|>assistant
<think>

</think>

GPU task scheduler<|im_end|>
```

**manifest** (hotshuttle/core/manifest.py), 1488 tokens resident:

```
istant
<think>

</think>

It explicitly handles the case where a role's `ctx_budget` exceeds the model's `ctx_per_slot`, raising a `ManifestError` to prevent impossible slot assignments.<|im_end|>
<|im_start|>user
Summarize its role in five words or fewer.<|im_end|>
<|im_start|>assistant
<think>

</think>

Manifest validation and parsing.<|im_end|>
```

## Acceptance

- [x] demo completed unattended
- [x] re-prefilled < 15% of prompt tokens
- [x] every worker still has its context
- [x] pool bookkeeping consistent

```json
{
  "turns_per_worker": 8,
  "workers": 4,
  "dispatched": 36,
  "errors": [],
  "wall_s": 118.24667560000671,
  "seed_reprefill_ratio": 1.0,
  "reprefilled": 5833,
  "total_prompt": 51013,
  "reprefill_ratio": 0.11434340266206654,
  "pool": {
    "free": [],
    "resident": {
      "0": "manifest"
    },
    "stats": {
      "cold_fills": 4,
      "evictions": 35,
      "restores": 32
    }
  },
  "compactions": {
    "pool": 0,
    "compiler": 0,
    "scheduler": 0,
    "manifest": 0
  },
  "n_ctx_used": {
    "pool": 1749,
    "compiler": 1359,
    "scheduler": 1738,
    "manifest": 1488
  }
}
```
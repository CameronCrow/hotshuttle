# M3 results -- compaction as reset

One worker, `ctx_budget=2500`, `compact_at=0.8`, local (Bonsai) summarizer.

| turn | n_ctx_used | evaluated | reused |
|---:|---:|---:|---:|
| 2 | 434 | 371 | 59 |
| 3 | 808 | 371 | 433 |
| 4 | 1182 | 371 | 807 |
| 5 | 1556 | 371 | 1181 |
| 6 | 1930 | 371 | 1555 |
| 7 | 0 | 371 | 1929 |

Peak context before reset: **1930** tokens.
First turn on the new seed prefilled **50** tokens (cache_n=0); the turn after evaluated **20** and reused 58.

## The summary that replaced the context

```
<|im_start|>system
You are a maintenance log worker. Answer briefly.

Prior progress:
Part # XR-3390 confirmed.<|im_end|>

```

Probe reply: `Part # XR-3390`

## Acceptance

- [x] compaction fired at the threshold
- [x] transcript cleared, worker COLD
- [x] planted fact survived the reset
- [x] new seed much smaller than old context
- [x] fresh seed prefilled once (no cache)
- [x] turns are suffix-only again
- [x] superseded blob deleted

```json
{
  "budget": 2500,
  "turns": [
    {
      "turn": 2,
      "evaluated": 371,
      "reused": 59,
      "n_ctx_used": 434,
      "compactions": 0
    },
    {
      "turn": 3,
      "evaluated": 371,
      "reused": 433,
      "n_ctx_used": 808,
      "compactions": 0
    },
    {
      "turn": 4,
      "evaluated": 371,
      "reused": 807,
      "n_ctx_used": 1182,
      "compactions": 0
    },
    {
      "turn": 5,
      "evaluated": 371,
      "reused": 1181,
      "n_ctx_used": 1556,
      "compactions": 0
    },
    {
      "turn": 6,
      "evaluated": 371,
      "reused": 1555,
      "n_ctx_used": 1930,
      "compactions": 0
    },
    {
      "turn": 7,
      "evaluated": 371,
      "reused": 1929,
      "n_ctx_used": 0,
      "compactions": 1
    }
  ],
  "blob_before": "C:\\Users\\Cameron\\Projects\\bonsai\\bench-logs\\slots\\engineer.bin",
  "blob_existed": true,
  "compacted": true,
  "transcript_cleared": true,
  "state_after": "COLD",
  "blob_deleted": true,
  "fresh_prefill_tokens": 50,
  "fresh_cache_n": 0,
  "probe_reply": "Part # XR-3390",
  "fact_survived": true,
  "fact_in_summary": true,
  "next_evaluated": 20,
  "next_reused": 58,
  "peak_n_ctx_used": 1930,
  "verdicts": {
    "compaction fired at the threshold": true,
    "transcript cleared, worker COLD": true,
    "planted fact survived the reset": true,
    "new seed much smaller than old context": true,
    "fresh seed prefilled once (no cache)": true,
    "turns are suffix-only again": true,
    "superseded blob deleted": true
  }
}
```
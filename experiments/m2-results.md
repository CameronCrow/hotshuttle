# M2 results -- two workers, one slot

Every turn below forced a save, an eviction and a restore: with `n_slots=1`,
dispatching either worker displaces the other.

| turn | worker | evaluated | reused | own code | leaked other | reply |
|---:|---|---:|---:|:-:|:-:|---|
| 1 | alpha | **22** | 1426 | yes | no | `ORCHID-4417` |
| 1 | beta | **22** | 1425 | yes | no | `GRANITE-9082` |
| 2 | alpha | **22** | 1456 | yes | no | `ORCHID-4417` |
| 2 | beta | **22** | 1455 | yes | no | `GRANITE-9082` |
| 3 | alpha | **22** | 1486 | yes | no | `ORCHID-4417` |
| 3 | beta | **22** | 1485 | yes | no | `GRANITE-9082` |

- paging: **1872 ms** over 13 save/restore ops
- generating: 160718 ms
- overhead: **1.2%** of generation time
- re-prefilled across the run: **25.4%** of prompt tokens
- pool: `{'free': [], 'resident': {0: 'beta'}, 'stats': {'cold_fills': 2, 'evictions': 7, 'restores': 6}}`

## Acceptance

- [x] no re-prefill on warm turns
- [x] every worker recalled its own code
- [x] no cross-worker contamination
- [x] paging overhead < 10% of generation
- [x] saves matched evictions

```json
{
  "rows": [
    {
      "turn": 1,
      "worker": "alpha",
      "evaluated": 22,
      "reused": 1426,
      "prompt_tokens": 1448,
      "recalled_own": true,
      "leaked_other": false,
      "reply": "ORCHID-4417"
    },
    {
      "turn": 1,
      "worker": "beta",
      "evaluated": 22,
      "reused": 1425,
      "prompt_tokens": 1447,
      "recalled_own": true,
      "leaked_other": false,
      "reply": "GRANITE-9082"
    },
    {
      "turn": 2,
      "worker": "alpha",
      "evaluated": 22,
      "reused": 1456,
      "prompt_tokens": 1478,
      "recalled_own": true,
      "leaked_other": false,
      "reply": "ORCHID-4417"
    },
    {
      "turn": 2,
      "worker": "beta",
      "evaluated": 22,
      "reused": 1455,
      "prompt_tokens": 1477,
      "recalled_own": true,
      "leaked_other": false,
      "reply": "GRANITE-9082"
    },
    {
      "turn": 3,
      "worker": "alpha",
      "evaluated": 22,
      "reused": 1486,
      "prompt_tokens": 1508,
      "recalled_own": true,
      "leaked_other": false,
      "reply": "ORCHID-4417"
    },
    {
      "turn": 3,
      "worker": "beta",
      "evaluated": 22,
      "reused": 1485,
      "prompt_tokens": 1507,
      "recalled_own": true,
      "leaked_other": false,
      "reply": "GRANITE-9082"
    }
  ],
  "pool": {
    "free": [],
    "resident": {
      "0": "beta"
    },
    "stats": {
      "cold_fills": 2,
      "evictions": 7,
      "restores": 6
    }
  },
  "paging_ms": 1871.5661000460386,
  "generate_ms": 160717.56390010705,
  "page_ops": 13,
  "overhead_ratio": 0.01164506264672664,
  "reprefill_ratio": 0.25422715627668657
}
```
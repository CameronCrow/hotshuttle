---
name: bonsai
description: Spin up and drive the local Bonsai 27B ternary model — a private, free, OpenAI-compatible LLM on this machine for bulk drafting, summarization, and scoped generation. Use whenever a task should run inference locally on Bonsai instead of the metered API, or asks to "start/launch bonsai", "use the local model", "draft/summarize with bonsai", "run this on bonsai", or route token-heavy local work off the API. Covers starting/stopping the server, calling it correctly (thinking disabled), running several persistent workers off one GPU slot, and its hard limits (~12K context per worker, GPU-only). Do NOT use Bonsai as an autonomous multi-step agent brain — it is a scoped worker.
---

# bonsai — local Bonsai 27B toolkit

Bonsai 27B (ternary, 2-bit; a build of Qwen3.6-27B) runs locally via PrismML's llama.cpp fork on
an OpenAI-compatible endpoint. Use it for private / free / offline drafting and summarization, and
as a **scoped worker** under a deterministic pipeline or a stronger orchestrator — **not** as an
autonomous agent (see Limits).

Scripts live at the plugin root and find the model via `BONSAI_DIR`
(default `~/Projects/bonsai`); set it if the weights live elsewhere.

## Start / stop

```bash
ROOT="${CLAUDE_PLUGIN_ROOT:-$HOME/Projects/hotshuttle}"
bash "$ROOT/bonsai.sh" start     # idempotent; detached, BLOCKS until ready, prints endpoint
bash "$ROOT/bonsai.sh" status    # "up: …/v1" or "down" (exit code reflects it)
bash "$ROOT/bonsai.sh" stop
bash "$ROOT/bonsai.sh" restart   # hard reset — reloads the model
```

Endpoint `http://127.0.0.1:8080/v1`, model name `bonsai-27b`. Defaults: 12K context, one slot,
q8_0 KV, slot save/restore enabled. Override with `BONSAI_CTX`, `BONSAI_SLOTS`,
`BONSAI_KV_QUANT`, `BONSAI_SLOT_DIR`.

## Call it — MCP tools (preferred, if this plugin's server is running)

- `bonsai_status` / `bonsai_start` — lifecycle
- `bonsai_chat(prompt, system, max_tokens)` — one-shot, thinking already disabled
- `worker_spawn` / `worker_ask` / `worker_list` / `worker_retire` — **persistent** workers
  that keep their own context across calls; see below

## Call it — Python

```python
import os, sys
root = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.expanduser("~/Projects/hotshuttle")
sys.path.insert(0, root)
from bonsai_client import Bonsai
b = Bonsai(); b.ensure_up()                       # starts the server if needed
print(b.complete("Summarize this file: ..."))
```

`Bonsai` disables thinking and strips any residual `<think>` block. Pass `think=True` to opt back in.

## Call it — raw HTTP (YOU MUST disable thinking)

POST `/v1/chat/completions` with `"chat_template_kwargs": {"enable_thinking": false}`. Otherwise the
model spends its whole token budget on `<think>` and returns **empty** content.

## Several workers at once (this is what hotshuttle adds)

A plain call is stateless: each request carries its full context and the server remembers nothing.
That is still true of `bonsai_chat` and `bonsai_client.py`.

The `worker_*` tools are different. Each worker keeps its own conversation across calls, and the
pool pages their state between VRAM and disk around every switch — so you can run **more workers
than the card has slots** (one, here) and each still only pays for its *new* turn. Measured over a
4-worker demo: 11.4 % of prompt tokens re-evaluated, the rest served from cache.

Rules that matter when using them:

- **Send each piece of context once.** The worker keeps it. Re-sending wastes its budget and breaks
  the append-only prompt that makes the caching work.
- **Keep workers narrow.** One job each. Scoped work is what a 2-bit model is good at.
- **Compaction is lossy.** A worker that outgrows `ctx_budget` is summarized and reset — and the
  summary is written by Bonsai itself, which is the weakest link in the system. `worker_ask` tells
  you when it happens; re-send anything that must survive.

## Limits (don't fight these; measured in `bench-results.md`)

- **12K context per worker, ~30 tok/s** on the 8 GB card. 16K also runs at full speed but only if
  the Windows desktop is holding under ~680 MiB of VRAM — over that it thrashes to ~6 tok/s
  **silently**, with the server still answering normally. If Bonsai suddenly feels slow, check
  `nvidia-smi` before anything else.
- **GPU only.** CPU offload is ~0.3 tok/s (no optimized ternary CPU kernel) — unusable.
- **2-bit quantized:** serviceable drafts, not final prose. Verify all claims and citations.
  It is ~7.4 % worse than FP16 at tool-calling, so don't build a tool-heavy loop on it.
- KV cache is **q8_0** by default. q4_0 fits more context and looks free on perplexity, but
  tool-call success is what a worker fleet actually lives on — spend the headroom on precision.
- Not an autonomous agent. Autonomous loops need ≥64K context; Bonsai starves.

# Bonsai 27B ternary — benchmark results

**Model:** qwen35 27B Q2_0 (6.66 GiB weights, 26.9B params, 64 layers, 4 KV heads, K/V dim 256).
**HW:** RTX 4060 Ti 8 GB VRAM / 16 GB RAM, flash-attn on.
**Date:** 2026-07-20.

Method: `llama-bench` for tok/s (pp = prompt-ingest, tg = generation); `llama-perplexity` for quality.
KV/token = 64·4·(256+256)·bytes → **f16 256 KB, q8_0 128 KB, q4_0 64 KB** per token.

---

## tok/s — GPU (RTX 4060 Ti, -ngl 99, full offload)

Weights alone take 6.66 GiB of the 8 GiB card, leaving only ~0.6–0.9 GiB for KV. So a cell runs **only
if its KV cache fits that sliver** — otherwise Windows silently spills to shared RAM and collapses to
~20 t/s. Cells marked "won't fit" were gated out by the KV-size calc (confirmed: f16/4K spills).

| KV quant | ctx | KV size | pp t/s | tg t/s | fits? |
| -------- | ---:| -------:| ------:| ------:| ----- |
| f16  |  4K | 1024 MB |   —   |   —   | ✗ won't fit (spills to RAM, ~20 t/s) |
| f16  |  8K | 2048 MB |   —   |   —   | ✗ won't fit |
| f16  | 16K | 4096 MB |   —   |   —   | ✗ won't fit |
| f16  | 32K | 8192 MB |   —   |   —   | ✗ won't fit |
| q8_0 |  4K |  512 MB | ~650  | ~18–28 | ✓ |
| q8_0 |  8K | 1024 MB |   —   |   —   | ✗ won't fit (just over) |
| q8_0 | 16K | 2048 MB |   —   |   —   | ✗ won't fit |
| q8_0 | 32K | 4096 MB |   —   |   —   | ✗ won't fit |
| q4_0 |  4K |  256 MB | ~680  | ~18–25 | ✓ |
| q4_0 |  8K |  512 MB | ~790  | ~28   | ✓ |
| q4_0 | 16K | 1024 MB |   —   |   —   | ✗ borderline (~1 GB KV; may fit with a minimized desktop) |
| q4_0 | 32K | 2048 MB |   —   |   —   | ✗ won't fit |

tg is single-rep and varies ±several t/s run-to-run; call GPU generation **~20–28 t/s**, ingest **~650–790 t/s**.

**GPU takeaway:** speed is fine; the card is *context*-limited, not speed-limited. The weights nearly
fill 8 GB, so max usable context is **~8K with q4_0 KV** (or ~4K with q8_0). 16K+ needs a bigger card.

---

## tok/s — CPU (-ngl 0, 6 threads, 16 GB RAM)

| KV quant | ctx | pp t/s | tg t/s | RAM fit (weights+KV) |
| -------- | ---:| ------:| ------:| -------------------- |
| any (f16/q8_0/q4_0) | 4K  | ~0.46 | ~0.32 | fits (~7 GB) |
| any | 8K  | ~0.46 | ~0.32 | fits (~7–8 GB) |
| any | 16K | ~0.46 | ~0.32 | fits (~8–11 GB) |
| any | 32K | ~0.46 | ~0.32 | f16 risky (~14.7 GB, may swap); q4_0 ok (~8.7 GB) |

CPU is **weight-bound at ~0.46 t/s ingest / ~0.32 t/s gen** — measured on f16 KV, and since per-token
attention is <0.3 % of compute (4 KV heads), it's the same across every KV-quant and context. The Q2_0
ternary format has **no optimized CPU kernel** in this fork (all the kernel work went into CUDA), so CPU
is ~15× below even memory-bandwidth expectations.

**CPU takeaway: unusable.** ~2 seconds *per token* — a single agent turn would take many minutes. GPU-only in practice.

---

## quality — perplexity (Alice probe, ~11K tokens, ctx 2048, GPU)

| KV quant | perplexity | vs f16 |
| -------- | ---------- | ------ |
| f16  | 8.6911 | baseline |
| q8_0 | 8.6947 | +0.04 % |
| q4_0 | 8.6938 | +0.03 % |

**KV quantization is quality-free** — q4_0 KV is within 0.03 % of f16. (Perplexity absolute value is
on literary text with 2-bit weights; only the *relative* KV-quant deltas are the signal here.)

---

## Bottom line / operating point

- **Always use q4_0 KV cache** (`--cache-type-k q4_0 --cache-type-v q4_0 -fa`): costs ~nothing in
  quality and fits the most context.
- **On this GPU: q4_0 KV, `-c 8192`, `-ngl 99`** → ~28 t/s gen, ~790 t/s ingest. That's the sweet spot.
  (The default f16 KV would spill to shared RAM and crawl — `serve.bat` updated to q4_0.)
- **16K+ context or CPU offload are off the table** on this box — need a ≥12 GB GPU for real long context.

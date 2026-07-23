# Bonsai 27B ternary — benchmark results

**Model:** qwen35 27B Q2_0 (6.66 GiB weights, 26.9B params, 64 layers, 4 KV heads, K/V dim 256).
**HW:** RTX 4060 Ti 8 GB VRAM / 16 GB RAM, flash-attn on.
**Date:** 2026-07-20.

Method: `llama-bench` for tok/s (pp = prompt-ingest, tg = generation); `llama-perplexity` for quality.
KV/token = 64·4·(256+256)·bytes → **f16 256 KB, q8_0 128 KB, q4_0 64 KB** per token.

> ### ⚠ Correction — 2026-07-22 (hotshuttle M0)
>
> **The KV/token formula above is wrong by ~3.4×, and every "won't fit" verdict derived
> from it below is wrong.** Measured by `experiments/m0_verify.py`; see
> [`docs/PLAN.md`](docs/PLAN.md) §3.
>
> Bonsai is not a dense transformer. It is a ternary build of Qwen3.6-27B, a *hybrid*:
> `full_attention_interval: 4` means only **16 of the 64 layers** carry a growing KV
> cache. The other **48 are Gated DeltaNet** — a fixed-size recurrent state that does not
> grow with context at all. Applying a 64-layer formula counts 4× too many layers.
>
> | quantity | this doc computed | measured 2026-07-22 |
> |---|---:|---:|
> | KV/token @ q8_0 | 128 KiB | **34.1 KiB** |
> | KV/token @ q4_0 | 64 KiB | **18.1 KiB** |
> | fixed recurrent state per slot | not modeled | **149.6 MiB** |
>
> The recurrent state is llama.cpp's own figure (`created context checkpoint … size =
> 149.626 MiB`, independent of token count), corroborated by a saved slot blob for a
> 5-token prompt weighing 149.8 MiB. It is f32 and **not quantizable** — decoding is
> recurrent, so quantization error accumulates down the sequence rather than staying
> local to its token.
>
> KV/token is measured by differencing **saved slot blobs** (M1): a 2611-token worker
> saves at 236.5 MiB under q8_0 and 195.7 MiB under q4_0, so subtracting the 149.6 MiB
> floor gives 34.1 and 18.1 KiB/token. This supersedes an earlier estimate here of
> ~37–40 / ~22–23 KiB/token taken from the *slope of `nvidia-smi` against `-c`* — that
> instrument is confounded by desktop VRAM drift and by allocator rounding; the blob is a
> direct measurement of the same bytes. Both figures land on the prediction derived from
> the Qwen3.6-27B config (34 and 18 KiB/token).
>
> Consequence: **the fixed 149.6 MiB floor, not the KV cache, is the dominant per-slot
> cost at these context lengths**, and the real ceiling is far higher than 8K.
>
> **Measured fit/throughput** (1501-token prompt, 48 predicted, one slot, `-ngl 99`):
>
> | total ctx | KV quant | server VRAM | desktop at the time | ingest t/s | decode t/s | |
> |---:|---|---:|---:|---:|---:|---|
> | 8192  | q4_0 | 7076 MiB | 769 MiB | 634 | 23.2 | ✓ |
> | 8192  | q8_0 | 6654 MiB | 777 MiB | 557 | 28.9 | ✓ (this doc said "won't fit") |
> | 10240 | q8_0 | 7267 MiB | 343 MiB | 716 | 30.3 | ✓ |
> | 12288 | q4_0 | 7149 MiB | 343 MiB | 634 | 28.9 | ✓ |
> | 12288 | q8_0 | 7345 MiB | 356 MiB | 684 | 30.2 | ✓ **new default** |
> | 14336 | q8_0 | 7419 MiB | 356 MiB | 713 | 29.9 | ✓ |
> | 16384 | q4_0 | 7241 MiB | 356 MiB | 713 | 29.9 | ✓ (this doc said "won't fit") |
> | 16384 | q8_0 | 7507 MiB | 346 MiB | 676 | 29.8 | ✓ (this doc said "won't fit") |
> | 16384 | q8_0 | 6976 MiB | 845 MiB |  32 |  6.4 | ✗ **thrashed** |
> | 16384 | q4_0 | 7150 MiB | 714 MiB |  29 |  6.9 | ✗ **thrashed** |
>
> The last two rows are the same configs as rows 7–8 — the *only* difference is how much
> VRAM the Windows desktop happened to be holding. **The binding constraint on this box is
> total headroom versus desktop usage, not KV quantization.** 16K @ q8_0 needs the desktop
> under ~680 MiB; a browser can take that. 12K @ q8_0 leaves ~845 MiB of desktop budget at
> full speed, which is why it is the standing default.
>
> Note the failure mode: WDDM lets the process oversubscribe and silently pages the
> overflow to host memory. An over-budget config **still starts, still answers, and still
> reports ~the same `memory.used`** — it just runs ~4× slower. `nvidia-smi` cannot
> distinguish fit from spill here; decode rate can, which is why the table reports it.

---

## tok/s — GPU (RTX 4060 Ti, -ngl 99, full offload)

Weights alone take 6.66 GiB of the 8 GiB card, leaving only ~0.6–0.9 GiB for KV. So a cell runs **only
if its KV cache fits that sliver** — otherwise Windows silently spills to shared RAM and collapses to
~20 t/s. Cells marked "won't fit" were gated out by the KV-size calc (confirmed: f16/4K spills).

> **Superseded — see the correction at the top of this file.** The "KV size" and "fits?"
> columns below come from the 64-layer formula and are ~3.4× too pessimistic; every q8_0
> row from 8K down, and the q4_0 16K row, were gated out on paper but measured healthy.
> The measured tok/s figures (the ✓ rows) stand.

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

> **Revised 2026-07-22 (M0)** — the first and third bullets above are wrong:
>
> - **Use q8_0 KV, not q4_0.** q8_0 at 12K measures 30.2 t/s and fits with room to spare.
>   q4_0 buys ~16 KiB/token, which no longer matters now that the real KV cost is known,
>   and it is the riskier default: perplexity says q4_0 is free, but perplexity is not the
>   metric a worker fleet lives on — tool-call success is, and 2-bit weights are already
>   −7.4 % there. Spend the headroom on precision. (Drop **K** to q4_0 and keep **V** at
>   q8_0 only if a second slot must be squeezed in.)
> - **`-c 12288` @ q8_0 is the standing default** — full speed with ~845 MiB of desktop
>   headroom. 16K @ q8_0 also runs at full speed but only survives a desktop under
>   ~680 MiB, so it is opt-in (`BONSAI_CTX=16384`) rather than default.
> - **16K is not off the table** — it works today on this 8 GB card. CPU offload remains
>   off the table (~0.3 t/s, no ternary CPU kernel).

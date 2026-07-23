# hotshuttle

*(formerly `bonsai-backend`)*

Local orchestration tooling: one loaded copy of a local model on a single 8 GB GPU serving
**many** small worker agents, by paging each worker's context state between VRAM (hot) and
system RAM (warm) through llama-server slots — driven by a capable orchestrator model.
**The orchestration layer works** — see [`docs/PLAN.md`](docs/PLAN.md) (implementation plan,
with every measurement) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (component specs).
Milestones M0–M5 are done and each has a runnable acceptance script under
[`experiments/`](experiments/).

## Orchestration quick-start

```bash
bash bonsai.sh start                       # 12K ctx @ q8_0 KV, slot save/restore enabled
python experiments/m5_fanout.py            # 4 workers, 1 GPU slot, end-to-end demo
python -m pytest tests/ -q                 # 70 tests, no GPU needed
```

```python
import asyncio
from hotshuttle.core import manifest
from hotshuttle.core.client import Llama
from hotshuttle.core.orchestrator import Orchestrator
from hotshuttle.core.pool import SlotPool
from hotshuttle.core.scheduler import Scheduler
from hotshuttle.core.worker import Task
from hotshuttle.profiles.bonsai import BonsaiProfile

m = manifest.load("workers.yaml")
profile = BonsaiProfile(**m.model_kwargs())
llama = Llama(profile.server_url)
orc = Orchestrator(llama, SlotPool(llama, profile.n_slots,
                                   slot_dir=profile.slot_save_path), profile)

for wid in ("alpha", "beta", "gamma"):     # more workers than slots -- that's the point
    orc.spawn(wid, m.roles["code-reader"])

s = Scheduler(orc, handle=my_processing)   # your work runs off-GPU, overlapping generation
s.submit(Task("alpha", "What does this do?", attach=(source_code,)),
         Task("beta", "Find the error paths.", attach=(other_file,)))
asyncio.run(s.run())
print(f"re-prefilled {orc.reprefill_ratio:.1%} of prompt tokens")
```

Each worker keeps its own context across turns even though only one fits in VRAM at a time:
the pool saves the occupant's state before reusing its slot and restores it on the way back
in, so a returning worker evaluates only its new turn. Measured across the M5 demo — 36
dispatches, 4 workers, 1 slot — **11.4 %** of prompt tokens were re-evaluated; the rest came
from cache. Workers whose context outgrows `ctx_budget` are compacted: retired and re-seeded
with a summary, because an append-only prompt cannot be shrunk in place.

**Pass your own `summarize=` to `Orchestrator`** if you are driving this from a capable model.
The built-in fallback asks Bonsai to summarize itself, and M3 caught it emitting
"Acknowledged." as an entire worker's memory while every mechanical check passed.

Stdlib only, except PyYAML for reading `workers.yaml` (build `Role` objects directly to skip it).

What exists today is the **serving layer** that the orchestration builds on: tooling to run
**Bonsai 27B** locally and serve it to agent harnesses — PrismML's ternary
(1.58-bit, `Q2_0_g128`) build of Qwen3.6-27B, run via PrismML's llama.cpp fork. Vision-capable,
262K-token model, tool-calling, OpenAI-compatible API. This repo is the scripts/config only; the
~10 GB weights and the vendored fork are fetched separately (see Setup).

## Setup

```bash
# 1. Clone the PrismML llama.cpp fork (has the Q2_0_g128 ternary kernels; mainline can't load them)
git clone https://github.com/PrismML-Eng/llama.cpp

# 2. Build it  (Windows: needs CUDA 12.8 + VS2022 Build Tools — see "Build" below)
build.bat            # GPU/CUDA   (build-cpu.bat for a CPU-only fallback)

# 3. Download the GGUF weights into models/  (public HF repo, no token needed)
pip install -U "huggingface_hub[cli]"
hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf --local-dir models

# 4. Serve
bash bonsai.sh start   # or serve.bat — OpenAI-compatible API on :8080
```

> Paths in some scripts currently assume the repo lives at `~/Projects/bonsai`; set `BONSAI_DIR`
> to override, or see the open issues for the portability pass.

## Layout
```
bonsai/
├─ llama.cpp/        PrismML fork (github.com/PrismML-Eng/llama.cpp), built with CUDA
├─ models/           GGUF weights
│   ├─ Ternary-Bonsai-27B-Q2_0.gguf         main weights (~7.2 GB)
│   ├─ Ternary-Bonsai-27B-mmproj-Q8_0.gguf  vision projector (optional)
│   └─ Ternary-Bonsai-27B-dspark-Q4_1.gguf  speculative-decoding drafter (optional)
├─ build.bat         one-shot build (vcvars + Ninja + CUDA arch 89)
├─ serve.bat         OpenAI-compatible server on :8080  ← use this for agent harnesses
└─ run.bat           quick CLI sanity check
```

## Build
```
build.bat         <- GPU (CUDA)  -- default, working
build-cpu.bat     <- CPU-only fallback
```
`build.bat` uses **VS2022 Build Tools (MSVC 14.44)**, not VS2026's 14.51: CUDA 12.8's `cudafe++`
crashes on 14.51 headers (only VS2017-2022 are supported). Build Tools are installed at
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`. Output: `llama.cpp\build\bin\`
(contains `ggml-cuda.dll`).

## Run as a server (for Hermes / agent harnesses)
```
serve.bat
```
Exposes an **OpenAI-compatible** API:

| Setting        | Value                          |
|----------------|--------------------------------|
| Base URL       | `http://127.0.0.1:8080/v1`     |
| Model name     | `bonsai-27b`                   |
| API key        | any non-empty string           |
| Tool calling   | enabled (`--jinja`)            |

Point any OpenAI-style client at it:
```
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=local
```
Same endpoint works for the Anthropic-less OpenAI SDK, LiteLLM, and most harnesses that accept a
custom base URL. Function/tool calls come back in OpenAI `tool_calls` format.

## VRAM note (8 GB card)
`-ngl 99` fully offloads the ~7.2 GB Q2_0 weights to the 4060 Ti — fits, but leaves little for KV cache.
If loading OOMs, lower `-ngl` (spills layers to CPU) or reduce `-c` (context) in `serve.bat`. Vision
(mmproj) + drafter are downloaded but left out of the default server — add them once the base is stable.

Sampling defaults follow PrismML's recommendation: temp 0.7 / top-p 0.95 / top-k 20.

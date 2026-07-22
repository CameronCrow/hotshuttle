@echo off
REM Bonsai 27B ternary -> OpenAI-compatible server for agent harnesses (Hermes, etc).
REM Endpoint:  http://127.0.0.1:8080/v1   model name: "bonsai-27b"   api key: any non-empty string
REM
REM GPU (CUDA) build, benchmark-tuned operating point (see bench-results.md):
REM weights fill ~6.7GB of the 8GB card, so KV MUST be q4_0 + flash-attn to fit -> ~28 tg t/s at 8K.
REM f16 KV would spill to shared RAM and crawl. 8K is the practical ceiling here; 16K+ needs a bigger GPU.
REM --parallel 1: default is 4 slots, each reserving a full -c KV -> ~4x KV, fills the 8GB card to 96%%
REM   and the WDDM driver thrashes (~3 t/s). One slot keeps it in VRAM at full ~28 t/s.
REM ponytail: q4_0 KV is quality-free (<0.05% ppl). To push ctx, first free VRAM (minimized desktop), then try -c 12288.
set LLAMA=%~dp0llama.cpp\build\bin\llama-server.exe
set MODELS=%~dp0models
"%LLAMA%" ^
  -m "%MODELS%\Ternary-Bonsai-27B-Q2_0.gguf" ^
  --alias bonsai-27b ^
  --host 127.0.0.1 --port 8080 ^
  -ngl 99 ^
  -c 8192 ^
  --parallel 1 ^
  -fa 1 --cache-type-k q4_0 --cache-type-v q4_0 ^
  --jinja ^
  --temp 0.7 --top-p 0.95 --top-k 20

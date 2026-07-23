@echo off
REM Bonsai 27B ternary -> OpenAI-compatible server for agent harnesses (Hermes, etc).
REM Endpoint:  http://127.0.0.1:8080/v1   model name: "bonsai-27b"   api key: any non-empty string
REM
REM GPU (CUDA) build, benchmark-tuned operating point (see bench-results.md):
REM weights fill ~6.7GB of the 8GB card, so KV MUST be q4_0 + flash-attn to fit -> ~28 tg t/s at 8K.
REM f16 KV would spill to shared RAM and crawl. 8K is the practical ceiling here; 16K+ needs a bigger GPU.
REM --parallel 1: default is 4 slots, each reserving a full -c KV -> ~4x KV, fills the 8GB card to 96%%
REM   and the WDDM driver thrashes (~3 t/s). One slot keeps it in VRAM at full ~28 t/s.
REM Defaults measured 2026-07-22 -- see the "Correction" section of bench-results.md:
REM   -c 12288 @ q8_0 runs at full speed (30 t/s) and leaves ~845 MiB for the Windows desktop.
REM   16384 also runs full speed but needs the desktop under ~680 MiB; over that it THRASHES
REM   SILENTLY (server still starts and answers, ~4x slower). So 16K is opt-in, not default.
REM   q8_0 over q4_0: only 16 of Bonsai's 64 layers grow a KV cache (the rest are DeltaNet with
REM   a fixed 149.6 MiB recurrent state), so KV is ~37-40 KiB/token and the headroom exists.
REM   q4_0 looks free on perplexity, but tool-call rate is the metric a worker fleet lives on.
REM --slots + --slot-save-path: needed by the orchestration layer (GET /slots, and
REM   POST /slots/{id}?action=save|restore, which is a no-op without a save path).
REM Env overrides: BONSAI_CTX (total ctx, split across slots), BONSAI_SLOTS, BONSAI_KV_QUANT, BONSAI_SLOT_DIR.
REM 32K "long input" mode = BONSAI_CTX=32768 with BONSAI_KV_QUANT=q4_0 (32K does not fit at q8_0 here).
if not defined BONSAI_CTX set BONSAI_CTX=12288
if not defined BONSAI_SLOTS set BONSAI_SLOTS=1
if not defined BONSAI_KV_QUANT set BONSAI_KV_QUANT=q8_0
if not defined BONSAI_SLOT_DIR set BONSAI_SLOT_DIR=%~dp0bench-logs\slots
set LLAMA=%~dp0llama.cpp\build\bin\llama-server.exe
set MODELS=%~dp0models
if not exist "%BONSAI_SLOT_DIR%" mkdir "%BONSAI_SLOT_DIR%"
"%LLAMA%" ^
  -m "%MODELS%\Ternary-Bonsai-27B-Q2_0.gguf" ^
  --alias bonsai-27b ^
  --host 127.0.0.1 --port 8080 ^
  -ngl 99 ^
  -c %BONSAI_CTX% ^
  --parallel %BONSAI_SLOTS% ^
  -fa 1 --cache-type-k %BONSAI_KV_QUANT% --cache-type-v %BONSAI_KV_QUANT% ^
  --slots --slot-save-path "%BONSAI_SLOT_DIR%" ^
  --jinja ^
  --temp 0.7 --top-p 0.95 --top-k 20

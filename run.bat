@echo off
REM Quick one-shot sanity check of Bonsai 27B ternary (no server).
set LLAMA=%~dp0llama.cpp\build\bin\llama-cli.exe
set MODELS=%~dp0models
"%LLAMA%" ^
  -m "%MODELS%\Ternary-Bonsai-27B-Q2_0.gguf" ^
  -ngl 99 -c 8192 ^
  --temp 0.7 --top-p 0.95 --top-k 20 ^
  -p "Explain quantum computing in simple terms." -n 256

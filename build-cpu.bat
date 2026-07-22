@echo off
REM CPU-only build of the PrismML llama.cpp fork (works today; no CUDA).
REM GPU build (build.bat) is blocked: CUDA 12.8's cudafe++ crashes on VS2026/MSVC-14.51 headers.
REM To enable GPU later: install VS2022 v143 build tools (CUDA 12.8-supported) OR a CUDA that supports VS2026.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
cd /d "C:\Users\Cameron\Projects\bonsai\llama.cpp" || exit /b 1
cmake -G Ninja -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build-cpu -j || exit /b 1
echo BUILD_OK

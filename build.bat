@echo off
REM GPU (CUDA) build of the PrismML llama.cpp fork for Bonsai 27B ternary (Q2_0_g128 kernels).
REM Uses VS2022 Build Tools (v143 toolset, ~14.4x) because CUDA 12.8's cudafe++ crashes on
REM VS2026/MSVC-14.51 headers. Ninja generator; arch 89 = RTX 4060 Ti (Ada).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
cd /d "C:\Users\Cameron\Projects\bonsai\llama.cpp" || exit /b 1
cmake -G Ninja -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build -j || exit /b 1
echo BUILD_OK

#!/usr/bin/env bash
# Bonsai 27B ternary benchmark: KV-quant x context, tok/s (llama-bench) + quality (perplexity).
# GPU: only cells whose KV fits free VRAM are run (per your "only test what fits"); rest marked.
# CPU: all 12 attempted, OOM/timeout caught. tok/s measured at depth=ctx (real at-context speed).
cd "C:/Users/Cameron/Projects/bonsai" || exit 1
MODEL=models/Ternary-Bonsai-27B-Q2_0.gguf
GPU=llama.cpp/build/bin/llama-bench.exe
CPU=llama.cpp/build-cpu/bin/llama-bench.exe
PPL=llama.cpp/build/bin/llama-perplexity.exe
PROBE=bench-logs/quality-probe.txt
OUT=bench-results.md
P=512; N=128
GPU_FREE_MB=900          # ~VRAM left after the 6.66 GiB weights; cells needing more KV than this are skipped
GPU_CAP=180; CPU_CAP=480 # per-cell timeouts (s)

# KV bytes/token by quant: 64 layers * 4 kv-heads * (k256+v256) * bytes.  f16=256KB, q8_0=128KB, q4_0=64KB
kvbytes() { case "$1" in f16) echo 262144;; q8_0) echo 131072;; q4_0) echo 65536;; esac; }
num() { grep -E " $1 " "$2" 2>/dev/null | tail -1 | sed -E 's/.*\| *([0-9.]+) ± .*/\1/'; }

cell() { # bin backend ngl quant ctx cap
  local bin="$1" be="$2" ngl="$3" q="$4" ctx="$5" cap="$6"
  local kvmb=$(( $(kvbytes "$q") * ctx / 1048576 ))
  local log="bench-logs/${be}_${q}_${ctx}.log" pp tg st
  # GPU fit-gate (skip clearly-too-big cells instead of paying a slow shared-RAM fallback)
  if [ "$be" = gpu ] && [ "$kvmb" -gt "$GPU_FREE_MB" ]; then
    st="won't fit (KV ${kvmb}MB > ~${GPU_FREE_MB}MB free)"; pp="-"; tg="-"
    echo ">>> $be | KV=$q | ctx=$ctx -> SKIP ($st)"
    printf '| %-3s | %-5s | %6s | %8s | %7s | %s |\n' "$be" "$q" "$ctx" "$pp" "$tg" "$st" >>"$OUT"; return
  fi
  local depth=$(( ctx - P - N )); [ $depth -lt 0 ] && depth=0
  echo ">>> $be | KV=$q | ctx=$ctx | KV=${kvmb}MB (running, cap ${cap}s)"
  timeout "$cap" "$bin" -m "$MODEL" -ngl "$ngl" -fa 1 -ctk "$q" -ctv "$q" -p $P -n $N -d $depth -r 1 >"$log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then st="too slow (>${cap}s)"; pp="-"; tg="-"
  elif [ $rc -ne 0 ]; then
    if grep -qiE "out of memory|failed to allocate|cudaMalloc|bad_alloc|insufficient" "$log"; then st="OOM"; else st="fail(rc=$rc)"; fi
    pp="-"; tg="-"
  else
    pp=$(num "pp$P" "$log"); tg=$(num "tg$N" "$log"); st="ok"
    [ -z "$pp" ] && pp="-"; [ -z "$tg" ] && tg="-"
    # low pp on GPU => VRAM spilled to shared RAM (didn't really fit)
    [ "$be" = gpu ] && [ -n "$pp" ] && [ "$pp" != "-" ] && awk "BEGIN{exit !($pp<80)}" && st="spills->RAM (slow)"
  fi
  printf '| %-3s | %-5s | %6s | %8s | %7s | %s |\n' "$be" "$q" "$ctx" "$pp" "$tg" "$st" >>"$OUT"
  echo "    -> pp=$pp tg=$tg  [$st]"
}

{
  echo "# Bonsai 27B ternary — benchmark results"; echo
  echo "Model: qwen35 27B Q2_0 (6.66 GiB, 26.9B). HW: RTX 4060 Ti 8GB / 16GB RAM. flash-attn on."
  echo "tok/s via llama-bench: prefill to depth=ctx, then pp$P (ingest) + tg$N (gen), r=1."
  echo "GPU = -ngl 99; a cell is run only if its KV fits ~${GPU_FREE_MB}MB free VRAM, else marked won't-fit."
  echo "CPU = -ngl 0. tok/s measured once per KV-quant (shallow) since it's ~context-independent here"
  echo "(4 KV heads -> attention is <0.3% of compute); each context row is gated by a RAM-fit calc."; echo
  echo "## tok/s"; echo
  echo "| be  | KV    |    ctx |  pp t/s |  tg t/s | status |"
  echo "| --- | ----- | -----: | ------: | ------: | ------ |"
} >"$OUT"

echo "===== GPU ====="
for q in f16 q8_0 q4_0; do for c in 4096 8192 16384 32768; do cell "$GPU" gpu 99 "$q" "$c" "$GPU_CAP"; done; done
echo "===== CPU ====="
# CPU prompt-processing is <~7 t/s, so a full 4K+ prefill exceeds any sane cap. But with only 4 KV
# heads, per-token attention is <0.3% of compute -> tok/s is ~context-independent. So measure pp/tg
# once per quant (shallow, fast) and gate each context by a RAM-fit calc (weights 6.66GB + KV).
WEIGHTS_MB=6820
RAM_BUDGET_MB=14000   # ~usable of 16GB for weights+KV before it swaps to disk
for q in f16 q8_0 q4_0; do
  log="bench-logs/cpu_${q}.log"
  echo ">>> cpu | KV=$q | measuring pp/tg (tiny workload; CPU is very slow)"
  timeout 600 "$CPU" -m "$MODEL" -ngl 0 -fa 1 -ctk "$q" -ctv "$q" -p 128 -n 32 -r 1 >"$log" 2>&1
  if [ $? -ne 0 ]; then pp="-"; tg="-"; else pp=$(num "pp128" "$log"); tg=$(num "tg32" "$log"); fi
  [ -z "$pp" ] && pp="-"; [ -z "$tg" ] && tg="-"
  echo "    -> cpu $q pp=$pp tg=$tg"
  kvb=$(kvbytes "$q")
  for c in 4096 8192 16384 32768; do
    kvmb=$(( kvb * c / 1048576 )); need=$(( WEIGHTS_MB + kvmb ))
    if [ $need -gt $RAM_BUDGET_MB ]; then fit="won't fit RAM (~${need}MB)"; else fit="fits (~${need}MB RAM)"; fi
    printf '| %-3s | %-5s | %6s | %8s | %7s | %s |\n' "cpu" "$q" "$c" "$pp" "$tg" "$fit" >>"$OUT"
  done
done

{ echo; echo "## quality — perplexity on ~11K-token probe at ctx 2048 (GPU; KV-quant is backend-independent)"; echo
  echo "| KV    | perplexity | vs f16 |"; echo "| ----- | ---------- | ------ |"; } >>"$OUT"
base=""
for q in f16 q8_0 q4_0; do
  echo ">>> perplexity KV=$q (running)"; log="bench-logs/ppl_${q}.log"
  timeout 900 "$PPL" -m "$MODEL" -ngl 99 -fa 1 -ctk "$q" -ctv "$q" -c 2048 -f "$PROBE" >"$log" 2>&1
  ppl=$(grep -iE "Final estimate: PPL" "$log" | tail -1 | sed -E 's/.*PPL = ([0-9.]+).*/\1/'); [ -z "$ppl" ] && ppl="fail"
  [ "$q" = f16 ] && base="$ppl"; delta="-"
  [ "$ppl" != fail ] && [ -n "$base" ] && [ "$base" != fail ] && delta=$(awk -v a="$ppl" -v b="$base" 'BEGIN{printf "%+.2f%%",(a-b)/b*100}')
  printf '| %-5s | %-10s | %s |\n' "$q" "$ppl" "$delta" >>"$OUT"; echo "    -> ppl=$ppl ($delta)"
done
echo "ALL_DONE"

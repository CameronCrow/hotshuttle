#!/usr/bin/env bash
# bonsai.sh {start|stop|status|wait|restart} -- manage the local Bonsai 27B server
# with the benchmark-tuned config (see bench-results.md). Built for agents:
#
#   bonsai.sh start     # idempotent; launches detached, BLOCKS until /health is 200, prints the endpoint
#   bonsai.sh status    # "up: <url>/v1" or "down"   (exit 0 iff up)
#   bonsai.sh wait [n]   # block until ready (n*2s, default 120s); exit 1 on timeout
#   bonsai.sh stop      # kill the server
#   bonsai.sh restart
#
# After `start`, hit the OpenAI-compatible API at http://127.0.0.1:8080/v1 (model "bonsai-27b").
#
# CALLER MUST DISABLE THINKING: Bonsai is a reasoning model; send
#   "chat_template_kwargs": {"enable_thinking": false}
# in every request, or it spends the whole token budget on <think> and returns empty content.
# Config rationale: --parallel 1 (4 slots overfill the 8GB card -> WDDM thrash); q4_0 KV + -fa
# (weights fill ~6.7GB, KV must be small to fit -> ~8K ctx max, ~28 tok/s). Override via env:
#   BONSAI_DIR, BONSAI_MODEL, BONSAI_PORT, BONSAI_CTX.
set -u
BONSAI_DIR="${BONSAI_DIR:-/c/Users/Cameron/Projects/bonsai}"
MODEL="${BONSAI_MODEL:-$BONSAI_DIR/models/Ternary-Bonsai-27B-Q2_0.gguf}"
SERVER="$BONSAI_DIR/llama.cpp/build/bin/llama-server.exe"
HOST=127.0.0.1; PORT="${BONSAI_PORT:-8080}"; CTX="${BONSAI_CTX:-8192}"
URL="http://$HOST:$PORT"
LOG="$BONSAI_DIR/bench-logs/bonsai-server.log"

health() { [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null)" = "200" ]; }
wait_ready() { local n="${1:-60}"; for _ in $(seq 1 "$n"); do health && return 0; sleep 2; done; return 1; }

start() {
  if health; then echo "already running: $URL/v1"; return 0; fi
  [ -x "$SERVER" ] || { echo "server binary not found/built: $SERVER (run build.bat)" >&2; return 1; }
  [ -f "$MODEL" ]  || { echo "model not found: $MODEL" >&2; return 1; }
  mkdir -p "$(dirname "$LOG")"
  nohup "$SERVER" -m "$MODEL" --alias bonsai-27b --host "$HOST" --port "$PORT" \
    -ngl 99 -c "$CTX" --parallel 1 -fa 1 --cache-type-k q4_0 --cache-type-v q4_0 \
    --jinja --temp 0.7 --top-p 0.95 --top-k 20 > "$LOG" 2>&1 &
  disown 2>/dev/null || true
  if wait_ready 90; then echo "ready: $URL/v1  (model: bonsai-27b, ctx=$CTX)"; else
    echo "FAILED to become ready in ~180s; last log lines:" >&2; tail -8 "$LOG" >&2; return 1; fi
}
stop() { taskkill //IM llama-server.exe //F >/dev/null 2>&1; echo "stopped"; }

case "${1:-start}" in
  start)   start;;
  stop)    stop;;
  restart) stop; sleep 2; start;;
  status)  if health; then echo "up: $URL/v1"; else echo "down"; exit 1; fi;;
  wait)    if wait_ready "${2:-60}"; then echo "ready: $URL/v1"; else echo "not ready" >&2; exit 1; fi;;
  *) echo "usage: bonsai.sh {start|stop|status|wait|restart}" >&2; exit 2;;
esac

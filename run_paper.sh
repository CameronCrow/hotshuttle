#!/usr/bin/env bash
# One-shot driver: start Bonsai server (single slot), run the NIO paper pipeline, stop server.
# Self-contained so it's a single background task. Safe to re-run (pipeline resumes from cache).
set -u
BONSAI=/c/Users/Cameron/Projects/bonsai
NIO=/c/Users/Cameron/Projects/Neurological-Imitative-Organism-NIO
LLAMA="$BONSAI/llama.cpp/build/bin/llama-server.exe"

# fresh start: no stray servers
taskkill //IM llama-server.exe //F >/dev/null 2>&1
sleep 2

echo "STARTING_SERVER"
"$LLAMA" -m "$BONSAI/models/Ternary-Bonsai-27B-Q2_0.gguf" --alias bonsai-27b \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 --parallel 1 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 --jinja --temp 0.7 --top-p 0.95 --top-k 20 \
  > "$BONSAI/bench-logs/server2.log" 2>&1 &
SRV=$!

for i in $(seq 1 40); do
  if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null)" = "200" ]; then
    echo "SERVER_READY (~$((i*5))s)"; break
  fi
  sleep 5
done

# drop the empty/thinking-poisoned cache so factsheets are rebuilt cleanly
rm -rf "$NIO/paper/cache"

echo "RUNNING_PIPELINE"
cd "$NIO" && python -u paper/build_paper.py
RC=$?
echo "PIPELINE_RC=$RC"

taskkill //IM llama-server.exe //F >/dev/null 2>&1
echo "DRIVER_DONE rc=$RC"

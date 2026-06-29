#!/usr/bin/env bash
#
# Measure streaming TTFT vs blocking latency for the MultiDocChat RAG app.
#
# Prereq: the app must already be running in another terminal:  python main.py
# (and GROQ_API_KEY set in .env, since this makes real LLM calls).
#
# Usage:
#   scripts/measure_ttft.sh [HOST] [RUNS] [CORPUS_FILE] [QUESTION]
# Defaults:
#   HOST=http://localhost:8000  RUNS=3
#   CORPUS=eval/corpus/ai_engineering_report.txt  QUESTION="What is RAG?"
set -euo pipefail

HOST="${1:-http://localhost:8000}"
RUNS="${2:-3}"
CORPUS="${3:-eval/corpus/ai_engineering_report.txt}"
QUESTION="${4:-What is RAG?}"

# 0. Is the app up?
if ! curl -sf "$HOST/health" >/dev/null; then
  echo "ERROR: app not reachable at $HOST/health — start it first: python main.py" >&2
  exit 1
fi

# 1. Upload the corpus to get a session_id.
echo "Uploading $CORPUS ..."
SID=$(curl -s -X POST "$HOST/upload" -F "files=@$CORPUS" \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))")
if [ -z "$SID" ]; then
  echo "ERROR: upload failed (no session_id returned)" >&2
  exit 1
fi
echo "session: $SID"
PAYLOAD=$(python3 -c "import json,sys; print(json.dumps({'session_id':sys.argv[1],'message':sys.argv[2]}))" "$SID" "$QUESTION")

# 2. Blocking /chat — time to the full answer.
echo
echo "Blocking /chat  ($RUNS runs)..."
BLOCK=()
for i in $(seq 1 "$RUNS"); do
  t=$(curl -s -o /dev/null -w "%{time_total}" -X POST "$HOST/chat" \
        -H 'Content-Type: application/json' -d "$PAYLOAD")
  echo "  run $i: ${t}s"
  BLOCK+=("$t")
done

# 3. Streaming /chat/stream — time to first byte (TTFT) and full stream.
echo
echo "Streaming /chat/stream  ($RUNS runs)..."
TTFT=()
SFULL=()
for i in $(seq 1 "$RUNS"); do
  read -r ttft full < <(curl -sN -o /dev/null \
        -w "%{time_starttransfer} %{time_total}" -X POST "$HOST/chat/stream" \
        -H 'Content-Type: application/json' -d "$PAYLOAD")
  echo "  run $i: TTFT ${ttft}s | full ${full}s"
  TTFT+=("$ttft")
  SFULL+=("$full")
done

# 4. Medians + summary.
python3 - "${BLOCK[@]}" "__" "${TTFT[@]}" "__" "${SFULL[@]}" <<'PY'
import sys, statistics
rest = sys.argv[1:]
i = rest.index("__"); block = list(map(float, rest[:i])); rest = rest[i+1:]
j = rest.index("__"); ttft = list(map(float, rest[:j])); sfull = list(map(float, rest[j+1:]))
mb, mt, ms = statistics.median(block), statistics.median(ttft), statistics.median(sfull)
print()
print(f"{'metric':<34}{'median (s)':>12}")
print("-" * 46)
print(f"{'blocking /chat (full answer)':<34}{mb:>12.3f}")
print(f"{'streaming /chat/stream TTFT':<34}{mt:>12.3f}")
print(f"{'streaming /chat/stream (full)':<34}{ms:>12.3f}")
print("-" * 46)
if mt > 0:
    print(f"\nTTFT speedup vs blocking: {mb/mt:.1f}x  ({mb:.2f}s -> {mt:.2f}s first token)")
print("\nNote: medians of {} runs. Groq free-tier 429 retries can inflate times;".format(len(block)))
print("re-run if a number looks like an outlier.")
PY

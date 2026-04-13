#!/usr/bin/env bash
set -euo pipefail

MODEL_ARG="${1:-}"            # HF repo id OR local dir (model dir or HF-cache parent)
TP="${TP:-1}"
HOST="${HOST:-"127.0.0.1"}"
PORT="${PORT:-8000}"

# Base cache root for *your* project (matches your HF helper default)
CACHE_ROOT="${CACHE_ROOT:-./models}"

# Optional performance/memory tuning knobs (forwarded to vLLM if set)
# Examples:
#   GPU_MEM_UTIL=0.80  → trims preallocated KV cache pool
#   MAX_MODEL_LEN=8192 → limits context length, shrinking KV cache
#   MAX_NUM_SEQS=16    → caps concurrent sequences
#   MAX_BATCHED_TOKENS=4096 → caps total tokens per step
#   KV_CACHE_DTYPE=int8|fp8|fp16 (fp8 needs Hopper)
#   DTYPE=auto|float16|bfloat16|float32
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
DTYPE="${DTYPE:-}"

if [[ -z "$MODEL_ARG" ]]; then
  echo "Usage: $0 <HF_REPO_ID | MODEL_DIR | HF_CACHE_PARENT_DIR>"
  echo "Env: TP, HOST, PORT, CACHE_ROOT (default: ./models)"
  exit 1
fi

resolve_snapshot_dir () {
  local p="$1"
  if [[ -f "$p/config.json" ]]; then
    echo "[vLLM Launcher] Using model dir directly: $p" >&2
    echo "$p"; return 0
  fi
  if [[ -d "$p/refs" && -d "$p/snapshots" ]]; then
    if [[ -f "$p/refs/main" ]]; then
      local h; h="$(tr -d '[:space:]' < "$p/refs/main")"
      local snap="$p/snapshots/$h"
      if [[ -f "$snap/config.json" ]]; then
        echo "[vLLM Launcher] Resolved snapshot via refs/main: $snap" >&2
        echo "$snap"; return 0
      fi
    fi
    local newest; newest="$(ls -1dt "$p"/snapshots/*/ 2>/dev/null | head -n1 || true)"
    if [[ -n "${newest:-}" && -f "$newest/config.json" ]]; then
      echo "[vLLM Launcher] Resolved snapshot via newest: $newest" >&2
      echo "$newest"; return 0
    fi
  fi
  echo ""; return 1
}

MODEL_TO_SERVE="$MODEL_ARG"
EXTRA_FLAGS=()

if [[ -d "$MODEL_ARG" ]]; then
  # Local path → resolve if it's an HF-cache parent
  if resolved="$(resolve_snapshot_dir "$MODEL_ARG")"; then
    MODEL_TO_SERVE="$resolved"
  else
    echo "[vLLM Launcher] Could not resolve a model dir under: $MODEL_ARG" >&2
    exit 2
  fi
else
  # Treat as HF repo id ORG/NAME
  ORG="${MODEL_ARG%%/*}"
  NAME="${MODEL_ARG##*/}"

  # Preferred existing-cache location that matches your HF helper:
  #   ./models/<NAME>/models--<ORG>--<NAME>/
  CACHED_PARENT="$CACHE_ROOT/$NAME/models--${ORG//\//-}--$NAME"

  if [[ -d "$CACHED_PARENT" ]]; then
    echo "[vLLM Launcher] Found existing HF cache: $CACHED_PARENT" >&2
    if resolved="$(resolve_snapshot_dir "$CACHED_PARENT")"; then
      MODEL_TO_SERVE="$resolved"
    else
      echo "[vLLM Launcher] Cache exists but could not resolve snapshot; falling back to download." >&2
      mkdir -p "$CACHE_ROOT/$NAME"
      EXTRA_FLAGS+=(--download-dir "$CACHE_ROOT/$NAME")
    fi
  else
    # No existing cache in your layout → download into ./models/<NAME>
    echo "[vLLM Launcher] No local cache found; will download to: $CACHE_ROOT/$NAME" >&2
    mkdir -p "$CACHE_ROOT/$NAME"
    EXTRA_FLAGS+=(--download-dir "$CACHE_ROOT/$NAME")
  fi
fi

echo "[vLLM Launcher] Tensor Parallel: $TP  Host: $HOST  Port: $PORT" >&2
echo "[vLLM Launcher] Starting vLLM with model: $MODEL_TO_SERVE" >&2

# Forward optional flags only if provided
if [[ -n "$GPU_MEM_UTIL" ]]; then
  EXTRA_FLAGS+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  EXTRA_FLAGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
  EXTRA_FLAGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$MAX_BATCHED_TOKENS" ]]; then
  EXTRA_FLAGS+=(--max-num-batched-tokens "$MAX_BATCHED_TOKENS")
fi
if [[ -n "$KV_CACHE_DTYPE" ]]; then
  EXTRA_FLAGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ -n "$DTYPE" ]]; then
  EXTRA_FLAGS+=(--dtype "$DTYPE")
fi

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_TO_SERVE" \
  "${EXTRA_FLAGS[@]}" \
  --tensor-parallel-size "$TP" \
  --host "$HOST" \
  --port "$PORT"

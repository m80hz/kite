#!/usr/bin/env bash

MODEL_URL="http://127.0.0.1:8000/v1"
# Default model path to fall back to when server does not report models.
# Set FALLBACK_MODEL_PATH to your local model snapshot or HF repo id.
# FALLBACK_MODEL_PATH="${FALLBACK_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
# FALLBACK_MODEL_PATH="./models/Qwen2.5-VL-7B-Instruct/"
FALLBACK_MODEL_PATH="./models/Qwen2.5-VL-7B-Instruct/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"


# Try to auto-detect model id from the running API server. If that fails,
# fall back to the local snapshot path used previously.
MODEL_NAME="${FALLBACK_MODEL_PATH}"
if command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  set +e
  resp=$(curl -sS --max-time 2 "${MODEL_URL}/models" 2>/dev/null)
  rc=$?
  set -e
  if [ $rc -eq 0 ] && [ -n "${resp}" ]; then
    # Try to parse the first model id from the response JSON
    parsed=$(printf '%s' "${resp}" | python3 - <<'PY'
import sys, json
try:
    j = json.load(sys.stdin)
    # expected shape: {"data": [{"id": "<model-id>", ...}, ...]}
    data = j.get('data') or j.get('models') or []
    if data and isinstance(data, list) and len(data) > 0:
        # support either {'id':...} or simple list of ids
        first = data[0]
        if isinstance(first, dict) and 'id' in first:
            print(first['id'])
        elif isinstance(first, str):
            print(first)
except Exception:
    pass
PY
)
    if [ -n "${parsed}" ]; then
      MODEL_NAME="${parsed}"
      echo "[eval_kite.sh] Using detected model id: ${MODEL_NAME}"
    else
      echo "[eval_kite.sh] Could not parse model id from server; using fallback path: ${MODEL_NAME}"
    fi
  else
    echo "[eval_kite.sh] Could not reach model server at ${MODEL_URL}; using fallback model path: ${MODEL_NAME}"
  fi
else
  echo "[eval_kite.sh] curl or python3 not available; using fallback model path: ${MODEL_NAME}"
fi

echo "=============================="
echo "Evaluating KITE on real-world and simulation data"
echo "Using model: ${MODEL_NAME}"
echo "Model URL: ${MODEL_URL}"
echo "Output directory: ./outputs/${MODEL_NAME}/"
echo "=============================="
# stop here to let user confirm the settings before running the evaluation
read -p "Press Enter to continue with the evaluation..."

#  eval on sample real-world data
python -m kite.cli \
  --dataset_folder datasets/robofac/realworld_data \
  --model_name "${MODEL_NAME}" \
  --model_url "${MODEL_URL}" \
  --ovd_backend groundingdino \
  --enable_3d_graph \
  --robot_profile examples/robot_profiles/sim_single_arm.json \
  --dump_htatc \
  --out_dir "./outputs/${MODEL_NAME}/realworld" \
  --test_dir datasets/robofac/test_qa_realworld 
  # --enable_final_narrative \
  # --test_file datasets/robofac/test_qa_realworld/annos_per_video_split1.json \


python -m kite.cli \
  --dataset_folder datasets/robofac/simulation_data \
  --model_name "${MODEL_NAME}" \
  --model_url "${MODEL_URL}" \
  --ovd_backend groundingdino \
  --enable_3d_graph \
  --robot_profile examples/robot_profiles/sim_single_arm.json \
  --dump_htatc \
  --out_dir ./outputs/${MODEL_NAME}/sim \
  --test_dir datasets/robofac/test_qa_sim
  # --enable_final_narrative \
  # --test_file datasets/robofac/test_qa_sim/dummy.json


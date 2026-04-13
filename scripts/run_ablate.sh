#!/usr/bin/env bash
set -e

DATA_ROOT=${DATA_ROOT:-datasets/robofac/simulation_data}
TEST_DIR=${TEST_DIR:-datasets/robofac/test_qa_sim}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}
MODEL_URL=${MODEL_URL:-http://127.0.0.1:8000/v1}
OUT_ROOT=${OUT_ROOT:-./outputs/ablation_v7}
PROFILE=${PROFILE:-examples/robot_profiles/dart_dual_arm.json}
OVD=${OVD:-owlvit}
YOLO_WEIGHTS=${YOLO_WEIGHTS:-yolov8n.pt}

mkdir -p "${OUT_ROOT}"

python tools/ablate.py   --cli "python -m kite.cli"  --dataset_folder "${DATA_ROOT}"   --test_dir "${TEST_DIR}"   --model_name "${MODEL_NAME}"   --model_url "${MODEL_URL}"   --out_root "${OUT_ROOT}"   --robot_profile "${PROFILE}"   --yolo_weights "${YOLO_WEIGHTS}"   --ovd_backend "${OVD}"   --enable_3d_graph   --enable_final_narrative   --grid "PLAN,SCENE3D,EVENTS,ROBOT,CONTACT,BIMANUAL"

echo "Ablations done. Results in ${OUT_ROOT}/*/stats_data.json and htatc_dump.jsonl"

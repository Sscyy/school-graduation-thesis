#!/bin/bash
set -e

export PATH=$PATH:/home/hadoop-ba-dealrank/.local/bin

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TRAIN_DIR="${PROJECT_ROOT}/hover_net_new"

# ── 推理参数（在这里修改）────────────────────────────────────────────────────
CHECKPOINT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/results/hovernet_baseline/best.pth"
PARQUET="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/PanNuke/fold3-00000-of-00001.parquet"
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/results/hovernet_baseline/infer"
VIZ_SAMPLES=100
MODE="fast"
DEVICE="cuda"
# ────────────────────────────────────────────────────────────────────────────

echo "Project root: ${PROJECT_ROOT}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Output dir:   ${OUTPUT_DIR}"

# install dependencies
pip3 install -r "${PROJECT_ROOT}/requirements.txt" -i https://pip.sankuai.com/simple

cd "${TRAIN_DIR}"
python3 infer.py \
    --checkpoint  "${CHECKPOINT}" \
    --parquet     "${PARQUET}" \
    --output_dir  "${OUTPUT_DIR}" \
    --viz_samples "${VIZ_SAMPLES}" \
    --mode        "${MODE}" \
    --device      "${DEVICE}"

#!/bin/bash
set -e

export PATH=$PATH:/home/hadoop-ba-dealrank/.local/bin

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
INFER_DIR="${PROJECT_ROOT}/sam_hovernet"

# ── 推理参数（在这里修改）────────────────────────────────────────────────────
CHECKPOINT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/results/sam_amfr_uniform/best.pth"
SAM2_CHECKPOINT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/SAM_ckp/sam2.1_hiera_base_plus.pt"
PARQUET="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/PanNuke/fold3-00000-of-00001.parquet"
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/results/sam_amfr_uniform/infer"
VIZ_SAMPLES=100
MODE="fast"
FREEZE_STAGES=2
USE_BBE=False          # 需与训练时一致
USE_EA_SKIP=False      # 需与训练时一致
USE_EDGE_BRANCH=False  # 需与训练时一致
DEVICE="cuda"
# ────────────────────────────────────────────────────────────────────────────

echo "Project root: ${PROJECT_ROOT}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Output dir:   ${OUTPUT_DIR}"

pip3 install -r "${PROJECT_ROOT}/requirements.txt" -i https://pip.sankuai.com/simple

cd "${INFER_DIR}"
python3 infer.py \
    --checkpoint      "${CHECKPOINT}" \
    --sam2_checkpoint "${SAM2_CHECKPOINT}" \
    --parquet         "${PARQUET}" \
    --output_dir      "${OUTPUT_DIR}" \
    --viz_samples     "${VIZ_SAMPLES}" \
    --mode            "${MODE}" \
    --freeze_stages   "${FREEZE_STAGES}" \
    --use_bbe          "${USE_BBE}" \
    --use_ea_skip      "${USE_EA_SKIP}" \
    --use_edge_branch  "${USE_EDGE_BRANCH}" \
    --device           "${DEVICE}"

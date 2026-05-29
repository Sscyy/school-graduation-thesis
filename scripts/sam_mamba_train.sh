#!/bin/bash
set -e

export PATH=$PATH:/home/hadoop-ba-dealrank/.local/bin

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TRAIN_DIR="${PROJECT_ROOT}/sam_hovernet"

# ── 实验参数（在这里修改）────────────────────────────────────────────────────
# 第三章消融（A系列，EA/Edge均关闭）：
#   A2：USE_BBE=False  USE_EA_SKIP=False  USE_EDGE_BRANCH=False  doing
#   A3：USE_BBE=True   USE_EA_SKIP=False  USE_EDGE_BRANCH=False
# 第四章消融（B系列，BBE保持True）：
#   B2：USE_BBE=True   USE_EA_SKIP=False  USE_EDGE_BRANCH=True
#   B3：USE_BBE=True   USE_EA_SKIP=True   USE_EDGE_BRANCH=False
#   B4：USE_BBE=True   USE_EA_SKIP=True   USE_EDGE_BRANCH=True   done

# 数据
PANNUKE_ROOT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/PanNuke"
# 注：train_folds / val_folds 请在 sam_hovernet/config.py 中修改（List 类型）

# 模型
SAM2_CHECKPOINT="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/SAM_ckp/sam2.1_hiera_base_plus.pt"
FREEZE_STAGES=2
MODE="fast"
USE_BBE=True
USE_EA_SKIP=False
USE_EDGE_BRANCH=False

# 输出
OUTPUT_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-ba-dealrank/suchenyan/thesis/results/sam_A2"
SAVE_EVERY=10
LOG_INTERVAL=50

# 训练
EPOCHS=50
BATCH_SIZE=8
NUM_WORKERS=8
SEED=42
LR_ENCODER=1e-5
LR_AMFR=2e-4
LR_DECODER=2e-4
WEIGHT_DECAY=1e-4

# 损失权重（一般不需要改）
LOSS_NP_BCE=1.0
LOSS_NP_DICE=1.0
LOSS_HV_MSE=1.0
LOSS_HV_MSGE=1.0
LOSS_EDGE_DICE=1.0
# ────────────────────────────────────────────────────────────────────────────

echo "Project root:   ${PROJECT_ROOT}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "use_bbe:          ${USE_BBE}  use_ea_skip: ${USE_EA_SKIP}  use_edge_branch: ${USE_EDGE_BRANCH}"

# install dependencies
pip3 install -r "${PROJECT_ROOT}/requirements.txt" -i https://pip.sankuai.com/simple

# resolve torchrun launch command (single-node or multi-node via AFO_SPEC)
LAUNCH_CMD=$(python3 "${PROJECT_ROOT}/scripts/hope_torch_distributed_launch.py")

echo "Launch cmd: ${LAUNCH_CMD}"

cd "${TRAIN_DIR}"
${LAUNCH_CMD} train.py \
    --pannuke_root    "${PANNUKE_ROOT}" \
    --output_dir      "${OUTPUT_DIR}" \
    --sam2_checkpoint "${SAM2_CHECKPOINT}" \
    --freeze_stages   "${FREEZE_STAGES}" \
    --mode            "${MODE}" \
    --use_bbe          "${USE_BBE}" \
    --use_ea_skip      "${USE_EA_SKIP}" \
    --use_edge_branch  "${USE_EDGE_BRANCH}" \
    --epochs          "${EPOCHS}" \
    --batch_size      "${BATCH_SIZE}" \
    --num_workers     "${NUM_WORKERS}" \
    --seed            "${SEED}" \
    --lr_encoder      "${LR_ENCODER}" \
    --lr_amfr         "${LR_AMFR}" \
    --lr_decoder      "${LR_DECODER}" \
    --weight_decay    "${WEIGHT_DECAY}" \
    --save_every      "${SAVE_EVERY}" \
    --log_interval    "${LOG_INTERVAL}" \
    --loss_np_bce     "${LOSS_NP_BCE}" \
    --loss_np_dice    "${LOSS_NP_DICE}" \
    --loss_hv_mse     "${LOSS_HV_MSE}" \
    --loss_hv_msge    "${LOSS_HV_MSGE}" \
    --loss_edge_dice  "${LOSS_EDGE_DICE}"

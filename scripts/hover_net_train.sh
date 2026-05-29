#!/bin/bash
set -e

export PATH=$PATH:/home/hadoop-ba-dealrank/.local/bin

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
TRAIN_DIR="${PROJECT_ROOT}/hover_net_new"

echo "Project root: ${PROJECT_ROOT}"
echo "Train dir:    ${TRAIN_DIR}"

# install dependencies
pip3 install -r "${PROJECT_ROOT}/requirements.txt" -i https://pip.sankuai.com/simple

# resolve torchrun launch command (single-node or multi-node via AFO_SPEC)
LAUNCH_CMD=$(python3 "${PROJECT_ROOT}/scripts/hope_torch_distributed_launch.py")


cd "${TRAIN_DIR}"
${LAUNCH_CMD} train.py "$@"

# /bin/bash

CONFIG_FILE=$1
WORK_DIR=./output-train

python mmdetection/tools/train.py \
    ${CONFIG_FILE} \
    --work-dir ${WORK_DIR} \
    --amp \
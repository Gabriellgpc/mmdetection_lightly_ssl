#!/bin/bash
# @Author: Luis Condados
# @Date:   2023-07-09 22:39:56
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-07-15 11:56:35

TRAIN_CONFIG=$1
WORK_DIR=./train-output
GPUS=1

# mim train mmpetrain \
#     ${TRAIN_CONFIG} \
#     -G ${GPUS} \
#     --work-dir ${WORK_DIR} \
#     --amp

bash mmpretrain/tools/dist_train.sh \
    ${TRAIN_CONFIG} \
    ${GPUS} \
    --work-dir ${WORK_DIR} \
    --amp
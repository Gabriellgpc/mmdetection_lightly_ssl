#!/bin/bash

# @Author: Luis Condados
# @Date:   2023-07-02 22:56:19
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-07-02 23:22:09


apt install -y build-essentials htop
pip install nvitop

pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e . && cd ..
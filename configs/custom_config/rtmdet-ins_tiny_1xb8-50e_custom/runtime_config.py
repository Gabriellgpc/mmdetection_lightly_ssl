# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-07-02 23:56:42
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-07-03 01:58:44
_base_ = [
    './training_and_test_config.py',
]

default_scope = 'mmdet'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False

backend_args = None
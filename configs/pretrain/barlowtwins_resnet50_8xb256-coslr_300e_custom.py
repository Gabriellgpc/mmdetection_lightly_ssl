# -*- coding: utf-8 -*-
# @Author: Luis Condados
# @Date:   2023-07-15 11:45:11
# @Last Modified by:   Luis Condados
# @Last Modified time: 2023-07-15 12:16:53
dataset_type = 'CustomDataset'
data_root = '/home/lcondados/workspace/data/coco-2017/train'
batch_size = 2
max_epochs = 10
scale=8

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
view_pipeline1 = [
    dict(type='Resize', scale=(scale, scale), keep_ratio=True),
    dict(
        type='RandomResizedCrop',
        scale=scale,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1),
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(
            0.114,
            0.587,
            0.2989,
        )),
    dict(
        type='GaussianBlur',
        magnitude_range=(
            0.1,
            2.0,
        ),
        magnitude_std='inf',
        prob=1.0),
    dict(type='Solarize', thr=128, prob=0.0),
]
view_pipeline2 = [
    dict(type='Resize', scale=(scale, scale), keep_ratio=True),
    dict(
        type='RandomResizedCrop',
        scale=scale,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1),
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(
            0.114,
            0.587,
            0.2989,
        )),
    dict(
        type='GaussianBlur',
        magnitude_range=(
            0.1,
            2.0,
        ),
        magnitude_std='inf',
        prob=0.1),
    dict(type='Solarize', thr=128, prob=0.2),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiView',
        num_views=[
            1,
            1,
        ],
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline
    )
)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
model = dict(
    type='BarlowTwins',
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=8192,
        out_channels=8192,
        num_layers=3,
        with_last_bn=False,
        with_last_bn_affine=False,
        with_avg_pool=True,
        init_cfg=dict(
            type='Kaiming', distribution='uniform', layer=[
                'Linear',
            ])),
    head=dict(
        type='LatentCrossCorrelationHead',
        in_channels=8192,
        loss=dict(type='CrossCorrelationLoss')))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='LARS', lr=1.6, momentum=0.9, weight_decay=1e-06),
    paramwise_cfg=dict(
        custom_keys=dict({
            'bn':
            dict(decay_mult=0, lr_mult=0.024, lars_exclude=True),
            'bias':
            dict(decay_mult=0, lr_mult=0.024, lars_exclude=True),
            'downsample.1':
            dict(decay_mult=0, lr_mult=0.024, lars_exclude=True)
        })))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00016,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=0.0016,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
auto_scale_lr = dict(base_batch_size=2048)

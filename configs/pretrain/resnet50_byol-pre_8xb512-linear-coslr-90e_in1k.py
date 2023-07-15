model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=4),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(
            1,
            5,
        )))
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=1000,
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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=512,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        split='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224, backend='pillow'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        split='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(
    type='Accuracy', topk=(
        1,
        5,
    ))
test_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        split='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(
    type='Accuracy', topk=(
        1,
        5,
    ))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='LARS', lr=1.6, momentum=0.9, weight_decay=0.0))
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=0, end=90),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)
val_cfg = dict()
test_cfg = dict()
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
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

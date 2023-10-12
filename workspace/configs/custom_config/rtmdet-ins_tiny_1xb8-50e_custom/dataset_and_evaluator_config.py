
_base_ = [
    './base_hparams.py',
]

dataset_type = 'CocoDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='CachedMosaic',
        img_scale={{_base_.imsize}},
        pad_val=114.0,
        max_cached_images=10,
        random_pop=False),
    dict(type='RandomResize', scale=(320, 320), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size={{_base_.imsize}}),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size={{_base_.imsize}}, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale={{_base_.imsize}},
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale={{_base_.imsize}}, keep_ratio=True),
    dict(type='Pad', size={{_base_.imsize}}, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


train_dataloader = dict(
    batch_size={{_base_.batch_size}},
    num_workers={{_base_.num_workers}},
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        metainfo={{_base_.metainfo}},
        data_root={{_base_.data_root}},
        ann_file={{_base_.train_annotation_file}},
        data_prefix=dict(img={{_base_.train_images_dir}}),
        test_mode=False,
        pipeline=train_pipeline,
        backend_args=None),
    pin_memory=True)

val_dataloader = dict(
    batch_size={{_base_.batch_size}},
    num_workers={{_base_.num_workers}},
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo={{_base_.metainfo}},
        type='CocoDataset',
        data_root={{_base_.data_root}},
        ann_file={{_base_.val_annotation_file}},
        data_prefix=dict(img={{_base_.val_images_dir}}),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None)
)

test_dataloader = dict(
    batch_size={{_base_.batch_size}},
    num_workers={{_base_.num_workers}},
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo={{_base_.metainfo}},
        type='CocoDataset',
        data_root={{_base_.data_root}},
        ann_file={{_base_.test_annotation_file}},
        data_prefix=dict(img={{_base_.test_images_dir}}),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

val_evaluator = dict(
    type='CocoMetric',
    ann_file={{_base_.val_evaluator_annotation_file}},
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10))

test_evaluator = dict(
    type='CocoMetric',
    ann_file={{_base_.test_evaluator_annotation_file}},
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10))

#####################################
#      Train pipeline stage 2
#####################################

img_scales = [{{_base_.imsize}}, (128, 128)]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale={{_base_.imsize}},
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size={{_base_.imsize}},
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size={{_base_.imsize}}, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

#####################################
# Test Time Augmentation (TTA)
#####################################

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                {'type': 'Resize', 'scale': {{_base_.imsize}}, 'keep_ratio': True},
            ],
            [
                {'type': 'RandomFlip', 'prob': 1.0},
                {'type': 'RandomFlip', 'prob': 0.0}
            ],
            [
                {'type': 'Pad', 'size': {{_base_.imsize}}, 'pad_val':{'img': (114, 114, 114)}}
            ],
            [
                {'type': 'LoadAnnotations', 'with_bbox': True}
            ],
            [
                {'type':'PackDetInputs', 'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor', 'flip', 'flip_direction')}
            ]
        ]
    )
]


tta_model = dict( type='DetTTAModel', tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
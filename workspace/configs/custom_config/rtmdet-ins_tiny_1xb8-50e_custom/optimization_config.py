_base_ = [
    './dataset_and_evaluator_config.py',
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr={{_base_.base_lr}}, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

auto_scale_lr = dict(enable=False, base_batch_size={{_base_.batch_size}})

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin={{_base_.param_scheduler_begin}},
        end={{_base_.max_epochs}},
        T_max={{_base_.param_scheduler_begin}},
        by_epoch=True,
        convert_to_iter_based=True)
]

_base_ = [
    './model_config.py'
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs={{_base_.max_epochs}},
    val_interval={{_base_.eval_interval}})

val_cfg = dict(type='ValLoop')

test_cfg = dict(type='TestLoop')
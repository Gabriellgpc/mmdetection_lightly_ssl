# dataloader hparams
data_root = '/home/lcondados/workspace/data/hair-seg'
train_images_dir = 'train/'
train_annotation_file = 'train/_annotations.coco.json'

val_images_dir = 'valid/'
val_annotation_file = 'valid/_annotations.coco.json'

test_images_dir = 'test/'
test_annotation_file = 'test/_annotations.coco.json'

val_evaluator_annotation_file = f'{data_root}/{val_annotation_file}'
test_evaluator_annotation_file = val_evaluator_annotation_file

metainfo = {
    'classes': ('hair', ),
    'palette': [
        (220, 20, 60),
    ]
}

imsize = (128, 128)
batch_size = 4
num_workers= 10

# train and testing hparams
max_epochs = 1000
base_lr = 0.004

stage2_num_epochs = 10

eval_interval = 1

# model hparams
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
num_classes = 1

# optimization hparams
param_scheduler_begin = max_epochs//2

# hooks hparams
log_interval = 50
save_checkpoint_interval = 1
switch_epoch = max_epochs//2  # when to swtich to a different data pipeline
_base_ = [
    '/root/mmpretrain/configs/_base_/models/resnet50.py',           # 模型设置
    '/root/mmpretrain/configs/_base_/datasets/imagenet_bs32.py',    # 数据设置
    '/root/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '/root/mmpretrain/configs/_base_/default_runtime.py',           # 运行设置
]
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=30,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    init_cfg=dict(
           type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',prefix='backbone',
        )
    )
# dataset settings
dataset_type = 'CustomDataset'
data_root = '/root/'
data_preprocessor = dict(
    num_classes=30,
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=True)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),  
#     dict(type='RandomCrop', crop_size=32, padding=4),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomCrop', crop_size=32, padding=4),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='fruit30_train',
        ann_file='',
        with_label=True,),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='fruit30_train',
        ann_file='',
        with_label=True,),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[40, 50], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=128)
# defaults to use registries in mmpretrain
# default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    
    checkpoint=dict(type='CheckpointHook', interval=50),
)

# # configure environment
# env_cfg = dict(
#     # whether to enable cudnn benchmark
#     cudnn_benchmark=False,

#     # set multi process parameters
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

#     # set distributed parameters
#     dist_cfg=dict(backend='nccl'),
# )

# # set visualizer
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# # set log level
# log_level = 'INFO'

# # load from which checkpoint
# load_from = None

# # whether to resume training from the loaded checkpoint
# resume = False

# # Defaults to use random seed and disable `deterministic`
# randomness = dict(seed=None, deterministic=False)

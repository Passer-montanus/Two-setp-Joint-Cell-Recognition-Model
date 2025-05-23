model_cfg = dict(
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=8,
        base_width=26,
        deep_stem=False,
        avg_down=False,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=4,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataloader pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    # dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# train
data_cfg = dict(
    batch_size = 64,
    num_workers = 8,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = '/root/autodl-tmp/datas/premodel/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 200,
    ),
    test=dict(
        ckpt = 'D:/Awesome-Backbones-main/new_logs/Res2Net/runs/Train_Epoch098-Loss0.031.pth',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# batch 32
# lr = 0.1 *32 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.1 * 32/256,
    momentum=0.9,
    weight_decay=1e-4)

# learning 
lr_config = dict(
    type='StepLrUpdater',
    step=[40, 80, 120]
)

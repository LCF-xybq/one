exp_name = 'Underwater Image Enhancement'

model = dict(
    type='UIEC',
    generator=dict(
        type='UIECNet'),
    loss_percep=dict(
        type='PerceptualLoss',
        vgg_type='vgg19',
        layer_weights={
            '4': 1.,
        },
        perceptual_weight=0.05,
        style_weight=0.0,
        pretrained=('torchvision://vgg19')),
    loss_l1=dict(
        type='L1Loss',
        loss_weight=1.,
    ),
    loss_tv=dict(
        type='MaskedTVLoss',
        loss_weight=0.1,
    ),
    pretrained=None)

train_cfg = None
test_cfg = dict(metrics=['psnr', 'ssim'], crop_border=4)

# dataset settings
dataset_type = 'UWFolderDataset'

input_shape = (256, 256)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(
        type='Resize',
        keys=['lq', 'gt'],
        scale=input_shape,
        keep_ratio=False,
    ),
    dict(
        type='Crop',
        keys=['lq', 'gt'],
        crop_size=(224, 224),
        random_crop=True,
    ),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path'])
]

train_raw = r'D:\Program_self\DeepLearning\paper_mm\data\UIEB\train\raw'
train_ref = r'D:\Program_self\DeepLearning\paper_mm\data\UIEB\train\ref'
val_raw = r'D:\Program_self\DeepLearning\paper_mm\data\UIEB\val\raw'
val_ref = r'D:\Program_self\DeepLearning\paper_mm\data\UIEB\val\ref'

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=dataset_type,
        lq_folder=train_raw,
        gt_folder=train_ref,
        # ann_file='data/UIEB/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        lq_folder=val_raw,
        gt_folder=val_ref,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        lq_folder=val_raw,
        gt_folder=val_ref,
        pipeline=test_pipeline),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1.e-4))

# learning policy
total_iters = 3000
lr_config = dict(policy='Step', by_epoch=False, step=[2000], gamma=0.5)

checkpoint_config = dict(interval=50, save_optimizer=True, by_epoch=False)
# evaluation = dict(interval=50, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
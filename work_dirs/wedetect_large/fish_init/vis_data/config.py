artaxor_evaluator = dict(
    ann_file='data/ArTaxOr/annotations/test.json',
    metric='bbox',
    type='CocoMetric')
artaxor_val_dataset = dict(
    class_text_path='data/texts/artaxor_zh_class_texts.json',
    dataset=dict(
        ann_file='annotations/test.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='test'),
        data_root='data/ArTaxOr',
        debug_mode=False,
        metainfo=dict(classes='data/texts/artaxor_class_texts.json'),
        test_mode=True,
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1280,
            1280,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1280,
                1280,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
backend_args = None
base_lr = 0.0005
clipart1k_evaluator = dict(
    ann_file='data/clipart1k/annotations/test.json',
    metric='bbox',
    type='CocoMetric')
clipart1k_val_dataset = dict(
    class_text_path='data/texts/clipart1k_zh_class_texts.json',
    dataset=dict(
        ann_file='annotations/test.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='test'),
        data_root='data/clipart1k',
        debug_mode=False,
        metainfo=dict(classes='data/texts/clipart1k_class_texts.json'),
        test_mode=True,
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1280,
            1280,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1280,
                1280,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
close_mosaic_epochs = 2
coco_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    metric='bbox',
    type='CocoMetric')
coco_val_dataset = dict(
    class_text_path='data/texts/coco_zh_class_texts.json',
    dataset=dict(
        ann_file='data/coco/annotations/instances_val2017.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='val2017'),
        data_root='data/coco/',
        test_mode=True,
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1280,
            1280,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1280,
                1280,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
current_dataset = 'fish'
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'wedetect',
    ])
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmdet'
dist_cfg = dict(backend='nccl', timeout=10800)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
fish_evaluator = dict(
    ann_file='data/FISH/annotations/test.json',
    metric='bbox',
    type='CocoMetric')
fish_val_dataset = dict(
    class_text_path='data/texts/fish_zh_class_texts.json',
    dataset=dict(
        ann_file='annotations/test.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='test'),
        data_root='data/FISH',
        debug_mode=False,
        metainfo=dict(classes='data/texts/fish_class_texts.json'),
        test_mode=True,
        type='WeCocoDataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1280,
            1280,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1280,
                1280,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
img_scale = (
    1280,
    1280,
)
launcher = 'pytorch'
load_from = 'checkpoints/WeDetect/wedetect_large.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lvis_minival_dataset = dict(
    class_text_path='data/texts/lvis_v1_zh_class_texts.json',
    dataset=dict(
        ann_file='data/lvis/lvis_v1_minival_inserted_image_name.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root='data/coco/',
        test_mode=True,
        type='YOLOv5LVISV1Dataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1280,
            1280,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1280,
                1280,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
lvis_minival_evaluator = dict(
    ann_file='data/lvis/lvis_v1_minival_inserted_image_name.json',
    metric='bbox',
    type='LVISMetric')
lvis_od_val_dataset = dict(
    class_text_path='data/texts/lvis_v1_zh_class_texts.json',
    dataset=dict(
        ann_file='data/lvis/lvis_od_val.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img=''),
        data_root='data/coco/',
        test_mode=True,
        type='YOLOv5LVISV1Dataset'),
    pipeline=[
        dict(backend_args=None, type='LoadImageFromFile'),
        dict(scale=(
            1280,
            1280,
        ), type='WeDetectKeepRatioResize'),
        dict(
            allow_scale_up=False,
            pad_val=dict(img=114),
            scale=(
                1280,
                1280,
            ),
            type='WeDetectLetterResize'),
        dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
        dict(type='LoadText'),
        dict(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'pad_param',
                'texts',
            ),
            type='PackDetInputs'),
    ],
    type='MultiModalDataset')
lvis_od_val_evaluator = dict(
    ann_file='data/lvis/lvis_od_val.json', metric='bbox', type='LVISMetric')
max_epochs = 80
model = dict(
    backbone=dict(
        image_model=dict(
            frozen_modules=[],
            model_name='large',
            type='ConvNextVisionBackbone'),
        text_model=dict(
            frozen_modules=[],
            model_name='./xlm-roberta-large/',
            model_size='large',
            type='XLMRobertaLanguageBackbone'),
        type='MultiModalYOLOBackbone'),
    bbox_head=dict(
        bbox_coder=dict(type='WeDetectDistancePointBBoxCoder'),
        head_module=dict(
            embed_dims=768,
            in_channels=[
                256,
                512,
                1024,
            ],
            model_size='large',
            num_classes=80,
            type='YOLOWorldHeadModule',
            use_bn_head=True),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='mmyoloIoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375, reduction='mean', type='DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        type='YOLOWorldHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        type='YOLOWDetDataPreprocessor'),
    mm_neck=False,
    neck=dict(model_size='large', scale_factor=1.5, type='CSPRepBiFPANNeck'),
    num_test_classes=1203,
    num_train_classes=80,
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=1203,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLOWorldDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
neck_embed_channels = [
    128,
    256,
    512,
]
neck_num_heads = [
    4,
    8,
    16,
]
num_classes = 1203
num_training_classes = 80
resume = False
save_epoch_intervals = 1
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/texts/fish_zh_class_texts.json',
        dataset=dict(
            ann_file='annotations/test.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='test'),
            data_root='data/FISH',
            debug_mode=False,
            metainfo=dict(classes='data/texts/fish_class_texts.json'),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1280,
                1280,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    1280,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/FISH/annotations/test.json',
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        1280,
        1280,
    ), type='WeDetectKeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            1280,
            1280,
        ),
        type='WeDetectLetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(type='LoadText'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
            'texts',
        ),
        type='PackDetInputs'),
]
text_channels = 768
train_batch_size_per_gpu = 10
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        class_text_path='data/texts/fish_zh_class_texts.json',
        dataset=dict(
            ann_file='annotations/test.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='test'),
            data_root='data/FISH',
            debug_mode=False,
            metainfo=dict(classes='data/texts/fish_class_texts.json'),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1280,
                1280,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    1280,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_dataset_list = dict(
    artaxor=dict(
        class_text_path='data/texts/artaxor_zh_class_texts.json',
        dataset=dict(
            ann_file='annotations/test.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='test'),
            data_root='data/ArTaxOr',
            debug_mode=False,
            metainfo=dict(classes='data/texts/artaxor_class_texts.json'),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1280,
                1280,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    1280,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    clipart1k=dict(
        class_text_path='data/texts/clipart1k_zh_class_texts.json',
        dataset=dict(
            ann_file='annotations/test.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='test'),
            data_root='data/clipart1k',
            debug_mode=False,
            metainfo=dict(classes='data/texts/clipart1k_class_texts.json'),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1280,
                1280,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    1280,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'),
    fish=dict(
        class_text_path='data/texts/fish_zh_class_texts.json',
        dataset=dict(
            ann_file='annotations/test.json',
            batch_shapes_cfg=None,
            data_prefix=dict(img='test'),
            data_root='data/FISH',
            debug_mode=False,
            metainfo=dict(classes='data/texts/fish_class_texts.json'),
            test_mode=True,
            type='WeCocoDataset'),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                1280,
                1280,
            ), type='WeDetectKeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    1280,
                    1280,
                ),
                type='WeDetectLetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                    'texts',
                ),
                type='PackDetInputs'),
        ],
        type='MultiModalDataset'))
val_evaluator = dict(
    ann_file='data/FISH/annotations/test.json',
    metric='bbox',
    type='CocoMetric')
val_evaluator_list = dict(
    artaxor=dict(
        ann_file='data/ArTaxOr/annotations/test.json',
        metric='bbox',
        type='CocoMetric'),
    clipart1k=dict(
        ann_file='data/clipart1k/annotations/test.json',
        metric='bbox',
        type='CocoMetric'),
    fish=dict(
        ann_file='data/FISH/annotations/test.json',
        metric='bbox',
        type='CocoMetric'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.025
work_dir = './work_dirs/wedetect_large'

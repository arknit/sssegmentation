'''base config for segformer'''
# config for dataset
DATASET_CFG = {
    'type': '',
    'rootdir': '',
    'train': {
        'set': 'train',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}
# config for dataloader
DATALOADER_CFG = {
    'train': {
        'type': ['nondistributed', 'distributed'][1],
        'batch_size': 16,
        'num_workers': 16,
        'shuffle': True,
        'pin_memory': True,
        'drop_last': True,
    },
    'test': {
        'type': ['nondistributed', 'distributed'][1],
        'batch_size': 1,
        'num_workers': 16,
        'shuffle': False,
        'pin_memory': True,
        'drop_last': False,
    }
}
# config for optimizer
OPTIMIZER_CFG = {
    'type': 'adamw',
    'adamw': {
        'learning_rate': 0.00006,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
        'min_lr': 0.0,
    },
    'max_epochs': 0,
    'params_rules': {'backbone_net_zerowd': (1.0, 0.0), 'backbone_net_nonzerowd': (1.0, 1.0), 'others': (10.0, 1.0)},
    'policy': {
        'type': 'poly',
        'opts': {'power': 1.0, 'max_iters': None, 'num_iters': None, 'num_epochs': None},
        'warmup': {'type': 'linear', 'ratio': 1e-6, 'iters': 1500},
    },
    'adjust_period': ['iteration', 'epoch'][0],
}
# config for losses
LOSSES_CFG = {
    'loss_cls': {
        'celoss': {'scale_factor': 1.0, 'opts': {'ignore_index': 255, 'reduction': 'mean'}}
    },
}
# config for segmentor
SEGMENTOR_CFG = {
    'type': 'segformer',
    'num_classes': -1,
    'benchmark': True,
    'is_multi_gpus': True,
    'align_corners': False,
    'distributed': {'is_on': True, 'backend': 'nccl'},
    'norm_cfg': {'type': 'syncbatchnorm', 'opts': {}},
    'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
    'backbone': {
        'type': 'mit-b0',
        'series': 'mit',
        'pretrained': True,
        'pretrained_model_path': 'mit_b0.pth',
        'selected_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm', 'opts': {'eps': 1e-6}},
    },
    'decoder': {
        'in_channels_list': [32, 64, 160, 256],
        'out_channels': 256,
        'dropout': 0.1,
    },
}
# config for inference
INFERENCE_CFG = {
    'mode': 'whole',
    'opts': {}, 
    'tricks': {
        'multiscale': [1],
        'flip': False,
        'use_probs_before_resize': True
    }
}
# config for common
COMMON_CFG = {
    'train': {
        'backupdir': '',
        'logfilepath': '',
        'loginterval': 50,
        'saveinterval': 1
    },
    'test': {
        'backupdir': '',
        'logfilepath': '',
        'resultsavepath': ''
    }
}
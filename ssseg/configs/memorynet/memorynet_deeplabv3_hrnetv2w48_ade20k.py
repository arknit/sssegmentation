'''memorynet_deeplabv3_hrnetv2w48_ade20k'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'ade20k',
    'rootdir': os.path.join(os.getcwd(), 'ADE20k'),
})
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update({
    'type': 'sgd',
    'sgd': {
        'learning_rate': 0.004,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    },
    'max_epochs': 180
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
LOSSES_CFG.pop('loss_aux')
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 150,
    'backbone': {
        'type': 'hrnetv2_w48',
        'series': 'hrnet',
        'pretrained': True,
        'selected_indices': (0, 0, 0, 0),
    },
    'auxiliary': None,
})
SEGMENTOR_CFG['memory']['in_channels'] = sum([48, 96, 192, 384])
SEGMENTOR_CFG['memory']['update_cfg']['momentum_cfg']['base_lr'] = 0.004
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynet_deeplabv3_hrnetv2w48_ade20k'
COMMON_CFG['logfilepath'] = 'memorynet_deeplabv3_hrnetv2w48_ade20k/memorynet_deeplabv3_hrnetv2w48_ade20k.log'
COMMON_CFG['resultsavepath'] = 'memorynet_deeplabv3_hrnetv2w48_ade20k/memorynet_deeplabv3_hrnetv2w48_ade20k_results.pkl'
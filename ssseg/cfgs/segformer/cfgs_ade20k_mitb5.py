'''define the config file for ade20k and mit-b5'''
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
OPTIMIZER_CFG.update(
    {
        'max_epochs': 130
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update(
    {
        'num_classes': 150,
        'backbone': {
            'type': 'mit-b5',
            'series': 'mit',
            'pretrained': True,
            'pretrained_model_path': 'mit_b5.pth',
            'selected_indices': (0, 1, 2, 3),
            'norm_cfg': {'type': 'layernorm', 'opts': {'eps': 1e-6}},
        },
        'decoder': {
            'in_channels_list': [64, 128, 320, 512],
            'out_channels': 256,
            'dropout': 0.1,
        },
    }
)
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'segformer_mitb5_ade20k_train',
        'logfilepath': 'segformer_mitb5_ade20k_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'segformer_mitb5_ade20k_test',
        'logfilepath': 'segformer_mitb5_ade20k_test/test.log',
        'resultsavepath': 'segformer_mitb5_ade20k_test/segformer_mitb5_ade20k_results.pkl'
    }
)
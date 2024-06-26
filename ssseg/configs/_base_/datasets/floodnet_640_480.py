'''floodnet_640x480'''
import os


'''DATASET_CFG_FLOODNET_640x480'''
DATASET_CFG_FLOODNET_640X480 = {
    'type': 'FloodNet',
    'rootdir': os.path.join(os.getcwd(), 'floodnet'),
    'train': {
        'set': 'train',
        'data_pipelines': [
            # ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            # ('RandomCrop', {'crop_size': (512, 1024), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 1024), 'data_type': 'tensor'}),
        ],
    },
    'test': {
        'set': 'val',
        'data_pipelines': [
            # ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}

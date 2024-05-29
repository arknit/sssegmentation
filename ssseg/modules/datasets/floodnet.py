import os
import pandas as pd
from .base import BaseDataset

"""
VERSION NO. = FloodNet-Supervised_v1.0
##----------------------------------------------
Features:
  1. Dataset is distributed same as FloodNet challenge.
  2. Masks are same size as original images.
  3. Total class: 10 ('Background':0, 'Building-flooded':1, 'Building-non-flooded':2, 'Road-flooded':3, 'Road-non-flooded':4, 'Water':5, 'Tree':6, 'Vehicle':7, 'Pool':8, 'Grass':9).
  4. Total image: 2343 (Train: 1445, Val: 450, Test: 448)
"""


'''FloodNet supervised dataset'''
class FloodNet(BaseDataset):
    num_classes = 10
    classnames = ['__background__', 'building-flooded', 'building-non-flooded', 
                  'road-flooded', 'road-non-flooded',
                  'water', 'tree', 'vehicle', 'pool', 'grass']
    # palette = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), 
    #            (5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8), (9, 9,9)]
    palette = [(127, 63, 63), (127, 106, 63), (106, 127, 63), (63, 127, 63), (63, 127, 106),
     (63, 106, 127), (63, 63, 127), (106, 63, 127), (127, 63, 106), (127, 63, 63)]
    clsid2label = {0: 0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}
    assert num_classes == len(classnames) and num_classes == len(palette)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(FloodNet, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        # train/train-org-img
        self.image_dir = os.path.join(rootdir, dataset_cfg['set'], dataset_cfg['set'] + "-org-img")
        self.ann_dir = os.path.join(rootdir, dataset_cfg['set'], dataset_cfg['set'] + "-label-img")
        # obatin imageids
        df = pd.read_csv(os.path.join(rootdir, dataset_cfg['set']+'.txt'), names=['imageids'])
        self.imageids = df['imageids'].values
        self.imageids = [str(_id) for _id in self.imageids]
        self.ann_ext = '.png'
        self.image_ext = '.png'

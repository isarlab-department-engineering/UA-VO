#!/usr/bin/env python

import numpy as np
import torch
import random
import os

from __init__ import ROOT_PATH
from vo_framework.utils import get_config
from trainer import Trainer

if __name__ == "__main__":

    config = get_config('configurations/uavo_config.yaml')

    seed = config['random_seed']
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    config['kitti_all_dirs'] = ['00', '01', '02',
                                '03', '04', '05',
                                '06', '07', '08', '09', '10']

    config['kitti_train_dirs'] = ['00', '01', '02',
                                  '03', '04', '05',
                                  '06', '07']



    trainer = Trainer(config)
    with torch.cuda.device(config['gpu_id']):
        trainer.test(stochastic_eval=True)

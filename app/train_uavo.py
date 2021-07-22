#!/usr/bin/env python

import numpy as np
import torch
import random

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

    config['kitti_all_dirs'] = ['04']

    config['kitti_train_dirs'] = ['04']


    trainer = Trainer(config)
    trainer.train()

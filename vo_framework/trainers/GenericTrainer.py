import os
import shutil

from tensorboardX import SummaryWriter


class GenericTrainer(object):

    def __init__(self, model, opt):

        self.model = model
        self.opt = opt
        self.writer_dir = ''

    def init_summary_writer(self, task, dataset_name):

        self.writer_dir = os.path.join('runs', task, 'UA-VO', dataset_name, self.opt['exp_name'])

        if os.path.exists(self.writer_dir):
            shutil.rmtree(self.writer_dir)
        os.makedirs(self.writer_dir)

        self.writer = SummaryWriter(log_dir=self.writer_dir)

    def init_data_loader(self):
        pass

    def train(self):
        pass

    def test(self, epoch=-1):
        pass

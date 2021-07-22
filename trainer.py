from models.UAVO import UAVOModel
from trainers.UAVOTrainer import UAVOTrainer

from vo_framework.ws_definitions import LOG_PATH


class Trainer(object):
    def __init__(self, config):
        self.config = config

        self.model_dir = LOG_PATH

        if self.config['strategy'] == 'uavo':
            model = UAVOModel(config)
            self.trainer = UAVOTrainer(model, self.config)


    def train(self):
        print("[*] Training starts...")
        self.trainer.init_data_loader()
        self.trainer.train()

    def test(self, stochastic_eval):
        self.trainer.init_data_loader()
        self.trainer.test(epoch=self.config['start_epoch'], stochastic_eval=stochastic_eval)

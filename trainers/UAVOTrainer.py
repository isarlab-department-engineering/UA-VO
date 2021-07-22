from torch.utils.data import DataLoader
import torch
from vo_framework.datasets.sample_type.SampleTransform import  Rescale
from vo_framework.datasets.VODataset import KittiDataset, ISARLABCARDataset, MalagaDataset
from vo_framework.trainers.GenericTrainer import GenericTrainer
from vo_framework.datasets.Dataset import TrainingDataset
from datasets.DataGenerationStrategy import SequenceGenerationStrategy


class UAVOTrainer(GenericTrainer):
    def __init__(self, model, opt):
        super(UAVOTrainer, self).__init__(model, opt)

        self.dataset = {
            'train': None,
            'test': None
        }

    def init_data_loader(self):

        data_generation_strategy_train = SequenceGenerationStrategy(
            num_steps=self.opt['timesteps'],
            ram_pre_loading=self.opt['ram_pre_loading'],
            transforms=[Rescale(output_size=(self.opt['input_height'], self.opt['input_width'])),
                        ]
        )

        data_generation_strategy_test = SequenceGenerationStrategy(
            num_steps=self.opt['timesteps'],
            ram_pre_loading=self.opt['ram_pre_loading'],
            transforms=[Rescale(output_size=(self.opt['input_height'], self.opt['input_width'])),
                      ]
        )

        if self.opt['use_Kitti']:
            print('--------------------------------------')
            print('------Processing Kitti Dataset--------')
            print('--------------------------------------')

            kitti_trainset = KittiDataset(
                config=self.opt,
                mode=TrainingDataset.TRAIN,
                sample_generation_strategy=data_generation_strategy_train
            )
            kitti_testset = KittiDataset(
                self.opt,
                mode=TrainingDataset.TEST,
                sample_generation_strategy=data_generation_strategy_test
            )
            self.dataset['train'] = kitti_trainset
            self.dataset['test'] = kitti_testset

        if self.opt['use_ISARLAB_CAR']:
            print('--------------------------------------')
            print('------Processing ISARLAB CAR Dataset--------')
            print('--------------------------------------')

            isarlab_car_trainset = ISARLABCARDataset(
                config=self.opt,
                mode=TrainingDataset.TRAIN,
                sample_generation_strategy=data_generation_strategy_train
            )
            isarlab_car_testset = ISARLABCARDataset(
                self.opt,
                mode=TrainingDataset.TEST,
                sample_generation_strategy=data_generation_strategy_test
            )
            self.dataset['train'] = isarlab_car_trainset
            self.dataset['test'] = isarlab_car_testset

        if self.opt['use_Malaga']:
            print('--------------------------------------')
            print('------Processing Malaga Dataset--------')
            print('--------------------------------------')

            malaga_trainset = MalagaDataset(
                config=self.opt,
                mode=TrainingDataset.TRAIN,
                sample_generation_strategy=data_generation_strategy_train
            )
            malaga_testset = MalagaDataset(
                self.opt,
                mode=TrainingDataset.TEST,
                sample_generation_strategy=data_generation_strategy_test
            )
            self.dataset['train'] = malaga_trainset
            self.dataset['test'] = malaga_testset


        self.dataset['train'].print_info(show_sequence=False)
        print('Number of samples:', len(self.dataset['train']))

        self.trainloader = DataLoader(self.dataset['train'], batch_size=self.opt['batch_size'],
                                      shuffle=True, num_workers=self.opt['num_workers'], drop_last=True)

        self.dataset['test'].print_info(show_sequence=False)
        print('Number of samples:', len(self.dataset['test']))

        if self.opt['is_train']:
            self.init_summary_writer(task='UAVO', dataset_name=self.opt['train_set_name'])

    def train(self):

        print("[*] Training starts...")

        if self.opt['continue_train']:
            total_steps = self.opt['start_epoch'] * len(self.trainloader) * self.opt['batch_size']
        else:
            total_steps = 0

        for epoch in range(self.opt['start_epoch'], self.opt['max_epochs']):

            epoch_iter = 0
            for i, data in enumerate(self.trainloader, 0):

                total_steps += self.opt['batch_size']
                epoch_iter += self.opt['batch_size']
                inputs, labels = data
                if self.opt['gpu_id'] >= 0 and torch.cuda.is_available():
                    input = input.cuda()
                    labels = labels.cuda()
                self.model.optimize(epoch=epoch, inputs=inputs, labels=labels)

            if epoch % self.opt['test_epoch_freq'] == 0 and epoch != 0:
                print('testing the model at epoch %d, iters %d' %
                      (epoch, total_steps))
                self.test(epoch, stochastic_eval=True)
                self.model.scheduler_step()

            self.model.update_learning_rate()

    def test(self, epoch=-1, stochastic_eval=False):

        print("[*] Testing ...")

        if stochastic_eval:
            self.model.eval(dropout_eval=True)
        else:
            self.model.eval(dropout_eval=False)

        test_seqs = self.dataset['test'].generate_test_data_by_sequence()

        vo_prediction_list = []
        vo_gt_list = []
        vo_uncertainty_list = []

        sequences = list(test_seqs.items())
        curr_seq_index = 0
        curr_img = 0

        testloader = DataLoader(self.dataset['test'], batch_size=1,
                                     shuffle=False, num_workers=self.opt['num_workers'])

        for i, data in enumerate(testloader, 0):
            if i % 500 == 0:
                print(i)
            inputs, labels = data

            self.model.test(inputs, stochastic_test=stochastic_eval)
            vo_prediction = self.model.vo_estimate.squeeze(0).cpu().data.numpy()
            vo_gt = labels.squeeze(0).cpu().data.numpy()

            if stochastic_eval:
                vo_uncertainty = self.model.vo_variance.squeeze(0).cpu().numpy()

            if curr_img == sequences[curr_seq_index][1].get_num_label() - (self.opt['timesteps'] - 1):
                vo_prediction_list.append(vo_prediction[-1])
                vo_gt_list.append(vo_gt[-1])
                if stochastic_eval:
                    vo_uncertainty_list.append(vo_uncertainty[-1])

                vo_prediction_list = []
                vo_gt_list = []
                vo_uncertainty_list = []
                curr_img = 0
                curr_seq_index += 1

            else:
                if curr_img == 0:
                    for t in range(self.opt['timesteps']-1):
                        vo_prediction_list.append(vo_prediction[t])
                        vo_gt_list.append(vo_gt[t])
                        if stochastic_eval:
                            vo_uncertainty_list.append(vo_uncertainty[t])
                else:
                    vo_prediction_list.append(vo_prediction[-1])
                    vo_gt_list.append(vo_gt[-1])
                    if stochastic_eval:
                        vo_uncertainty_list.append(vo_uncertainty[-1])
                curr_img += 1





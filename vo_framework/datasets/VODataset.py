import os
from collections import OrderedDict
import numpy as np

from vo_framework.datasets.Dataset import TrainingDataset
from vo_framework.datasets.Sequence import Sequence
from vo_framework.ws_definitions import DATASET_PATH


class VODataset(TrainingDataset):

    def __init__(self, config, mode, sample_generation_strategy, name):
        self.training_seqs = OrderedDict()
        self.test_seqs = OrderedDict()
        self.generation_strategy = sample_generation_strategy
        super(VODataset, self).__init__(config, mode, name)

    def generate_samples(self):

        if self.mode == TrainingDataset.TRAIN:
            self.train_data = self.generation_strategy.generate_samples(self.training_seqs)
        else:
            self.test_data = self.generation_strategy.generate_samples(self.test_seqs)

    def print_info(self, show_sequence=False):
        print('--------------------------------------')
        print('------Dataset Info--------')
        print('Dataset Name: {}'.format(self.name))
        if self.mode == TrainingDataset.TRAIN:
            print('Number of Training dirs: {}'.format(len(self.training_seqs)))
            print('Training dirs:')
            for directory in self.training_seqs:
                curr_sequence = self.training_seqs[directory]
                print(directory,
                      curr_sequence.sequence_dir,
                      'Num imgs: {}'.format(curr_sequence.get_num_imgs()),
                      'Num label: {}'.format(curr_sequence.get_num_label()))
                if show_sequence:
                    curr_sequence.visualize_sequence()
        else:
            print('Number of Test dirs: {}'.format(len(self.test_seqs)))
            print('Test dirs:')
            for directory in self.test_seqs:
                curr_sequence = self.test_seqs[directory]
                print(directory,
                      curr_sequence.sequence_dir,
                      'Num imgs: {}'.format(curr_sequence.get_num_imgs()),
                      'Num label: {}'.format(curr_sequence.get_num_label()))
                if show_sequence:
                    curr_sequence.visualize_sequence()

    def get_seq_stat(self):
        num_tr_seq = len(self.training_seqs)
        num_te_seq = len(self.test_seqs)
        num_tr_imgs = 0
        num_te_imgs = 0
        for seq in self.training_seqs:
            num_tr_imgs += self.training_seqs[seq].get_num_imgs()
        for seq in self.test_seqs:
            num_te_imgs += self.test_seqs[seq].get_num_imgs()
        return num_tr_seq, num_te_seq, num_tr_imgs, num_te_imgs

    def show_trajectories(self):
        for sequence in self.training_seqs:
            self.training_seqs[sequence].show_sequence_trajectory()
        for sequence in self.test_seqs:
            self.test_seqs[sequence].show_sequence_trajectory()

    def generate_test_data_by_sequence(self):
        for sequence in self.test_seqs:
            curr_generated_samples = self.generation_strategy.get_sample_set(self.test_seqs[sequence])

            self.test_seqs[sequence].set_generated_sample(curr_generated_samples)
            print(np.shape(self.test_seqs[sequence].generated_sample))
        return self.test_seqs


class KittiDataset(VODataset):

    def __init__(self, config, mode, sample_generation_strategy):

        self._dataset_path = os.path.join(DATASET_PATH, 'KITTI_RGB')

        super(KittiDataset, self).__init__(config, mode, sample_generation_strategy, 'Kitti')

    @property
    def dataset_path(self):
        return self._dataset_path

    def read_data(self):
        if self.config['use_subsampled']:
            subdir = 'image_0/downsampled_' + str(self.config['input_height']) + '_' + str(self.config['input_width'])
        else:
            subdir = 'image_0'

        if self.mode == TrainingDataset.TRAIN:
            for dir in self.config['kitti_train_dirs']:
                seq_dir = os.path.join(self._dataset_path, dir)
                self.training_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                                   extension='png',
                                                   label_file=os.path.join(seq_dir, dir + '.txt'),
                                                   dir=dir,
                                                   is_grayscale=False,
                                                   name='Kitti_train/' + dir
                                                   )

        else:
            for dir in self.config['kitti_all_dirs']:
                seq_dir = os.path.join(self._dataset_path, dir)
                self.test_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                               extension='png',
                                               label_file=os.path.join(seq_dir, dir + '.txt'),
                                               dir=dir,
                                               is_grayscale=False,
                                               name='Kitti_test/' + dir
                                               )

        self.generate_samples()


class MalagaDataset(VODataset):

    def __init__(self, config, mode, sample_generation_strategy):

        self._dataset_path = os.path.join(DATASET_PATH, 'Malaga')

        super(MalagaDataset, self).__init__(config, mode, sample_generation_strategy, 'Malaga')

    @property
    def dataset_path(self):
        return self._dataset_path

    def read_data(self):
        if self.config['use_subsampled']:
            subdir = 'image_0/downsampled_' + str(self.config['input_height']) + '_' + str(self.config['input_width'])
        else:
            subdir = 'image_0'

        if self.mode == TrainingDataset.TRAIN:
            for dir in self.config['malaga_train_dirs']:
                seq_dir = os.path.join(self._dataset_path, dir)
                self.training_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                                   extension='jpg',
                                                   label_file=os.path.join(seq_dir, dir + '.txt'),
                                                   dir=dir,
                                                   is_grayscale=False,
                                                   name='Malaga_train/' + dir
                                                   )
        else:
            for dir in self.config['malaga_all_dirs']:
                seq_dir = os.path.join(self._dataset_path, dir)
                self.test_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                               extension='jpg',
                                               label_file=os.path.join(seq_dir, dir + '.txt'),
                                               dir=dir,
                                               is_grayscale=False,
                                               name='Malaga_test/' + dir
                                               )

        self.generate_samples()


class ISARLABCARDataset(VODataset):

    def __init__(self, config, mode, sample_generation_strategy):

        self._dataset_path = os.path.join(DATASET_PATH, 'ISARLAB_CAR')

        super(ISARLABCARDataset, self).__init__(config, mode, sample_generation_strategy, 'ISARLAB_CAR')

    @property
    def dataset_path(self):
        return self._dataset_path

    def read_data(self):
        if self.config['use_subsampled']:
            subdir = 'image_0/downsampled_' + str(self.config['input_height']) + '_' + str(self.config['input_width'])
        else:
            subdir = 'image_0'

        if self.mode == TrainingDataset.TRAIN:

            for dir in self.config['isarlab_car_train_dirs']:
                seq_dir = os.path.join(self._dataset_path, dir)
                self.training_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                                   extension='png',
                                                   label_file=os.path.join(seq_dir, dir + '.txt'),
                                                   dir=dir,
                                                   is_grayscale=False,
                                                   name='ISARLAB_CAR_train/' + dir
                                                   )
        else:
            for dir in self.config['isarlab_car_all_dirs']:
                seq_dir = os.path.join(self._dataset_path, dir)
                self.test_seqs[dir] = Sequence(os.path.join(seq_dir, subdir),
                                               extension='png',
                                               label_file=os.path.join(seq_dir, dir + '.txt'),
                                               dir=dir,
                                               is_grayscale=False,
                                               name='ISARLAB_CAR_test/' + dir
                                               )

        self.generate_samples()

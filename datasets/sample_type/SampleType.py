import numpy as np
import cv2
from pyquaternion import Quaternion
import torch
from itertools import tee, islice

from vo_framework.datasets.sample_type.AbstractSample import AbstractSample


class ImageSequenceQuat(AbstractSample):
    def __init__(self, set, labels, is_grayscale = False, ram_pre_loading=False, transforms=None):

        self.img_paths = set
        self.relative_transform = labels
        self.is_grayscale = is_grayscale

        super(ImageSequenceQuat, self).__init__(ram_pre_loading, transforms)
        if self.ram_pre_loading:
            self.features = self.init_features()

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return zip(*iters)

    def read_features(self):
        if not self.ram_pre_loading:

            img_sequence = torch.Tensor()

            for img_file_first, img_file_second in self.nwise(self.img_paths, 2):
                if self.is_grayscale:
                    curr_img_first = cv2.imread(img_file_first, cv2.IMREAD_GRAYSCALE)
                    curr_img_second = cv2.imread(img_file_second, cv2.IMREAD_GRAYSCALE)

                else:
                    curr_img_first = cv2.imread(img_file_first, cv2.IMREAD_COLOR)
                    curr_img_second = cv2.imread(img_file_second, cv2.IMREAD_COLOR)

                curr_img_first = torch.from_numpy(curr_img_first).permute(2, 0, 1).float()/255
                curr_img_second = torch.from_numpy(curr_img_second).permute(2, 0, 1).float()/255
                if self.transforms:
                    for transform in self.transforms:
                        curr_img_first = transform(curr_img_first)
                        curr_img_second = transform(curr_img_second)

                img_pair = torch.cat((curr_img_first, curr_img_second), dim=0)
                img_sequence = torch.cat((img_sequence, img_pair.unsqueeze(0)), dim=0)
            return img_sequence

        return self.features

    def init_features(self):
        img_sequence = []
        for img_file_first, img_file_second in self.nwise(self.img_paths, 2):
            if self.is_grayscale:
                curr_img_first = cv2.imread(img_file_first, cv2.IMREAD_GRAYSCALE)
                curr_img_second = cv2.imread(img_file_second, cv2.IMREAD_GRAYSCALE)

            else:
                curr_img_first = cv2.imread(img_file_first, cv2.IMREAD_COLOR)
                curr_img_second = cv2.imread(img_file_second, cv2.IMREAD_COLOR)

            if self.transforms:
                for transform in self.transforms:
                    curr_img_first = transform(curr_img_first)
                    curr_img_second = transform(curr_img_second)

            img_pair = torch.cat((curr_img_first, curr_img_second), dim=0)
            img_sequence = torch.cat((img_sequence, img_pair.unsqueeze(0)), dim=0)

        return img_sequence

    def read_labels(self):

        label_sequence = torch.Tensor()

        for pose in self.relative_transform:

            quaternion = Quaternion(matrix=pose)
            new_label = [quaternion.w,
                         quaternion.x,
                         quaternion.y,
                         quaternion.z,
                         pose[0, 3],
                         pose[1, 3],
                         pose[2, 3]
                         ]
            new_label = torch.from_numpy(np.asarray(new_label)).float()
            label_sequence = torch.cat((label_sequence, new_label.unsqueeze(0)), dim=0)

        return label_sequence


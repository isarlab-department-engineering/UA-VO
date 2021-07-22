import os
from glob import glob
import numpy as np
from vo_framework.utils import abs2rel


class Sequence(object):

    def __init__(self, directory, extension, label_file, dir, is_grayscale=False, name='Sequence'):

        self.sequence_name = name
        self.sequence_dir = directory
        self.is_grayscale = is_grayscale
        self.image_paths = []
        self.dir = dir
        self.labels = []
        self.generated_sample = []
        self.extension = extension
        self.load_img_paths(extension)
        if label_file:
            self.load_label(label_file)

    def load_img_paths(self, extension):
        print(self.sequence_dir)
        self.image_paths = sorted(glob(os.path.join(self.sequence_dir, '*' + '.' + extension)))

    def load_label(self, label_file):
        abs_labels = np.loadtxt(label_file)
        abs_labels = abs_labels.reshape((abs_labels.shape[0], 3, 4))
        self.labels = abs2rel(abs_labels)

    def get_num_imgs(self):
        return len(self.image_paths)

    def get_num_label(self):
        return len(self.labels)

    def get_image_paths(self):
        return self.image_paths

    def get_dir(self):
        return self.dir

    def get_labels(self):
        return self.labels

    def get_is_grayscale(self):
        return self.is_grayscale

    def set_generated_sample(self, generated_sample):
        self.generated_sample = generated_sample

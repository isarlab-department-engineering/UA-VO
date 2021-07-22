from itertools import tee, islice

import numpy as np


class SampleGenerationStrategy(object):

    def __init__(self, ram_pre_loading, transforms=None):
        self.ram_pre_loading = ram_pre_loading
        self.transforms = transforms

    def generate_samples(self, sequences):

        img_sets = []
        for seq in sequences:
            print("Generating samples for sequence: ", seq)
            curr_seq = sequences[seq]
            seq_set = self.get_sample_set(curr_seq)
            img_sets = np.append(img_sets, seq_set)

        return img_sets

    def get_sample_set(self, sequence):

        raise ("Not implemented - this is an abstract method")

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return zip(*iters)


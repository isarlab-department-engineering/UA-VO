from vo_framework.datasets.DataGenerationStrategy import SampleGenerationStrategy

from datasets.sample_type.SampleType import ImageSequenceQuat


class SequenceGenerationStrategy(SampleGenerationStrategy):
    def __init__(self, num_steps, ram_pre_loading=False, transforms=None):

        self.num_steps = num_steps

        super(SequenceGenerationStrategy, self).__init__(ram_pre_loading, transforms)

    def get_sample_set(self, sequence):

        sequences = []
        for img_set, label_set in zip(self.nwise(sequence.get_image_paths(), self.num_steps), self.nwise(sequence.get_labels(), self.num_steps-1)):
            sequences.append(ImageSequenceQuat(img_set, label_set, sequence.get_is_grayscale(), self.ram_pre_loading, transforms=self.transforms))
        return sequences




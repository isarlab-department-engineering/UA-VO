

class AbstractSample(object):

    def __init__(self, ram_pre_loading, transforms=None):
        self.ram_pre_loading = ram_pre_loading
        self.transforms = transforms

    def read_features(self):

        raise ("Not implemented - this is an abstract method")

    def read_labels(self):

        raise ("Not implemented - this is an abstract method")
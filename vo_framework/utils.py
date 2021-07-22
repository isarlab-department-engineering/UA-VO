import numpy as np
import numpy.matlib as npmat
from scipy.linalg import inv
import yaml
import scipy.io as sio

loadmat = sio.loadmat
#

def abs2rel(T):

    data_size = T.shape[0]

    Tl = []

    T_o = np.zeros((data_size - 1, 4, 4))
    for k in range(1, data_size):
        if T.shape[1] == 3:
            Tn = np.concatenate((T[k - 1], npmat.mat([0, 0, 0, 1])), 0)
            Tn1 = np.concatenate((T[k], npmat.mat([0, 0, 0, 1])), 0)
        else:
            Tn = T[k - 1]
            Tn1 = T[k]
        Tl.append((inv(Tn)).dot(Tn1))
        T_o[k - 1, :, :] = Tl[k - 1]
    return T_o


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


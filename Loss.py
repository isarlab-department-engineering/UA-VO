import torch
from torch.nn import Module

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "No gradient should be computed w.r.t this variable"


class SequenceMVELoss(Module):

    def __init__(self, size_average=True):
        super(SequenceMVELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, s2, target):
        _assert_no_grad(target)

        loss = 0.0
        for t in range(input.shape[1]):
            quad_term = ((target[:, t, :] - input[:, t, :]) ** 2) * torch.exp(-s2[:, t, :])
            log_term_trasl = s2[:, t, :]
            loss += (torch.sum(1 / 2 * (quad_term + log_term_trasl))) / (input.shape[0])


        return loss

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import os
from vo_framework import ROOT_PATH_VO
from vo_framework.models.flownet.FlowNet import FlowNet

from Loss import SequenceMVELoss


class UAVOModel:

    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt['is_train']
        self.save_dir = '.'

        self.netFeatureExtractionNet = self.define_Flownet(
            flownet_weight_path=os.path.join(ROOT_PATH_VO, 'models/flownet/flownets_pytorch.pth'))
        self.netRecurrentNet = self.define_RecurrentNet()
        self.netPredictionNet = self.define_PredictionNet()
        self.netStdNet = self.define_StdNet()

        self.networks = [self.netFeatureExtractionNet, self.netRecurrentNet, self.netPredictionNet, self.netStdNet]

        if self.is_train:
            self.lossVO = SequenceMVELoss()

            self.optimizer_rec = torch.optim.SGD(filter(lambda p: p.requires_grad, self.netRecurrentNet.parameters()),
                                                 lr=opt['init_lr'])
            self.optimizer_pred = torch.optim.SGD(filter(lambda p: p.requires_grad, self.netPredictionNet.parameters()),
                                                  lr=opt['init_lr_mean'])
            self.optimizer_std = torch.optim.SGD(filter(lambda p: p.requires_grad, self.netStdNet.parameters()),
                                                 lr=opt['init_lr_std'])

        if self.opt['gpu_id'] >= 0 and torch.cuda.is_available():
            self.netFeatureExtractionNet.cuda()
            self.netRecurrentNet.cuda()
            self.netStdNet.cuda()
            self.netPredictionNet.cuda()

        self.schedulers = []
        self.schedulers.append(lr_scheduler.StepLR(self.optimizer_rec, step_size=opt['lr_decay_iters'], gamma=opt['lr_step_gamma']))
        self.schedulers.append(lr_scheduler.StepLR(self.optimizer_pred, step_size=opt['lr_decay_iters'], gamma=opt['lr_step_gamma']))
        self.schedulers.append(lr_scheduler.StepLR(self.optimizer_std, step_size=opt['lr_decay_iters'], gamma=opt['lr_step_gamma']))

    @staticmethod
    def set_networks_requires_grad(net, grad=False):
        for param in net.parameters():
            param.requires_grad = grad

    def scheduler_step(self, iter=None):
        for scheduler in self.schedulers:
            scheduler.step(epoch=iter)

    def eval(self, dropout_eval=False):
        for net in self.networks:
            if dropout_eval:
                for module in net.modules():
                    if isinstance(module, torch.nn.modules.dropout.Dropout) or isinstance(module,
                                                                                          torch.nn.modules.dropout.Dropout2d) or isinstance(
                            module, torch.nn.modules.LSTM):
                        module.train()
                    else:
                        module.eval()
            else:
                net.eval()

    def set_train_mode(self):
        for net in self.networks:
            net.train()

    def define_Flownet(self, flownet_weight_path):

        return FlowNet(flownet_weight_path)

    def define_RecurrentNet(self):

        recurrent_net = nn.Module()

        recurrent_net.conv_response_reduction = nn.Sequential(

            nn.Conv2d(1024, 64, kernel_size=(1, 1)),
            nn.Dropout2d(p=self.opt['drop_prob']),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(6, 20))
        )
        recurrent_net.lstm = nn.LSTM(20 * 6 * 64, 1000, 2, batch_first=True, dropout=self.opt['drop_prob'])
        recurrent_net.lstm_dropout = nn.Dropout(p=self.opt['drop_prob'])
        return recurrent_net

    def define_PredictionNet(self):

        prediction_net = nn.Module()
        prediction_net.pose_predictions = nn.Sequential(
            nn.Linear(1000, 7)
        )
        return prediction_net

    def define_StdNet(self):

        std_net = nn.Module()
        std_net.std_predictions = nn.Sequential(
            nn.Linear(1000, 7)
        )
        return std_net

    def forward(self, input):
        cnn_input = input.view(input.shape[0] * (self.opt['timesteps']-1), 6, self.opt['input_height'], self.opt['input_width'])
        out_conv6 = self.netFeatureExtractionNet.forward_encoder(cnn_input)
        features = self.netRecurrentNet.conv_response_reduction(out_conv6)
        r_in = features.view(input.shape[0], (self.opt['timesteps']-1), 64*20*6)
        lstm_out, (h_in, c_in) = self.netRecurrentNet.lstm(r_in)
        lstm_out = self.netRecurrentNet.lstm_dropout(lstm_out)
        x = self.netPredictionNet.pose_predictions(lstm_out)
        std = self.netStdNet.std_predictions(lstm_out)
        norm = x[:, :, 0:4].norm(p=2, dim=2, keepdim=True)
        x_normalized = x[:, :, 0:4].div(norm)
        x_out = x.clone()
        x_out[:, :, 0:4] = x_normalized
        self.vo_estimate = x_out
        self.s2_estimate = std

    def optimize(self, epoch, inputs, labels):

        UAVOModel.set_networks_requires_grad(self.netRecurrentNet, True)
        UAVOModel.set_networks_requires_grad(self.netPredictionNet, True)
        if epoch < 5:
            UAVOModel.set_networks_requires_grad(self.netStdNet, False)
        else:
            UAVOModel.set_networks_requires_grad(self.netStdNet, True)

        self.optimizer_rec.zero_grad()
        self.optimizer_pred.zero_grad()
        self.optimizer_std.zero_grad()

        self.forward(inputs)
        loss_VO = self.lossVO(self.vo_estimate, self.s2_estimate, labels)
        print('Loss VO: ', loss_VO)
        loss_VO.backward(retain_graph=False)
        self.loss_VO = loss_VO

        torch.nn.utils.clip_grad_norm_(self.netRecurrentNet.parameters(), 50.0)
        torch.nn.utils.clip_grad_norm_(self.netStdNet.parameters(), 50.0)
        torch.nn.utils.clip_grad_norm_(self.netPredictionNet.parameters(), 50.0)
        self.optimizer_rec.step()
        self.optimizer_pred.step()
        self.optimizer_std.step()

    def deterministic_eval_forward(self, inputs):
        cnn_input = inputs.view(inputs.shape[0] * (self.opt['timesteps'] - 1), 6, self.opt['input_height'],
                                    self.opt['input_width'])
        features = self.netFeatureExtractionNet.forward_encoder(cnn_input)
        features = self.netRecurrentNet.conv_response_reduction(features)

        return features

    def stochastic_eval_forward(self, inputs, features):

        r_in = features.view(inputs.shape[0], (self.opt['timesteps'] - 1), 64 * 20 * 6)
        lstm_out, (h_in, c_in) = self.netRecurrentNet.lstm(r_in)
        lstm_out = self.netRecurrentNet.lstm_dropout(lstm_out)

        x = self.netPredictionNet.pose_predictions(lstm_out)
        std = self.netStdNet.std_predictions(lstm_out)
        norm = x[:, :, 0:4].norm(p=2, dim=2, keepdim=True)
        x_normalized = x[:, :, 0:4].div(norm)
        x_out = x.clone()
        x_out[:, :, 0:4] = x_normalized
        self.vo_estimate = x_out
        self.s2_estimate = std

    def test(self, inputs, stochastic_test=False):
        with torch.no_grad():
            if stochastic_test:
                x = self.deterministic_eval_forward(inputs)
                samples = torch.zeros((self.opt['stochastic_sample'], self.opt['timesteps'] - 1, 7))
                samples_std = torch.zeros((self.opt['stochastic_sample'], self.opt['timesteps'] - 1, 7))

                for t in range(self.opt['stochastic_sample']):
                    self.stochastic_eval_forward(inputs, x)
                    vo_prediction = self.vo_estimate.squeeze(0)
                    std_prediction = torch.exp(self.s2_estimate.squeeze(0))
                    samples[t, :, :] = vo_prediction
                    samples_std[t, :, :] = std_prediction

                mean = samples.mean(dim=0)
                mean_std = samples_std.mean(0)
                mean_std_diag = torch.zeros(mean.shape[0], mean.shape[1], mean.shape[1])
                mean_std_diag.as_strided(mean_std.size(), [mean_std_diag.stride(0), mean_std_diag.size(2) + 1]).copy_(mean_std)
                variance = sum(torch.bmm((mean-sample).unsqueeze(2), (mean-sample).unsqueeze(1)) for sample in samples)/samples.shape[0]
                variance_tot = variance + mean_std_diag
                self.vo_prediction_mean = mean
                self.vo_variance = variance_tot
            else:
                self.forward(inputs)




import os
import math

import torch
import torch.nn as nn
import torch.utils.data

"""
Pytorch re-implementation of Pelletier et al. 2019
https://github.com/charlotte-pel/temporalCNN
https://www.mdpi.com/2072-4292/11/5/523

From 
https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TempCNN.py

Transformed fo multilabel/multi target segmentation.

"""

__all__ = ['TempCNN']

class TempCNN(torch.nn.Module):
    def __init__(self, input_dim, n_type, n_sev, n_date, sequencelength, kernel_size=9, hidden_dims=16, dropout=0.4):
        super(TempCNN, self).__init__()
        self.modelname = f"TempCNN_input-dim={input_dim}_n_type={n_type}_n_sev={n_sev}_n_date={n_date}_sequencelenght={sequencelength}_" \
                         f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(input_dim, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(hidden_dims * sequencelength, 4 * hidden_dims, drop_probability=dropout)
        # self.logsoftmax = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes), nn.LogSoftmax(dim=-1))
        # self.sig = nn.Sequential(nn.Linear(4 * hidden_dims, num_classes), nn.Sigmoid())

        self.type_ = nn.Sequential(nn.Linear(4 * hidden_dims, n_type))
        self.sev_ = nn.Sequential(nn.Linear(4 * hidden_dims, n_sev))
        self.date_ = nn.Sequential(nn.Linear(4 * hidden_dims, n_date))



        # self.initialize_weights()

    def forward(self, x):
        # require NxTxD
        x = x.transpose(1,2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        #print('last', size(x))

        x = self.flatten(x)
        x = self.dense(x)
        # last= self.logsoftmax(x)
        # last= self.sig(x)
        #print('softmax', size(last))



        return {
            'type': self.type_(x),
            'sev': self.sev_(x),
            'date': self.date_(x)
        }

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

    def initialize_weights(self):
        # see also https://github.com/pytorch/pytorch/issues/18182
        for m in self.modules():
            if type(m) in {
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)



class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size=7, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

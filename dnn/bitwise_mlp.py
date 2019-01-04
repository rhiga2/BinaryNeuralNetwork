import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dnn.binary_layers import *

class BitwiseMLP(nn.Module):
    def __init__(self, in_size, out_size, fc_sizes=[], dropout=0,
        sparsity=95, temp=1, use_gate=False, output_activation=torch.tanh,
        use_batchnorm=True):
        super(BitwiseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_gate = use_gate
        self.temp = nn.Parameter(torch.tensor(temp, dtype=torch.float), requires_grad=False)
        self.activation = torch.tanh
        self.use_noise = use_noise
        self.use_batchnorm = use_batchnorm

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [out_size,]
        isize = in_size
        self.filter_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for i, osize in enumerate(fc_sizes):
            self.filter_list.append(BitwiseLinear(isize, osize,
                use_gate=use_gate, activation=self.activation,
                use_noise=use_noise))
            if use_batchnorm:
                self.bn_list.append(nn.BatchNorm1d(osize))
            if i < self.num_layers - 1:
                self.dropout_list.append(nn.Dropout(dropout))
            isize = osize

        self.output_activation = output_activation
        self.sparsity = sparsity
        self.mode = 'real'

    def forward(self, x):
        '''
        Bitwise neural network forward
        * Input is a tensor of shape (batch, channels)
        * Output is a tensor of shape (batch, channels)
            - batch is the batch size
            - time is the sequence length
            - channels is the number of input channels = num bits in qad
        '''
        for i in range(self.num_layers):
            x = self.filter_list[i](x)
            if i < self.num_layers - 1:
                if self.use_batchnorm:
                    x = self.bn_list[i](x)
                if self.use_noise and self.mode != 'inference':
                    x = add_logistic_noise(x)
                x = self.activation(x)
                x = self.dropout_list[i](x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

    def noisy(self):
        '''
        Converts real network to noisy training network
        '''
        self.mode = 'noisy'
        self.activation = bitwise_activation
        for layer in self.filter_list:
            layer.noisy()

    def inference(self):
        '''
        Converts noisy training network to bitwise network
        '''
        self.mode = 'inference'
        self.activation = bitwise_activation(x)
        for layer in self.filter_list:
            layer.inference()

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.mode != 'noisy' or self.use_gate:
            return

        for layer in self.filter_list:
            layer.update_beta(sparsity=self.sparsity)

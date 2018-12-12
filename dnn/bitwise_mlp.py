import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dnn.binary_layers import *

class BitwiseMLP(nn.Module):
    def __init__(self, in_size, out_size, fc_sizes=[], dropout=0,
        sparsity=95, temp=1, use_gate=False, use_noise=False):
        super(BitwiseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_gate = use_gate
        self.temp = nn.Parameter(torch.tensor(temp, dtype=torch.float), requires_grad=False)
        self.activation = lambda x : squeezed_tanh(x, temp, noise=use_noise)
        self.use_noise = use_noise

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [out_size,]
        isize = in_size
        self.filter_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for i, osize in enumerate(fc_sizes):
            self.filter_list.append(BitwiseLinear(isize, osize,
                use_gate=use_gate, activation=self.activation))
            self.bn_list.append(nn.BatchNorm1d(osize))
            if i < self.num_layers - 1:
                self.dropout_list.append(nn.Dropout(dropout))
            isize = osize

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
            x = self.bn_list[i](x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout_list[i](x)
        return x

    def noisy(self):
        '''
        Converts real network to noisy training network
        '''
        self.mode = 'noisy'
        self.activation = lambda x : bitwise_activation(x, use_noise=self.use_noise)
        for layer in self.filter_list:
            layer.noisy()

    def inference(self):
        '''
        Converts noisy training network to bitwise network
        '''
        self.mode = 'inference'
        self.activation = lambda x : bitwise_activation(x, use_noise=self.use_noise)
        for layer in self.filter_list:
            layer.inference()
        for bn in self.bn_list:
            sign_weight = torch.sign(bn.weight)
            bias = -sign_weight * bn.running_mean
            bias += bn.bias * bn.running_var / torch.abs(bn.weight)
            bn.bias = nn.Parameter(bias, requires_grad=False)
            bn.weight = nn.Parameter(sign_weight, requires_grad=False)
            bn.running_var = torch.ones_like(running_var)
            bn_running_mean = torch.zeros_like(running_mean)

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.mode != 'noisy' or self.use_gate:
            return

        for layer in self.filter_list:
            layer.update_beta(sparsity=self.sparsity)

    def update_temp(self, temp):
        if self.mode != 'real':
            return

        self.temp = nn.Parameter(torch.tensor(temp, dtype=self.temp.dtype, device=self.temp.device), requires_grad=False)
        self.activation = lambda x : squeezed_tanh(x, temp, noise=self.use_noise)
        for layer in self.filter_list:
            layer.activation = self.activation

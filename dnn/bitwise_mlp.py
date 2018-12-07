import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dnn.binary_layers import *

class BitwiseMLP(nn.Module):
    def __init__(self, in_size, out_size, fc_sizes=[], dropout=0,
        sparsity=95, gamma=1, use_gate=False):
        super(BitwiseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_gate = use_gate
        self.gamma = gamma
        self.activation = lambda x: squeezed_tanh(x, gamma)

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [out_size,]
        input_size = in_size
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for i, layer_size in enumerate(fc_sizes):
            self.linear_list.append(BitwiseLinear(input_size, layer_size, use_gate=use_gate))
            input_size = layer_size
            self.bn_list.append(nn.BatchNorm1d(layer_size))
            if i < self.num_layers - 1:
                self.dropout_list.append(nn.Dropout(dropout))

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
            x = self.linear_list[i](x)
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
        self.activation = bitwise_activation
        for layer in self.linear_list:
            layer.noisy()

    def inference(self):
        '''
        Converts noisy training network to bitwise network
        '''
        self.mode = 'inference'
        self.activation = bitwise_activation
        for layer in self.linear_list:
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

        for layer in self.linear_list:
            layer.update_beta(sparsity=self.sparsity)

    def update_gamma(self, gamma):
        self.gamma = gamma
        for layer in linear_list:
            layer.gamma = gamma
        self.activation = lambda x: squeezed_tanh(x, gamma)

import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dnn.binary_layers as binary_layers

class BitwiseMLP(nn.Module):
    def __init__(self, in_size, out_size, fc_sizes=[], dropout=0,
        sparsity=0, use_gate=False, activation=torch.tanh,
        weight_activation=torch.tanh, use_batchnorm=True, bn_momentum=0.1):
        super(BitwiseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_gate = use_gate
        self.activation = activation
        self.weight_activation = weight_activation
        self.use_batchnorm = use_batchnorm

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [out_size,]
        isize = in_size
        self.filter_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for i, osize in enumerate(fc_sizes):
            self.filter_list.append(
                binary_layers.BitwiseLinear(isize, osize,
                    use_gate=use_gate, activation=weight_activation)
            )
            if use_batchnorm:
                self.bn_list.append(nn.BatchNorm1d(osize, momentum=bn_momentum))
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
            if self.use_batchnorm:
                x = self.bn_list[i](x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout_list[i](x)
        return x

    def clip_weights(self):
        for layer in self.filter_list:
            layer.clip_weights()

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.sparsity == 0:
            return

        for layer in self.filter_list:
            layer.update_beta(sparsity=self.sparsity)

import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dnn.binary_layers as binary_layers

def flatten(x):
    batch, channels, time = x.size()
    x = x.permute(0, 2, 1).contiguous().view(-1, channels)
    return x

def unflatten(x, batch, time, permutation=(0, 2, 1)):
    x = x.view(batch, time, -1)
    x = x.permute(*permutation).contiguous()
    return x

class BitwiseMLP(nn.Module):
    def __init__(self, in_size, out_size, fc_sizes=[], dropout=0,
        sparsity=0, use_gate=False, binactiv=None, bn_momentum=0.1,
        scale_weights=False, bias=True, num_binarizations=1):
        super(BitwiseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_gate = use_gate
        self.activation = activation

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [out_size,]
        isize = in_size
        self.filter_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout = dropout
        if dropout > 0:
            self.dropout_list = nn.ModuleList()
        for i, osize in enumerate(fc_sizes):
            binfunc = binactiv
            if i == 0:
                binfunc = None
            self.filter_list.append(
                binary_layers.BitwiseLinear(isize, osize, use_gate=use_gate,
                binactiv=binfunc, scale_weights=scale_weights, bias=bias,
                num_binarizations=num_binarizations)
            )
            if i < self.num_layers - 1:
                self.activation_list.append(nn.PReLU())
                if dropout > 0:
                    self.dropout_list.append(nn.Dropout(dropout))
                self.bn_list.append(nn.BatchNorm1d(osize, momentum=bn_momentum))
            isize = osize
        self.sparsity = sparsity

    def forward(self, x):
        '''
        Bitwise neural network forward
        * Input is a tensor of shape (batch, channels)
        * Output is a tensor of shape (batch, channels)
            - batch is the batch size
            - time is the sequence length
            - channels is the number of input channels = num bits in qad
        '''
        h = x
        for i in range(self.num_layers):
            h = self.filter_list[i](h)
            if i < self.num_layers - 1:
                h = self.activation_list[i](h)
                if self.dropout > 0:
                    h = self.dropout_list[i](h)
                h = self.bn_list[i](h)
        return h

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

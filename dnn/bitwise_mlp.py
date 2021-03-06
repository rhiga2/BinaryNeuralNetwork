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
        sparsity=0, bias=True, use_gate=False, in_binactiv=None,
        w_binactiv=None, bn_momentum=0.1, scale_weights=None,
        scale_activations=None, binarize_first_layer=False):
        super(BitwiseMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [out_size,]
        isize = in_size
        self.filter_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.dropout = dropout
        if dropout > 0:
            self.dropout_list = nn.ModuleList()
        for i, osize in enumerate(fc_sizes):
            if i == 0 and not binarize_first_layer:
                self.filter_list.append(nn.Linear(isize, osize, bias=bias))
            else:
                self.filter_list.append(
                    binary_layers.BitwiseLinear(isize, osize, use_gate=use_gate,
                    in_binactiv=in_binactiv, w_binactiv=w_binactiv,
                    scale_weights=scale_weights, scale_activations=scale_activations, bias=bias)
                )
            if i < self.num_layers - 1:
                if dropout > 0:
                    self.dropout_list.append(nn.Dropout(dropout, inplace=True))
                self.bn_list.append(nn.BatchNorm1d(osize, momentum=bn_momentum))
            isize = osize
        self.out_bn = binary_layers.ScaleLayer(osize)
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
                if self.dropout > 0:
                    h = self.dropout_list[i](h)
                h = self.bn_list[i](h)
        return self.out_bn(h)

    def clip_weights(self):
        for layer in self.filter_list:
            if hasattr(layer, 'bitwise'):
                layer.clip_weights()

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.sparsity == 0:
            return

        for layer in self.filter_list[1:]:
            layer.update_beta(sparsity=self.sparsity)

    def load_partial_state_dict(self, state_dict):
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

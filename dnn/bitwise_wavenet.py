import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import dnn.binary_layers as binary_layers

class BitwiseWaveNetBlock(nn.Module):
    def __init__(self, res_channels, dl_channels, layers=10, kernel_size=2,
        filter_activation=torch.tanh, gate_activation=torch.sigmoid,
        in_bin=binary_layers.identity, weight_bin=binary_layers.identity,
        use_gate=True, adaptive_scaling=True):
        super(WaveNetBlock, self).__init__()
        dilation = 1
        self.layers = layers
        self.filter_activation = filter_activation
        self.gate_activation = gate_activation
        for i in range(layers):
            padding = dilation * (kernel_size-1) // 2
            self.filter_conv = binary_layers.BitwiseConv1d(res_channels,
                dl_channels, kernel_size=kernel_size, dilation=dilation,
                use_gate=use_gate, adaptive_scaling=adaptive_scaling,
                in_bin=in_bin, weight_bin=weight_bin, padding=padding)
            self.gate_conv = binary_layers.BitwiseConv1d(res_channels,
                dl_channels, kernel_size=kernel_size, dilation=dilation,
                use_gate=use_gate, adaptive_scaling=adaptive_scaling,
                in_bin=in_bin, weight_bin=weight_bin, padding=padding)
            self.residual_conv = binary_layers.BitwiseConv1d(dl_channels,
                res_channels, kernel_size=1, use_gate=use_gate,
                adaptive_scaling=adaptive_scaling, in_bin=in_bin,
                weight_bin=weight_bin)
            self.skip_conv = binary_layers.BitwiseConv1d(dl_channels,
                res_channels, kernel_size=1, use_gate=use_gate,
                adaptive_scaling=adaptive_scaling, in_bin=in_bin,
                weight_bin=weight_bin)
            dilation = dilation * 2

    def forward(self, x):
        '''
        x has shape (batch size, residual channels, length)
        '''
        resid = x
        skip = torch.zeros(x.size())
        for i in range(self.layers):
            filtered = self.filter_activation(self.filter_conv(resid))
            gated = self.gate_activation(self.gate_conv(resid))
            new_resid = filtered * gated
            new_resid = self.residual_conv(new_resid)
            skip += self.skip_conv(h)
            resid += new_resid
        return resid, skip

    def clip_weights(self):
        self.filter_conv.clip_weights()
        self.gate_conv.clip_weights()
        self.residual_conv.clip_weights()
        self.skip_conv.clip_weights()

def BitwiseWaveNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=4, layers=10,
        dl_channels=32, res_channels=32, skip_channels=256,
        kernel_size=kernel_size, filter_activation=torch.tanh,
        gate_activation=torch.sigmoid, in_bin=binary_layers.identity,
        weight_bin=binary_layers.identity, adaptive_scaling=True,
        use_gate=True):
        super(WaveNet, self).__init__()
        self.front_conv = binary_layers.BitwiseConv1d(in_channels, res_channels,
            kernel_size=1, in_bin=in_bin, weight_bin=weight_bin,
            use_gate=use_gate, adaptive_scaling=adaptive_scaling)
        self.blocks = blocks
        self.block_list = nn.ModuleList()
        for i in range(blocks):
            self.block_list.append(BitwiseWaveNetBlock(res_channels, dl_channels,
                layers=layers, kernel_size=kernel_size,
                filter_activation=filter_activation,
                gate_activation=gate_activation, use_gate=use_gate,
                adaptive_scaling=adaptive_scaling))
        self.end_conv1 = binary_layers.BitwiseConv1d(res_channels, out_channels,
            kernel_size=1, in_bin=in_bin, weight_bin=weight_bin, use_gate=use_gate,
            adaptive_scaling=adaptive_scaling)
        self.end_conv2 = binary_layers.BitwiseConv1d(res_channels, out_channels,
            out_channels, kernel_size=1, in_bin=in_bin, weight_bin=weight_bin,
            use_gate=use_gate, adaptive_scaling=adaptive_scaling)

    def forward(self, x):
        resid = self.front_conv(x)
        skip = torch.zeros(resid.size())
        for i in range(self.blocks):
            resid, new_skip = self.block_list[i](resid)
            skip += new_skip
        out = F.relu(skip)
        out = F.relu(self.end_conv1(out))
        return self.end_conv2(out)

    def clip_weights(self):
        self.front_conv.clip_weights()
        for i in range(self.blocks):
            self.block_list[i].clip_weights()
        self.end_conv1.clip_weights()
        self.end_conv2.clip_weights()

    def update_betas(self, sparsity):
        return

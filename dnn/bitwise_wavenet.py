import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import dnn.binary_layers as binary_layers

class BitwiseWavenetBlock(nn.Module):
    def __init__(self, res_channels, dl_channels, layers=10, kernel_size=2,
        filter_activation=torch.tanh, gate_activation=torch.sigmoid,
        in_bin=None, weight_bin=None,
        use_gate=True, adaptive_scaling=True, use_batchnorm=False):
        super(BitwiseWavenetBlock, self).__init__()
        dilation = 1
        self.layers = layers
        self.filter_activation = filter_activation
        self.gate_activation = gate_activation
        self.filter_list = nn.ModuleList()
        self.gate_list = nn.ModuleList()
        self.residual_list = nn.ModuleList()
        self.skip_list = nn.ModuleList()
        self.use_batchnorm = use_batchnorm

        if use_batchnorm:
            self.batchnorm_list = nn.ModuleList()

        for i in range(layers):
            padding = dilation * (kernel_size-1) // 2
            if i == 0:
                padding = 1

            self.filter_list.append(binary_layers.BitwiseConv1d(res_channels,
                dl_channels, kernel_size, dilation=dilation,
                use_gate=use_gate, adaptive_scaling=adaptive_scaling,
                in_bin=in_bin, weight_bin=weight_bin, padding=padding))
            self.gate_list.append(binary_layers.BitwiseConv1d(res_channels,
                dl_channels, kernel_size, dilation=dilation,
                use_gate=use_gate, adaptive_scaling=adaptive_scaling,
                in_bin=in_bin, weight_bin=weight_bin, padding=padding))

            if use_batchnorm:
                self.batchnorm_list.append(nn.BatchNorm1d(res_channels))

            self.residual_list.append(binary_layers.BitwiseConv1d(dl_channels,
                res_channels, 1, use_gate=use_gate,
                adaptive_scaling=adaptive_scaling, in_bin=in_bin,
                weight_bin=weight_bin))
            self.skip_list.append(binary_layers.BitwiseConv1d(dl_channels,
                res_channels, 1, use_gate=use_gate,
                adaptive_scaling=adaptive_scaling, in_bin=in_bin,
                weight_bin=weight_bin))
            dilation = dilation * 2

    def forward(self, x):
        '''
        x has shape (batch size, residual channels, length)
        '''
        resid = x
        skip = torch.zeros_like(x)
        for i in range(self.layers):
            filtered = self.filter_list[i](resid)
            gated = self.gate_list[i](resid)
            if i == 0:
                filtered = filtered[:, :, :-1]
                gated = gated[:, :, :-1]

            layer_out = filtered * gated
            if self.use_batchnorm:
                layer_out = self.batchnorm_list[i](layer_out)

            skip = skip + self.skip_list[i](layer_out)
            resid = resid + self.residual_list[i](layer_out)
        return resid, skip

    def clip_weights(self):
        for i in range(self.layers):
            self.filter_list[i].clip_weights()
            self.gate_list[i].clip_weights()
            self.residual_list[i].clip_weights()
            self.skip_list[i].clip_weights()

class BitwiseWavenet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=2, layers=10,
        dl_channels=32, res_channels=32, skip_channels=256,
        kernel_size=2, filter_activation=torch.tanh,
        gate_activation=torch.sigmoid, in_bin=None,
        weight_bin=None, adaptive_scaling=True,
        use_gate=True, use_batchnorm=False):
        super(BitwiseWavenet, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.start_conv = binary_layers.BitwiseConv1d(in_channels, res_channels,
            1, in_bin=in_bin, weight_bin=weight_bin,
            use_gate=use_gate, adaptive_scaling=adaptive_scaling)
        self.blocks = blocks
        self.block_list = nn.ModuleList()
        for i in range(blocks):
            self.block_list.append(BitwiseWavenetBlock(res_channels, dl_channels,
                kernel_size=kernel_size, layers=layers,
                filter_activation=filter_activation,
                gate_activation=gate_activation, use_gate=use_gate,
                adaptive_scaling=adaptive_scaling, use_batchnorm=False))
        self.end_conv1 = binary_layers.BitwiseConv1d(res_channels, out_channels,
            1, in_bin=in_bin, weight_bin=weight_bin, use_gate=use_gate,
            adaptive_scaling=adaptive_scaling)
        self.end_conv2 = binary_layers.BitwiseConv1d(out_channels, out_channels,
            1, in_bin=in_bin, weight_bin=weight_bin,
            use_gate=use_gate, adaptive_scaling=adaptive_scaling)

        if use_batchnorm:
            self.start_bn = nn.BatchNorm1d(res_channels, eps=5e-4)
            self.end_bn1 = nn.BatchNorm1d(res_channels, eps=5e-4)
            self.end_bn2 = nn.BatchNorm1d(out_channels, eps=5e-4)

    def forward(self, x):
        resid = self.start_conv(x)
        if self.use_batchnorm:
            resid = self.start_bn(resid)
        skip = torch.zeros_like(resid)
        for i in range(self.blocks):
            resid, new_skip = self.block_list[i](resid)
            skip = skip + new_skip
        out = F.relu(skip)
        if self.use_batchnorm:
            out = self.end_bn1(out)
        out = F.relu(self.end_conv1(out))
        if self.use_batchnorm:
            out = self.end_bn2(out)
        return self.end_conv2(out)

    def clip_weights(self):
        self.front_conv.clip_weights()
        for i in range(self.blocks):
            self.block_list[i].clip_weights()
        self.end_conv1.clip_weights()
        self.end_conv2.clip_weights()

    def update_betas(self):
        return

    def load_partial_state_dict(self, state_dict):
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import dnn.binary_layers as binary_layers

class BitwiseTasNetBlock(nn.Module):
    def __init__(self, bottleneck_channels, dconv_size, kernel_size=3,
        layers=4, in_bin=None, weight_bin=None, adaptive_scaling=False,
        use_gate=False):
        super(BitwiseTasNetBlock, self).__init__()
        self.layers = layers
        self.first1x1_list = nn.ModuleList()
        self.first_activation = nn.ModuleList()
        self.first_normalization = nn.ModuleList()
        self.dconvs = nn.ModuleList()
        self.second_activation = nn.ModuleList()
        self.second_normalization = nn.ModuleList()
        self.last1x1_list = nn.ModuleList()
        dilation = 1
        for i in range(layers):
            # 1x1 Conv
            self.first1x1_list.append(
                binary_layers.BitwiseConv1d(
                    bottleneck_channels, dconv_size, 1,
                    use_gate=use_gate, adaptive_scaling=adaptive_scaling,
                    in_bin=in_bin, weight_bin=weight_bin
                )
            )

            self.first_activation.append(nn.PReLU())
            self.first_normalization.append(nn.BatchNorm1d(dconv_size))
            padding = dilation * (kernel_size - 1) // 2
            self.dconvs.append(
                binary_layers.BitwiseConv1d(
                    dconv_size, dconv_size, kernel_size, groups=dconv_size,
                    use_gate=use_gate, padding=padding,
                    adaptive_scaling=adaptive_scaling,
                    in_bin=in_bin, weight_bin=weight_bin,
                    dilation = dilation
                )
            )
            self.second_activation.append(nn.PReLU())
            self.second_normalization.append(nn.BatchNorm1d(dconv_size))
            self.last1x1_list.append(
                binary_layers.BitwiseConv1d(
                    dconv_size, bottleneck_channels, 1,
                    use_gate=use_gate, adaptive_scaling=adaptive_scaling,
                    in_bin=in_bin, weight_bin=weight_bin
                )
            )
            dilation *= 2

    def forward(self, x):
        resid = x
        for i in range(self.layers):
            x = self.first1x1_list[i](x)
            x = self.first_activation[i](x)
            x = self.first_normalization[i](x)
            x = self.dconvs[i](x)
            x = self.second_activation[i](x)
            x = self.second_normalization[i](x)
            x = self.last1x1_list[i](x)
        x = x + resid
        return resid

class BitwiseTasNet(nn.Module):
    def __init__(self, in_channels, encoder_channels,
        dconv_size, blocks=2, front_kernel_size=20, front_stride=10,
        kernel_size=3, layers=4, in_bin=None, weight_bin=None,
        adaptive_scaling=False, use_gate=False):
        super(BitwiseTasNet, self).__init__()
        self.in_bin = in_bin
        self.front_kernel_size = front_kernel_size
        self.encoder = binary_layers.BitwiseConv1d(in_channels, encoder_channels,
            front_kernel_size, stride=front_stride, padding=front_kernel_size,
            groups=1, dilation=1, use_gate=False,
            adaptive_scaling=False, in_bin=None,
            weight_bin=None)
        self.block_list = nn.ModuleList()
        self.blocks = blocks
        for i in range(blocks):
            self.block_list.append(
                BitwiseTasNetBlock(
                    encoder_channels, dconv_size, kernel_size=kernel_size,
                    layers=layers, in_bin=in_bin, weight_bin=weight_bin,
                    adaptive_scaling=adaptive_scaling, use_gate=use_gate
                )
            )
        self.decoder = binary_layers.BitwiseConvTranspose1d(encoder_channels, in_channels,
            front_kernel_size, stride=front_stride, padding=0, groups=1,
            dilation=1, use_gate=False,
            adaptive_scaling=False, in_bin=None,
            weight_bin=None
        )

    def forward(self, x):
        time = x.size(2)
        x = self.encoder(x)
        for i in range(self.blocks):
            h = self.block_list[i](h)
        if self.in_bin is not None:
            h = self.in_bin(h)
        else:
            h = torch.sigmoid(h)
        x = x * h
        return self.decoder(x)[:,:,self.front_kernel_size:time+self.front_kernel_size]

    def update_betas(self):
        return

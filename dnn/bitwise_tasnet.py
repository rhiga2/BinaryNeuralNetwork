import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import dnn.binary_layers as binary_layers

class GlobalLayerNormalization(nn.Module):
    def __init__(self, channels, eps=1e-5, momentum=0.1,):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(1, channels, 1))
        self.bias = nn.Parameter(torch.Tensor(1, channels, 1))

        # Initialized statistics as scalar since I don't know what the
        # actual dim will be. I don't want to add batch size as input
        self.running_mean = 0
        self.running_var = 1
        self.reset_parameters()

    def reset_parameters():
        self.weight.data.fill_(1)
        self.bias.data.zero_()
        self.running_mean = 0
        self.running_var = 1

    def forward(self, x):
        '''
        x has shape (batch, channels, time frames)
        '''
        # training should be defined by nn.Module superclass
        if self.training:
            mean = x.mean(1, keepdim=True).mean(2, keepdim=True)
            var = x.var(1, keepdim=True).var(2, keepdim=True)
            self.running_mean = (1 - self.momentum)*self.running_mean \
                + self.momentum*mean
            self.running_var = (1 - self.momentum)*self.running_var \
                + self.momentum*var
            return self.weight*(x - self.mean)/torch.sqrt(var + self.eps) \
                + self.bias
        return self.weight*(x - self.running_mean)/torch.sqrt(self.running_var+self.eps) + self.bias

def get_normalization(normalization, channels, momentum=0.1):
    if normalization == 'batch':
        return nn.BatchNorm1d(channels, momentum=momentum)
    elif normalization == 'global':
        return GlobalLayerNormalization(channel, momentum=momentum)
    else:
        raise Exception('Normalization type is unknown')


class BitwiseTasNetRepeat(nn.Module):
    def __init__(self, bottleneck_channels, dconv_size, kernel_size=3,
        blocks=8, in_binactiv=None, w_binactiv=None,
        use_gate=False, bn_momentum=0.1, normalization='batch'):
        super().__init__()
        self.blocks = blocks
        self.first1x1_list = nn.ModuleList()
        # self.first_activation = nn.ModuleList()
        self.first_normalization = nn.ModuleList()
        self.dconvs = nn.ModuleList()
        self.in_binactiv = in_binactiv
        # self.second_activation = nn.ModuleList()
        self.second_normalization = nn.ModuleList()
        self.third_normalization = nn.ModuleList()
        self.last1x1_list = nn.ModuleList()
        dilation = 1
        for i in range(blocks):
            # 1x1 Conv
            self.first1x1_list.append(
                binary_layers.BitwiseConv1d(
                    bottleneck_channels, dconv_size, 1,
                    use_gate=use_gate, in_binactiv=in_binactiv,
                    w_binactiv=w_binactiv, bias=False
                )
            )

            # self.first_activation.append(nn.PReLU())
            self.first_normalization.append(nn.BatchNorm1d(bottleneck_channels,
                momentum=bn_momentum))
            padding = dilation * (kernel_size - 1) // 2
            self.dconvs.append(
                binary_layers.BitwiseConv1d(
                    dconv_size, dconv_size, kernel_size, groups=dconv_size,
                    use_gate=use_gate, padding=padding,
                    in_binactiv=in_binactiv, w_binactiv=w_binactiv,
                    dilation = dilation, bias=False
                )
            )
            # self.second_activation.append(nn.PReLU())
            self.second_normalization.append(
                get_normalization(normalization, dconv_size,
                    momentum=bn_momentum)
            )
            self.last1x1_list.append(
                binary_layers.BitwiseConv1d(
                    dconv_size, bottleneck_channels, 1,
                    use_gate=use_gate, in_binactiv=in_binactiv,
                    w_binactiv=w_binactiv, bias=False
                )
            )
            self.third_normalization.append(nn.BatchNorm1d(dconv_size,
               momentum=bn_momentum))
            dilation *= 2

    def forward(self, x):
        resid = x
        for i in range(self.blocks):
            h = self.first_normalization[i](resid)
            h = self.first1x1_list[i](h)
            # x = self.first_activation[i](x)
            h = self.second_normalization[i](h)
            h = self.dconvs[i](h)
            # x = self.second_activation[i](x)
            h = self.third_normalization[i](h)
            h = self.last1x1_list[i](h)
            resid = resid + h
        return resid

class BitwiseTasNet(nn.Module):
    def __init__(self, in_channels, encoder_channels,
        dconv_size, repeats=4, front_kernel_size=20, front_stride=10,
        kernel_size=3, blocks=8, in_binactiv=None, w_binactiv=None,
        use_gate=False, bn_momentum=0.1):
        super().__init__()
        self.in_binactiv = in_binactiv
        if in_binactiv is not None:
            self.in_binfunc = in_binactiv()
        self.front_kernel_size = front_kernel_size
        self.activation = nn.PReLU()
        self.encoder = binary_layers.BitwiseConv1d(in_channels, encoder_channels,
            front_kernel_size, stride=front_stride, padding=front_kernel_size,
            groups=1, dilation=1, use_gate=False, bias=False)
        self.block_list = nn.ModuleList()
        self.repeats = repeats
        for i in range(repeats):
            self.block_list.append(
                BitwiseTasNetRepeat(
                    encoder_channels, dconv_size, kernel_size=kernel_size,
                    blocks=blocks, in_binactiv=in_binactiv, w_binactiv=w_binactiv,
                    use_gate=use_gate
                )
            )
        self.out_batchnorm = nn.BatchNorm1d(encoder_channels, momentum=bn_momentum)
        self.decoder = binary_layers.BitwiseConvTranspose1d(encoder_channels,
            in_channels, front_kernel_size, stride=front_stride,
            padding=0, groups=1, dilation=1, use_gate=False,
            bias=False
        )

    def forward(self, x):
        time = x.size(2)
        x = self.activation(self.encoder(x))
        h = x
        for i in range(self.repeats):
            h = self.block_list[i](h)
        h = torch.sigmoid(self.out_batchnorm(h))
        x = x * h
        return self.decoder(x)[:,:,self.front_kernel_size:time+self.front_kernel_size]

    def update_betas(self):
        return

    def load_partial_state_dict(self, state_dict):
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

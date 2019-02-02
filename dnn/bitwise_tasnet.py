import torch
import torch.nn as nn
import dnn.binary_layers as binary_layers

def __init__(self, res_channels, dl_channels, layers=10, kernel_size=2,
    filter_activation=torch.tanh, gate_activation=torch.sigmoid,
    in_bin=binary_layers.identity, weight_bin=binary_layers.identity,
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

class BitwiseTasNetBlock(nn.Module):
    def __init__(self, bottleneck_channels, dconv_size, kernel_size=3,
        layers=4):
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
            padding = dilation * (kernel_size - 1)
            self.dconvs.append(
                binary_layers.BitwiseConv1d(
                    dconv_size, dconv_size, kernel_size, groups=dconv_size,
                    use_gate=use_gate, padding=padding,
                    adaptive_scaling=adaptive_scaling,
                    in_bin=in_bin, weight_bin=weight_bin
                )
            )
            self.second_activation.append(nn.PReLU())
            self.second_normalization.append(nn.BatchNorm1d(dconv_size))
            self.last1x1_list.append(
                binary_layers.BitwiseConv1d(
                    dconv_size, bottleneck_channels, kernel_size,
                    use_gate=use_gate, adaptive_scaling=adaptive_scaling,
                    in_bin=in_bin, weight_bin=weight_bin
                )
            )
            dilation *= 2

    def forward(self, x):
        resid = x
        for i in range(layers):
            x = self.first1x1_list[i](x)
            x = self.first_activation[i](x)
            x = self.first_normalization[i](x)
            x = self.dconvs[i](x)
            x = self.second_activation[i](x)
            x = self.second_normalization[i](x)
            x = self.last1x1_list[i](x)
        x = x + resid

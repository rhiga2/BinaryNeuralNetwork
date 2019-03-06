import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
from abc import ABC

class TanhSTE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output * (1 - torch.tanh(x)**2)

class ClippedSTE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output * (torch.abs(x) <= 1).to(grad_output.dtype)

class STE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output

class SignSwissSTE(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sign_swiss_deriv = (2 - x*torch.tanh(x/2))/(1 + torch.cosh(x))
        return grad_output * sign_swiss_deriv

clipped_ste = ClippedSTE.apply
ste = STE.apply
tanh_ste = TanhSTE.apply
sign_swiss_ste = SignSwissSTE.apply

class SignSwiss(nn.Module):
    def __init__(self, beta=1, ste=True):
        super().__init__()
        self.beta = nn.Parameter(torch.FloatTensor([beta]),
            requires_grad=True)
        self.ste = ste

    def forward(self, x):
        if self.ste:
            return sign_swiss_ste(self.beta*x)
        sig_x = torch.sigmoid(self.beta*x)
        return 2*sig_x*(1 + self.beta*x*(1 - sig_x)) - 1

class ParameterizedTanh(nn.Module):
    def __init__(self, beta=1, ste=False):
        super().__init__()
        self.beta = nn.Parameter(torch.FloatTensor([beta]),
            requires_grad=True)
        self.ste = ste

    def forward(self, x):
        if self.ste:
            return tanh_ste(self.beta*x)
        return torch.tanh(self.beta*x)

class ParameterizedHardTanh(nn.Module):
    def __init__(self, beta=1, ste=False):
        super().__init__()
        self.beta = nn.Parameter(torch.FloatTensor([beta]),
            requires_grad=True)
        self.hard_tanh = nn.Hardtanh
        self.ste = ste

    def forward(self, x):
        if self.ste:
            return clipped_ste(self.beta*x)
        return self.hard_tanh(beta*x)

def pick_activation(activation_name, **kwargs):
    if activation_name == 'ste':
        activation = lambda : ste
    elif activation_name == 'clipped_ste':
        activation = lambda : clipped_ste
    elif activation_name == 'relu':
        activation = lambda : nn.ReLU(**kwargs)
    elif activation_name == 'prelu':
        activation = lambda : nn.PReLU(**kwargs)
    elif activation_name == 'tanh':
        activation = lambda : torch.tanh
    elif activation_name == 'ptanh':
        activation = lambda : ParameterizedTanh(**kwargs)
    elif activation_name == 'ptanh':
        activation = lambda : ParameterizedTanh(**kwargs, ste=True)
    elif activation_name == 'tanh_ste':
        activation = lambda : tanh_ste
    elif activation_name == 'hard_tanh':
        activation = lambda : nn.Hardtanh(**kwargs)
    elif activation_name == 'phtanh':
        activation = lambda : ParameterizedHardTanh(**kwargs)
    elif activation_name == 'phtanh_ste':
        activation = lambda : ParameterizedHardTanh(**kwargs, ste=True)
    elif activation_name == 'sign_swiss':
        activation = lambda : SignSwiss(**kwargs)
    elif activation_name == 'sign_swiss_ste':
        activation = lambda : SignSwiss(**kwargs, ste=True)
    elif activation_name == 'identity':
        activation = None
    return activation

def init_weight(size, gain=1, one_sided=False):
    w = torch.empty(size)
    nn.init.xavier_uniform_(w, gain=gain)
    if one_sided:
        w = torch.abs(w)
    w = nn.Parameter(w, requires_grad=True)
    return w

class BitwiseAbstractClass(ABC):
    def initialize(self, use_gate=False, in_binactiv=None,
        w_binactiv=None, bn_momentum=0.1, scale_weights=None,
        scale_activations=None):
        '''
        You MUST call parent module __init__ function before calling this
        function.
        '''
        self.bitwise = True
        self.in_binactiv = None
        if in_binactiv is not None:
            self.in_binactiv = in_binactiv()
        self.w_binactiv = None
        if w_binactiv is not None:
            self.w_binactiv = w_binactiv()
        self.use_gate = use_gate
        self.scale_weights = scale_weights
        self.scale_activations = scale_activations

        self.gate = None
        if self.use_gate:
            self.gate = init_weight(self.weight, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        if self.scale_weights == 'learnable':
            self.alpha = nn.Parameter(torch.ones(self.weight.size(0)))
            for i in range(len(self.weight.size()) - 1):
                self.alpha = self.alpha.unsqueeze(-1)

    def update_beta(self, sparsity):
        if sparsity == 0:
            return
        w = self.weight.cpu().data.numpy()
        beta = torch.tensor(np.percentile(np.abs(w), sparsity),
            dtype=weight.dtype, device=weight.device)
        self.beta = nn.Parameter(beta, requires_grad=False)

    def clip_weights(self):
        self.weight.data.clamp_(-1, 1)

    def drop_weights(self):
        weight = self.weight
        if self.gate is not None:
            weight = weight*self.binarize_gate()
        if self.beta != 0:
            weight = weight*(torch.abs(weight) >= self.beta).to(torch.float)
        return weight

    def binarize_gate(self):
        return (self.w_binactiv(self.gate) + 1) / 2

    def binarize_inputs(self, x):
        x_bin = x
        if self.in_binactiv is not None:
            x_bin = self.in_binactiv(x)
            if self.scale_activations == 'average':
                x_scale = torch.abs(x).mean(1, keepdim=True)
                if self.scale_conv:
                    x_scale = self.scale_conv(x_scale)
                x_bin = x_scale * x_bin
        return x_bin

    def binarize_weights(self):
        weight = self.weight
        if self.w_binactiv is not None:
            weight = self.drop_weights()
            weight = self.w_binactiv(self.weight)
            if self.scale_weights == 'average':
                weight_scale = torch.abs(self.weight)
                for i in range(len(self.weight.size()) - 1):
                    weight_scale = weight_scale.mean(-1, keepdim=True)
                weight = weight_scale * weight
            elif self.scale_weights == 'learnable':
                weight = self.alpha * weight
        return weight

class BitwiseLinear(nn.Linear, BitwiseAbstractClass):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, bias=True, use_gate=False,
            in_binactiv=None, w_binactiv=None, bn_momentum=0.1,
            scale_weights=None, scale_activations=None):
        super().__init__(input_size, output_size, bias=bias)
        super().initialize(use_gate=use_gate, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv, bn_momentum=bn_momentum,
            scale_weights=scale_weights,
            scale_activations=scale_activations)

    def forward(self, x):
        layer_in = self.binarize_inputs(x)
        weight = self.binarize_weights()
        return F.linear(layer_in, weight, self.bias)

class BitwiseConv1d(nn.Conv1d, BitwiseAbstractClass):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, bias=True,
        use_gate=False, in_binactiv=None,
        w_binactiv=None, bn_momentum=0.1,
        scale_weights=None, scale_activations=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        super().initialize(use_gate=use_gate, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv, bn_momentum=bn_momentum,
            scale_weights=scale_weights,
            scale_activations=scale_activations)

        # self.scale_conv = nn.Conv1d(1, 1, kernel_size, stride=stride,
        #    padding=padding, dilation=dilation)
        # weight = self.scale_conv.weight
        # weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        # self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        '''
        x (batch size, channels, length)
        '''
        layer_in = self.binarize_inputs(x)
        weight = self.binarize_weights()
        layer_out = F.conv1d(layer_in, weight, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation)
        return layer_out

class BitwiseConv2d(nn.Conv2d, BitwiseAbstractClass):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, groups=1, dilation=1, bias=True,
                use_gate=False, in_binactiv=None, w_binactiv=None,
                bn_momentum=0.1, scale_weights=None,
                scale_activations=None):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation, bias=bias
        )
        super().initialize(use_gate=use_gate, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv, bn_momentum=bn_momentum,
            scale_weights=scale_weights,
            scale_activations=scale_activations)

        # self.scale_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
        #     padding=padding, dilation=dilation, bias=False)
        # weight = self.scale_conv.weight
        # weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        # self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        '''
        x (batch size, channels, height, width)
        '''
        layer_in = self.binarize_inputs(x)
        weight = self.binarize_weights()
        return F.conv2d(layer_in, weight, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation)

class BitwiseConvTranspose1d(nn.ConvTranspose1d, BitwiseAbstractClass):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, groups=1, bias=True, use_gate=False,
                in_binactiv=None, w_binactiv=None, dilation=1,
                bn_momentum=0.1, scale_weights=None,
                scale_activations=None):
        super(BitwiseConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation, bias=bias
        )
        super().initialize(use_gate=use_gate, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv, bn_momentum=bn_momentum,
            scale_weights=scale_weights)

        # self.scale_conv = nn.ConvTranspose1d(1, 1, kernel_size,
        #     stride=stride, padding=padding, dilation=dilation, bias=False)
        # weight = self.scale_conv.weight
        # weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        # self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        '''
        x (batch size, channels, length)
        '''
        layer_in = self.binarize_inputs(x)
        weight = self.binarize_weights()
        return F.conv_transpose1d(layer_in, self.weight, self.bias,
                stride=self.stride, padding=self.padding, groups=self.groups,
                dilation=self.dilation)

class ScaleLayer(nn.Module):
    def __init__(self, num_channels, len_size=2):
        '''
        Assume dimension 1 is the channel dimension
        '''
        super(ScaleLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_channels), requires_grad=True)
        for _ in range(len_size-2):
            self.gamma = self.gamma.unsqueeze(-1)

    def forward(self, x):
        return torch.abs(self.gamma) * x

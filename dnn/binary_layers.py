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

class HardTanh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, -1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return torch.clamp(grad_output, -1, 1)

class SignSwissSTE(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.saved_for_backward(x)
        ctx.saved_for_backward(beta)
        return torch.sign(x)

    @staticmethod
    def _sign_swish_back_helper(x, beta):
        numer = beta * (2 - beta*x*torch.tanh(beta*x/2))
        denom = 1 + torch.cosh(beta * x)
        return numer / denom

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        beta = ctx.saved_tensors[1]
        back_helper = _sign_swish_back_helper(x, beta)
        back_helper = grad_output * back_helper
        return beta * back_helper, torch.sum(x * back_helper)

def softsign(ctx, x, gamma):
    return x / (torch.abs(x) + gamma)

clipped_ste = ClippedSTE.apply
ste = STE.apply
tanh_ste = TanhSTE.apply
hard_tanh = HardTanh.apply
signswiss_ste = SignSwissSTE.apply

def pick_activation(activation_name):
    if activation_name == 'ste':
        activation = ste
    elif activation_name == 'clipped_ste':
        activation = clipped_ste
    elif activation_name == 'relu':
        activation = F.relu
    elif activation_name == 'tanh':
        activation = torch.tanh
    elif activation_name == 'tanh_ste':
        activation = tanh_ste
    elif activation_name == 'hard_tanh':
        activation = hard_tanh
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
    def initialize(self, binactiv=None, use_gate=False, num_binarizations=1,
        scale_weights=None):
        '''
        You must call parent module __init__ function before calling this
        function.
        '''
        self.bitwise = True
        self.binactiv = binactiv
        self.gate = None
        if use_gate:
            self.gate = init_weight(self.weight, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.num_binarizations = num_binarizations
        self.scale_conv = None
        self.scale_weights = scale_weights
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
        return (self.binactiv(self.gate) + 1) / 2

    def binarize_inputs(self, x):
        estimate = x
        if self.binactiv is not None:
            weight = self.drop_weights()
            residual = x
            estimate = torch.zeros_like(x)
            for _ in range(self.num_binarizations):
                x_bin = self.binactiv(residual)
                x_scale = torch.abs(residual).mean(1, keepdim=True)
                if self.scale_conv:
                    x_scale = self.scale_conv(x_scale)
                estimate += x_scale * x_bin
                residual = residual - estimate
        return estimate

    def binarize_weights(self):
        weight = self.weight
        if self.binactiv is not None:
            weight = self.binactiv(self.weight)
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
        binactiv=None, scale_weights=None, num_binarizations=1):
        super().__init__(input_size, output_size, bias=bias)
        super().initialize(binactiv=binactiv, use_gate=use_gate,
            scale_weights=scale_weights, num_binarizations=num_binarizations)

    def forward(self, x):
        layer_in = self.binarize_inputs(x)
        weight = self.binarize_weights()
        return F.linear(layer_in, weight, self.bias)

class BitwiseConv1d(nn.Conv1d, BitwiseAbstractClass):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        binactiv=None, bias=True, scale_weights=None, num_binarizations=1):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        super().initialize(binactiv=binactiv, use_gate=use_gate,
            scale_weights=scale_weights, num_binarizations=num_binarizations)

        self.scale_conv = nn.Conv1d(1, 1, kernel_size, stride=stride,
            padding=padding, dilation=dilation)
        weight = self.scale_conv.weight
        weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)

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
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        binactiv=None, bias=True, scale_weights=None, num_binarizations=1):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation, bias=bias
        )
        super().initialize(binactiv=binactiv, use_gate=use_gate,
            scale_weights=scale_weights, num_binarizations=num_binarizations)

        self.scale_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        weight = self.scale_conv.weight
        weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)

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
        stride=1, padding=0, groups=1, use_gate=False,
        dilation=1, binactiv=None, bias=True, scale_weight=None,
        num_binarizations=1):
        super(BitwiseConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation, bias=bias
        )
        super().initialize(binactiv=binactiv, use_gate=use_gate,
            scale_weights=scale_weights, num_binarizations=num_binarizations)

        self.scale_conv = nn.ConvTranspose1d(1, 1, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False)
        weight = self.scale_conv.weight
        weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)

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

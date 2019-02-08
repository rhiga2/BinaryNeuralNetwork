import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import math
from abc import ABC, abstractmethod

class BitwiseActivation(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output * (1 - torch.tanh(x)**2), None

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

class STE_Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.tanh(x)

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

def softsign(ctx, x, gamma):
    ctx.save_for_backward(x)
    return x / (torch.abs(x) + gamma)

bitwise_activation = BitwiseActivation.apply
clipped_ste = ClippedSTE.apply
ste = STE.apply
ste_tanh = STE_Tanh.apply
hard_tanh = HardTanh.apply

def pick_activation(activation_name):
    if activation_name == 'ste':
        activation = ste
    elif activation_name == 'clipped_ste':
        activation = clipped_ste
    elif activation_name == 'bitwise_activation':
        activation = bitwise_activation
    elif activation_name == 'relu':
        activation = F.relu
    elif activation_name == 'tanh':
        activation = torch.tanh
    elif activation_name == 'ste_tanh':
        activation = ste_tanh
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

def update_beta(weight, sparsity):
    if sparsity == 0:
        return
    w = weight.cpu().data.numpy()
    beta = torch.tensor(np.percentile(np.abs(w), sparsity),
        dtype=weight.dtype, device=weight.device)
    return nn.Parameter(beta, requires_grad=False)

def clip_weights(weight, gate, use_gate=False):
    weight.data.clamp_(-1, 1)
    if use_gate:
        gate.data.clamp_(-1, 1)

def binarize_gate(gate, binactiv):
    return (binactiv(gate) + 1) / 2

def drop_weights(weight, gate=None, binactiv=None, beta=0):
    if gate is not None:
        return weight * binarize_gate(gate, binactiv)
    if beta != 0:
        weight = weight * (torch.abs(weight) >= beta).to(torch.float)
    return weight

def binarize_weights_and_inputs(x, weight, gate=None, binactiv=None, beta=0):
    if binactiv is not None:
        x = binactiv(x)
        weight = drop_weights(weight, gate=gate, binactiv=binactiv,
            beta=beta)
        weight = binactiv(weight)
    return x, weight

class BitwiseLinear(nn.Linear):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, use_gate=False,
        adaptive_scaling=False, binactiv=None):
        super(BitwiseLinear, self).__init__(input_size, output_size, bias=True)
        self.input_size = input_size
        self.output_size = output_size
        self.binactiv = binactiv
        self.use_gate = use_gate
        self.gate = None
        if use_gate:
            self.gate = init_weight((output_size, input_size), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.adaptive_scaling = adaptive_scaling

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        layer_in, weight = binarize_weights_and_inputs(x, self.weight, self.gate,
            self.binactiv, beta=self.beta)
        layer_out = F.linear(layer_in, weight, self.bias)
        if self.adaptive_scaling:
            in_scale = torch.abs(x).mean(1, keepdim=True)
            weight_scale = torch.abs(self.weight).mean(1)
            return in_scale * weight_scale * layer_out
        return layer_out

    def __repr__(self):
        return 'BitwiseLinear({}, {}, use_gate={}, adaptive_scaling={})'.format(self.input_size,
        self.output_size, self.use_gate, self.adaptive_scaling)

class BitwiseConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        adaptive_scaling=False, binactiv=None):
        super(BitwiseConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups)
        self.use_gate = use_gate
        self.binactiv = binactiv
        self.gate = None
        if use_gate:
            self.gate = init_weight(self.weight.size(), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.adaptive_scaling = adaptive_scaling
        if adaptive_scaling:
            self.scale_conv = nn.Conv1d(1, 1, kernel_size, stride=stride,
                padding=padding, dilation=dilation)

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        '''
        x (batch size, channels, length)
        '''
        layer_in, weight = binarize_weights_and_inputs(x, self.weight, self.gate,
            self.binactiv, beta=self.beta)
        layer_out = F.conv1d(layer_in, weight, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation)
        if self.adaptive_scaling:
            in_scale = self.scale_conv(torch.abs(x).mean(1, keepdim=True))
            weight_scale = torch.abs(self.weight).mean(1).mean(1)
            weight_scale = weight_scale.unsqueeze(1)
            return weight_scale * in_scale * layer_out
        return layer_out

    def __repr__(self):
        return 'BitwiseConv1d({}, {}, {}, stride={}, padding={}, groups={}, dilation={}, use_gate={}, adaptive_scaling={})'.format(
        self.in_channels,
        self.out_channels, self.kernel_size, self.stride, self.padding,
        self.groups, self.dilation, self.use_gate, self.adaptive_scaling)

class BitwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        binactiv=None, adaptive_scaling=False):
        super(BitwiseConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation
        )
        self.use_gate = use_gate
        self.gate = None
        if use_gate:
            self.gate = init_weight(self.weight.size(), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.adaptive_scaling = adaptive_scaling
        if adaptive_scaling:
            self.scale_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
                padding=padding, dilation=dilation)

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        '''
        x (batch size, channels, height, width)
        '''
        layer_in, weight = binarize_weights_and_inputs(x, self.weight, self.gate,
            self.binactiv, beta=self.beta)
        layer_out = F.conv2d(layer_in, weight, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation)
        if self.adaptive_scaling:
            in_scale = self.scale_conv(torch.abs(x).mean(1, keepdim=True))
            weight_scale = torch.abs(self.weight).mean(1).mean(1).mean(1)
            weight_scale = weight_scale.unsqueeze(1).unsqueeze(1)
            return weight_scale * in_scale * layer_out
        return layer_out

    def __repr__(self):
        return 'BitwiseConv2d({}, {}, {}, stride={}, padding={}, groups={}, dilation={}, use_gate={}, adaptive_scaling={})'.format(
        self.in_channels, self.out_channels, self.kernel_size,
        self.stride, self.padding, self.groups, self.dilation,
        self.use_gate, self.adaptive_scaling)

class BitwiseConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, use_gate=False,
        dilation=1, adaptive_scaling=False, binactiv=None):
        super(BitwiseConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation
        )
        self.use_gate = use_gate
        self.gate = None
        self.binactiv = binactiv
        if use_gate:
            self.gate = init_weight(self.weight.size(), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.adaptive_scaling = adaptive_scaling
        if adaptive_scaling:
            self.scale_conv = nn.ConvTranspose1d(1, 1, kernel_size,
                stride=stride, padding=padding, dilation=dilation)

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        '''
        x (batch size, channels, length)
        '''
        layer_in, weight = binarize_weights_and_inputs(x, self.weight, self.gate,
            self.binactiv, beta=self.beta)
        layer_out = F.conv_transpose1d(layer_in, weight, self.bias,
            stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation)
        if self.adaptive_scaling:
            in_scale = self.scale_conv(torch.abs(x).mean(1, keepdim=True))
            weight_scale = torch.abs(self.weight).mean(0).mean(1)
            weight_scale = weight_scale.unsqueeze(1)
            return in_scale * weight_scale * layer_out
        return layer_out

    def __repr__(self):
        return 'BitwiseConvTranspose1d({}, {}, {}, stride={}, padding={}, groups={}, use_gate={}, dilation={}, adaptive_scaling={})'.format(
        self.in_channels, self.out_channels, self.kernel_size, self.stride,
        self.padding, self.groups, self.use_gate, self.dilation,
        self.adaptive_scaling)

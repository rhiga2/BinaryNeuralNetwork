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

def binarize_weights_and_inputs(x, weight, gate=None, binactiv=None, beta=0,
    scale_conv=None, num_binarizations=1, scale_weights=False):
    activations = [x]
    if binactiv is not None:
        weight = drop_weights(weight, gate=gate, binactiv=binactiv,
            beta=beta)
        residual = x
        activations = []
        for _ in range(num_binarizations):
            bin_x = binactiv(residual)
            x_scale = torch.abs(x).mean(1, keepdim=True)
            if scale_conv:
                x_scale = scale_con(x_scale)
            activations.append(x_scale * bin_x)
            residual = x - x_scale * bin_x

        if scale_weights:
            weight_scale = torch.abs(weight)
            for i in range(len(weight.size()) - 1):
                weight_scale = weight_scale.mean(-1, keepdim=True)
            weight = weight_scale * binactiv(weight)
        else:
            weight = binactiv(weight)

    return activations, weight

class BitwiseLinear(nn.Linear):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, use_gate=False,
        binactiv=None, bias=True, scale_weights=False, num_binarizations=1):
        super(BitwiseLinear, self).__init__(input_size, output_size, bias=bias)
        self.input_size = input_size
        self.output_size = output_size
        self.binactiv = binactiv
        self.use_gate = use_gate
        self.gate = None
        if use_gate:
            self.gate = init_weight((output_size, input_size), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.num_binarizations = num_binarizations
        self.scale_weights = scale_weights

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        inputs, weight = binarize_weights_and_inputs(x, self.weight, gate=self.gate,
            binactiv=self.binactiv, beta=self.beta, scale_conv=None,
            num_binarizations=self.num_binarizations,
            scale_weights=self.scale_weights)
        layer_out = 0
        for layer_in in inputs:
            layer_out += F.linear(layer_in, weight, self.bias)
        return layer_out

    def __repr__(self):
        return 'BitwiseLinear({}, {}, use_gate={})'.format(self.input_size,
        self.output_size, self.use_gate)

class BitwiseConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        binactiv=None, bias=True, scale_weights=False, num_binarizations=1):
        super(BitwiseConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.use_gate = use_gate
        self.binactiv = binactiv
        self.gate = None
        if use_gate:
            self.gate = init_weight(self.weight.size(), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.scale_conv = nn.Conv1d(1, 1, kernel_size, stride=stride,
            padding=padding, dilation=dilation)
        weight = self.scale_conv.weight
        weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)
        self.scale_weights = scale_weights
        self.num_binarizations = num_binarizations

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        '''
        x (batch size, channels, length)
        '''
        inputs, weight = binarize_weights_and_inputs(x, weight, gate=self.gate,
            binactiv=self.binactiv, beta=self.beta, scale_conv=self.scale_conv,
            num_binarizations=self.num_binarizations, scale_weights=self.scale_weights)
        layer_out = 0
        for layer_in in inputs:
            layer_out += F.conv1d(layer_in, weight, self.bias,
                stride=self.stride, padding=self.padding, groups=self.groups,
                dilation=self.dilation)
        return layer_out

    def __repr__(self):
        return 'BitwiseConv1d({}, {}, {}, stride={}, padding={}, groups={}, dilation={}, use_gate={})'.format(
        self.in_channels,
        self.out_channels, self.kernel_size, self.stride, self.padding,
        self.groups, self.dilation, self.use_gate)

class BitwiseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        binactiv=None, bias=True, scale_weights=False, num_binarizations=1):
        super(BitwiseConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation, bias=bias
        )
        self.use_gate = use_gate
        self.gate = None
        self.binactiv = binactiv
        if use_gate:
            self.gate = init_weight(self.weight.size(), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.scale_conv = nn.Conv2d(1, 1, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        weight = self.scale_conv.weight
        weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)
        self.scale_weights = scale_weights
        self.num_binarizations = num_binarizations

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        '''
        x (batch size, channels, height, width)
        '''
        inputs, weight = binarize_weights_and_inputs(x, self.weight, gate=self.gate,
            binactiv=self.binactiv, beta=self.beta, scale_conv=self.scale_conv,
            num_binarizations=self.num_binarizations,
            scale_weights=self.scale_weights)
        layer_out = 0
        for layer_in in inputs:
            layer_out += F.conv2d(layer_in, weight, self.bias,
                stride=self.stride, padding=self.padding, groups=self.groups,
                dilation=self.dilation)
        return layer_out

    def __repr__(self):
        return 'BitwiseConv2d({}, {}, {}, stride={}, padding={}, groups={}, dilation={}, use_gate={})'.format(
        self.in_channels, self.out_channels, self.kernel_size,
        self.stride, self.padding, self.groups, self.dilation,
        self.use_gate)

class BitwiseConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, groups=1, use_gate=False,
        dilation=1, binactiv=None, bias=True, scale_weight=True,
        num_binarizations=1):
        super(BitwiseConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, dilation=dilation, bias=bias
        )
        self.use_gate = use_gate
        self.gate = None
        self.binactiv = binactiv
        if use_gate:
            self.gate = init_weight(self.weight.size(), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.scale_conv = nn.ConvTranspose1d(1, 1, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False)
        weight = self.scale_conv.weight
        weight = 1 / (np.prod(weight.size())) * torch.ones_like(weight)
        self.scale_conv.weight = nn.Parameter(weight, requires_grad=False)
        self.scale_weights = scale_weights
        self.num_binarizations = num_binarizations

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        '''
        x (batch size, channels, length)
        '''
        inputs, weight = binarize_weights_and_inputs(x, self.weight, self.gate,
            self.binactiv, beta=self.beta, scale_conv=self.scale_conv,
            num_binarizations=self.num_binarizations,
            scale_weights=self.scale_weights)
        layer_out = 0
        for layer_in in inputs:
            layer_out += F.conv_transpose1d(layer_in, self.weight, self.bias,
                stride=self.stride, padding=self.padding, groups=self.groups,
                dilation=self.dilation)
        return layer_out

    def __repr__(self):
        return 'BitwiseConvTranspose1d({}, {}, {}, stride={}, padding={}, groups={}, use_gate={}, dilation={})'.format(
        self.in_channels, self.out_channels, self.kernel_size, self.stride,
        self.padding, self.groups, self.use_gate, self.dilation)

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

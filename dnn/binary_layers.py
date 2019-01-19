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

bitwise_activation = BitwiseActivation.apply
clipped_ste = ClippedSTE.apply
ste = STE.apply
ste_tanh = STE_Tanh.apply

def pick_activation(activation_name):
    if activation_name == 'ste':
        activation = ste
    elif activation_name == 'clipped_ste':
        activation = clipped_ste
    elif activation_name == 'bitwise_activation':
        activation = bitwise_activation
    elif activation_name == 'relu':
        activation = nn.ReLU()
    elif activation_name == 'tanh':
        activation = torch.tanh
    elif activation_name == 'ste_tanh':
        activation = ste_tanh
    elif activation_name == 'identity':
        activation = lambda x : x
    return activation

def add_logistic_noise(x, sigma=0.1):
    u = torch.rand_like(x)
    x = x + sigma * (torch.log(u) - torch.log(1 - u))
    return x

def init_weight(size, requires_grad=True, one_sided=False):
    w = torch.empty(size)
    nn.init.xavier_uniform_(w, gain=gain)
    if one_sided:
        w = torch.abs(w)
    w = nn.Parameter(w, requires_grad=requires_grad)
    return w

def update_beta(weight, sparsity):
    if sparsity == 0:
        return
    w = weight.cpu().data.numpy()
    beta = torch.tensor(np.percentile(np.abs(w), sparsity),
        dtype=self.weight.dtype, device=weight.device)
    return nn.Parameter(beta, requires_grad=False)

def clip_weights(weight, gate, use_gate=False):
    new_weight = nn.Parameter(torch.clamp(weight, -1, 1))
    new_gate = None
    if use_gate:
        new_gate = nn.Parameter(torch.clamp(gate, -1, 1))
    return new_weight, new_gate

def get_effective_weight(weight, gate, activation, beta=0, use_gate=False):
    w = weight
    if beta != 0:
        w = w * (torch.abs(w) >= beta).to(float)
    w = activation(w)
    if use_gate:
        w = w * ((activation(gate)+1)/2)
    return w

class BitwiseLinear():
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, use_gate=False,
        activation='tanh'):
        super(BitwiseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = init_weight((output_size, input_size), True)
        self.activation_name = activation
        self.activation = pick_activation(activation)
        self.use_gate = use_gate
        self.gate = None
        if use_gate:
            self.gate = init_weight((output_size, input_size), True,
                one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.mode = 'real'

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        self.weight, self.gate = clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        w = get_effective_weight(self.weight, self.gate, self.activation,
            self.beta=0, self.use_gate=False)
        return F.linear(x, w, None)

    def __repr__(self):
        return 'BitwiseLinear({}, {}, user_gate={}, \
            activation_name={})'.format(self.input_size, self.output_size,
            self.use_gate, self.activation_name)

class BitwiseConv1d(nn.Conv1d):
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        activation='tanh'):
        super(BitwiseConv1d, self).__init__(
            input_channels, output_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=False)
        self.use_gate = use_gate
        self.activation_name = activation
        self.activation = pick_activation(activation)
        if use_gate:
            self.gate = init_weight(self.weight.size(), one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        self.weight, self.gate = clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        w = get_effective_weight(self.weight, self.gate, self.activation,
            self.beta=0, self.use_gate=False)
        return F.conv1d(x, w, None, stride=self.stride,
            padding=self.padding, groups=self.groups, dilation=self.dilation)

    def __repr__(self):
        return 'BitwiseConv1d({}, {}, {}, stride={}, padding={}, groups={}, \
            dilation={}, use_gate={}, activation={})'.format(self.input_channels,
            self.output_channels, self.kernel_size, self.stride, self.padding,
            self.groups, self.dilation, self.use_gate, self.activation_name)

class BitwiseConv2d(nn.Conv2d):
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, groups=1, dilation=1, use_gate=False,
        activation=torch.tanh):
        super(BitwiseConv2d, self).__init__(
            input_channels, output_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False, dilation=dilation
        )
        self.use_gate = use_gate
        self.activation = activation
        if use_gate:
            self.gate = init_weight(self.weight.size(), requires_grad, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        self.weight, self.gate = clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        w = get_effective_weight(self.weight, self.gate, self.activation,
            self.beta=0, self.use_gate=False)
        return F.conv2d(x, w, None, stride=self.stride,
            padding=self.padding, groups=self.groups, dilation=self.dilation)

    def __repr__(self):
        return 'BitwiseConv2d'


class BitwiseConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, groups=1, requires_grad=True, use_gate=False,
        dilation=1, activation=torch.tanh):
        super(BitwiseConvTranspose1d, self).__init__(
            input_channels, output_channels, kernel_size, stride=stride,
            padding=padding, groups=groups, bias=False, dilation=dilation
        )
        self.use_gate = True
        self.activation = activation
        if use_gate:
            self.gate = init_weight(self.weight.size(), requires_grad, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)

    def update_beta(self, sparsity):
        self.beta = update_beta(self.weight, sparsity)

    def clip_weights(self):
        self.weight, self.gate = clip_weights(self.weight, self.gate, self.use_gate)

    def forward(self, x):
        w = get_effective_weight(self.weight, self.gate, self.activation,
            self.beta=0, self.use_gate=False)
        return F.conv_transpose1d(x, w, None, stride=self.stride,
            padding=self.padding, groups=self.groups, dilation=self.dilation)

    def __repr__(self):
        return 'BitwiseConvTranspose1d'

class BitwiseResidualLinear(nn.Module):
    def __init__(self, input_size):
        super(BitwiseResidualLinear, self).__init__()
        self.input_size = input_size
        self.activation = torch.tanh
        self.dense1 = BitwiseLinear(input_size, input_size)
        self.dense2 = BitwiseLinear(input_size, input_size)

    def forward(self, x):
        x = self.activation(self.dense1(x))
        return x + self.dense2(x)

    def noisy(self):
        self.dense1.noisy()
        self.dense2.noisy()

    def inference(self):
        self.dense1.inference()
        self.dense2.inference()

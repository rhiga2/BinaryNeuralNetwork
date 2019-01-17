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

class BitwiseParams(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        return (x > beta).to(dtype=x.dtype, device=x.device) - (x < -beta).to(dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output * (torch.abs(x) <= 1).to(grad_output.dtype), None

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
bitwise_params = BitwiseParams.apply
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
    elif activation_name == 'prelu':
        activation = nn.PReLU()
    elif args.activation == 'tanh':
        activation = torch.tanh
    elif args.activation == 'ste_tanh':
        activation = ste_tanh
    else:
        print('Activation not recognized, using default activation: idenity')
        activation = lambda x : x
    return activation

def add_logistic_noise(x, sigma=0.1):
    u = torch.rand_like(x)
    x = x + sigma * (torch.log(u) - torch.log(1 - u))
    return x

def init_weight(size, requires_grad=True, gain=1, one_sided=False, scale=1.0):
    w = torch.empty(size)
    nn.init.xavier_uniform_(w, gain=gain)
    if one_sided:
        w = torch.abs(w)
    w = nn.Parameter(scale * w, requires_grad=requires_grad)
    return w

class BitwiseAbstractClass(nn.Module):
    @abstractmethod
    def __init__(self):
        super(BitwiseAbstractClass, self).__init__()
        self.weight = None
        self.gate = None
        self.requires_grad = True
        self.use_gate = False
        self.activation = torch.tanh

    @abstractmethod
    def forward(self):
        pass

    def update_beta(self, sparsity):
        if sparsity == 0:
            return
        w = self.weight.cpu().data.numpy()
        beta = torch.tensor(np.percentile(np.abs(w), sparsity),
            dtype=self.weight.dtype, device=self.weight.device)
        self.beta = nn.Parameter(beta, requires_grad=False)

    def clip_weights(self):
        self.weight = nn.Parameter(torch.clamp(self.weight, -1, 1))
        if self.use_gate:
            self.gate = nn.Parameter(torch.clamp(self.weight, -1, 1))

    def get_effective_weight(self):
        w = self.weight
        if self.beta != 0:
            w = w * (torch.abs(self.weight) >= self.beta).to(float)
        w = self.activation(w)
        if self.use_gate:
            w = w*((self.activation(self.gate)+1)/2)
        return w

class BitwiseLinear(BitwiseAbstractClass):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, requires_grad=True, use_gate=False,
        activation=torch.tanh, scale=1.0):
        super(BitwiseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.requires_grad = requires_grad
        self.weight = init_weight((output_size, input_size), requires_grad,
            scale=scale)
        self.activation = activation
        self.use_gate = use_gate
        if use_gate:
            self.gate = init_weight((output_size, input_size), requires_grad,
                one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = self.get_effective_weight()
        return F.linear(x, w, None)

    def __repr__(self):
        return 'BitwiseLinear(%d, %d, requires_grad=%r, use_gate=%r)' % \
        (self.input_size, self.output_size, self.requires_grad, self.use_gate)

class BitwiseConv1d(BitwiseAbstractClass):
    '''
    1D bitwise (Kim et. al) convolution
    '''
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, groups=1, requires_grad=True, use_gate=False,
        activation=torch.tanh, scale=1.0):
        super(BitwiseConv1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.requires_grad = requires_grad
        weight_size = (output_channels, input_channels//self.groups, kernel_size)
        self.weight = init_weight(weight_size, requires_grad, scale=scale)
        self.use_gate = use_gate
        self.activation = activation
        if self.use_gate:
            self.gate = init_weight(weight_size, requires_grad, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype),
            requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = get_effective_weight()
        return F.conv1d(x, w, None, stride=self.stride,
            padding=self.padding, groups=self.groups)

    def __repr__(self):
        return 'BitwiseConv1d(%d, %d, kernel_size=%d, stride=%d, padding=%d, \
            groups=%d, requires_grad=%r, use_gate=%r)' % \
            (self.input_channels, self.output_channels, self.kernel_size,
            self.stride, self.padding, self.groups, self.requires_grad,
            self.use_gate)

class BitwiseConvTranspose1d(BitwiseAbstractClass):
    '''
    Issue: Almost copy paste of BitwiseConv1d. Parameter dimensions may be incorrect
    '''
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, groups=1, requires_grad=True, use_gate=False,
        activation=torch.tanh, scale=1.0):
        super(BitwiseConvTranspose1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_gate = use_gate
        self.requires_grad = requires_grad
        self.activation = activation
        weight_size = (input_channels, output_channels // groups, kernel_size)
        self.weight = init_weight(weight_size, requires_grad, scale=scale)
        if use_gate:
            self.gate = init_weight(weight_size, requires_grad, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = self.get_effective_weight()
        return F.conv_transpose1d(x, w, None, stride=self.stride,
            padding=self.padding, groups=self.groups)

    def __repr__(self):
        return 'BitwiseConvTranspose1d(%d, %d, kernel_size=%d, stride=%d, \
            padding=%d, groups=%d, requires_grad=%r, use_gate=%r)' % \
            (self.input_channels, self.output_channels, self.kernel_size,
            self.stride, self.padding, self.groups, self.requires_grad,
            self.use_gate)

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

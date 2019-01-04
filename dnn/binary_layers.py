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

class SqueezedTanh(Function):
    @staticmethod
    def forward(ctx, x, temp):
        ctx.save_for_backward(x)
        return torch.tanh(temp * x)

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
        # relax as tanh or use straight through estimator?
        x = ctx.saved_tensors[0]
        return grad_output * (torch.abs(x) <= 1).to(grad_output.dtype), None

class Binarize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        # There are two options:
        # 1. Clamp gradients exceeding +/- 1
        # 2. Zero gradients exceeding +/- 1
        x = ctx.saved_tensors[0]
        return grad_output * (torch.abs(x) <= 1).to(grad_output.dtype)

bitwise_activation = BitwiseActivation.apply
squeezed_tanh = SqueezedTanh.apply
bitwise_params = BitwiseParams.apply
binarize = Binarize.apply

def add_logistic_noise(x, sigma=0.1):
    u = torch.rand_like(x)
    x = x + sigma * (torch.log(u) - torch.log(1 - u))
    return x

def init_weight(size, requires_grad=True, gain=1, one_sided=False):
    w = torch.empty(size)
    nn.init.xavier_uniform_(w, gain=gain)
    if one_sided:
        w = torch.abs(w)
    w = nn.Parameter(w, requires_grad=requires_grad)
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
        if sparsity == 0 or self.use_gate:
            return
        w = self.weight.cpu().data.numpy()
        beta = torch.tensor(np.percentile(np.abs(w), sparsity),
            dtype=self.weight.dtype, device=self.weight.device)
        self.beta = nn.Parameter(beta, requires_grad=False)

    def noisy(self):
        self.mode = 'noisy'
        self.weight = nn.Parameter(torch.tanh(self.weight),
            requires_grad=self.requires_grad)
        if self.use_gate:
            self.gate = nn.Parameter(torch.tanh(self.gate),
                requires_grad=self.requires_grad)
        self.activation = lambda x : bitwise_params(x, self.beta)

    def inference(self):
        self.mode = 'inference'
        self.activation = lambda x : bitwise_params(x, self.beta)
        self.weight = nn.Parameter(bitwise_params(self.weight, self.beta),
            requires_grad=self.requires_grad)
        if self.use_gate:
            self.gate = nn.Parameter((bitwise_params(self.gate, 0)+1)/2,
                requires_grad=self.requires_grad)

    def get_effective_weight(self):
        w = self.weight
        if self.use_noise and self.mode != 'inference':
            w = add_logistic_noise(self.weight)
        w = self.activation(w)
        if self.use_gate:
            g = self.gate
            if self.use_noise and self.mode != 'inference':
                g = add_logistic_noise(self.gate)
            w = w*((self.activation(g)+1)/2)
        return w

class BitwiseLinear(BitwiseAbstractClass):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, requires_grad=True, use_gate=False,
        activation=torch.tanh, use_noise=False):
        super(BitwiseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.requires_grad = requires_grad
        self.weight = init_weight((output_size, input_size), requires_grad)
        self.activation = activation
        self.use_gate = use_gate
        self.use_noise = use_noise
        if use_gate:
            self.gate = init_weight((output_size, input_size), requires_grad, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
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
        activation=torch.tanh, use_noise=False):
        super(BitwiseConv1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.requires_grad = requires_grad
        weight_size = (output_channels, input_channels//self.groups, kernel_size)
        self.weight = init_weight(weight_size, requires_grad)
        self.use_gate = use_gate
        self.use_noise = use_noise
        self.activation = activation
        if self.use_gate:
            self.gate = init_weight(weight_size, requires_grad, one_sided=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
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
        activation=torch.tanh, use_noise=False):
        super(BitwiseConvTranspose1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_gate = use_gate
        self.use_noise = use_noise
        self.requires_grad = requires_grad
        self.activation = activation
        weight_size = (input_channels, output_channels // groups, kernel_size)
        self.weight = init_weight(weight_size, requires_grad)
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

class BLRLinear(nn.Module):
    '''
    Binary local reparameterization linear
    '''
    def __init__(self, input_size, output_size):
        super(BLRLinear, self).__init__()
        w = torch.empty(out_size, in_size)
        nn.init.xavier_uniform_(w)
        self.weight = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        expect = torch.tanh(self.weight)
        mean = F.linear(x, expect)
        var = F.linear(x**2, 1-expect**2)
        return mean, var

class BLRSampler(Function):
    '''
    Binary local reparameterization activation
    '''
    def __init__(self, temp=0.1, eps=1e-5):
        self.temp = temp
        self.eps = eps

    @staticmethod
    def forward(self, mean, var):
        q = Normal(mean, var).cdf(0)
        U = torch.rand_like(mean)
        L = torch.log(U) - torch.log(1 - U)
        return torch.tanh((torch.log(1/(1-q+eps)-1+eps) + L)/temp)

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

def scale_only_bn(gamma, x):
    '''
     x is shape(N, C)
    '''
    return torch.abs(gamma) * x / torch.sqrt((torch.var(x, dim=0) + 1e-5))

class Scaler(nn.Module):
    '''
    Scale-only batch normalization
    '''
    def __init__(self, num_features, requires_grad=True):
        super(Scaler, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features), requires_grad=requires_grad)

    def forward(self, x):
        '''
        x is shape (N, C)
        '''
        return scale_only_bn(self.gamma, x)

class ConvScaler1d(nn.Module):
    def __init__(self, num_features, requires_grad=True):
        super(ConvScaler1d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features), requires_grad=requires_grad)

    def forward(self, x):
        '''
        x is shape (N, C, T)
        '''
        N, C, T = x.size()
        # convert shape (N, C, T) to (NT, C)
        x = x.permute(0, 2, 1).contiguous().view(-1, C)
        x = scale_only_bn(self.gamma, x)
        return x.view(-1, T, C).permute(0, 2, 1).contiguous()

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
        return grad_output * (1 - torch.tanh(x)**2)

class BitwiseParams(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.save_for_backward(x)
        return (x > beta).to(dtype=x.dtype, device=x.device) - (x < -beta).to(dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        # relax as tanh or use straight through estimator?
        x = ctx.saved_tensors[0]
        return grad_output * (1 - torch.tanh(x)**2), None

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
bitwise_params = BitwiseParams.apply
binarize = Binarize.apply

def init_weight(size, requires_grad=True, scale=1):
    w = torch.empty(size)
    nn.init.xavier_uniform_(w)
    w = nn.Parameter(scale*w, requires_grad=requires_grad)
    return w

def clip_params(mod, min=-1, max=1):
    state_dict = mod.state_dict()
    for name, param in state_dict.items():
        if name.endswith('weight') or name.endswith('bias'):
            state_dict[name] = torch.clamp(param, min, max)

class BitwiseAbstractClass(nn.Module):
    @abstractmethod
    def __init__(self):
        super(BitwiseAbstractClass, self).__init__()
        self.weight = None
        self.gate = None
        self.requires_grad = True
        self.use_gate = False

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

    def inference(self):
        self.mode = 'inference'
        self.weight = nn.Parameter(bitwise_params(self.weight, self.beta),
            requires_grad=self.requires_grad)
        if self.use_gate:
            self.gate = nn.Parameter((bitwise_params(self.gate, 0)+1)/2,
                requires_grad=self.requires_grad)

    def get_effective_weight(self):
        if self.real:
            w = torch.tanh(self.weight)
            if self.use_gate:
                w *= torch.sigmoid(self.gate)
        elif self.noisy:
            w = bitwise_params(param, self.beta)
            if self.use_gate:
                w *= torch.sigmoid(self.gate)
        return w

class BitwiseLinear(BitwiseAbstractClass):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, requires_grad=True, use_gate=False):
        super(BitwiseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.requires_grad = requires_grad
        self.weight = init_weight((output_size, input_size), requires_grad)
        self.use_gate = use_gate
        if use_gate:
            self.gate = init_weight((output_size, input_size), requires_grad)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = self.get_effective_weight()
        return F.linear(x, w, None)

    def __repr__(self):
        return 'BitwiseLinear(%d, %d)' % (self.input_size, self.output_size)

class BitwiseConv1d(BitwiseAbstractClass):
    '''
    1D bitwise (Kim et. al) convolution
    '''
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, groups=1, requires_grad=True, use_gate=False):
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
        if self.use_gate:
            self.gate = init_weight(weight_size, requires_grad)
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
        stride=1, padding=0, groups=1, requires_grad=True):
        super(BitwiseConvTranspose1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_gate = gate
        self.requires_grad = requires_grad
        weight_size = (input_channels, output_channels // groups, kernel_size)
        self.weight = init_weight(weight_size, requires_grad)
        if use_gate:
            self.gate = init_weight(weight_size, requires_grad)
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

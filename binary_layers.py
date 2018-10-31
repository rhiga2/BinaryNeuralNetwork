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

class SineActivation(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(torch.sin(x))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output * torch.cos(x)

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
sine_activation = SineActivation.apply
binarize = Binarize.apply

def init_weight(size, requires_grad=True):
    w = torch.empty(size)
    nn.init.xavier_uniform_(w)
    w = nn.Parameter(w, requires_grad=requires_grad)
    return w

def init_bias(size, requires_grad=True):
    b = torch.zeros(size)
    b = nn.Parameter(b, requires_grad=requires_grad)
    return b

def clip_params(mod, min=-1, max=1):
    state_dict = mod.state_dict()
    for name, param in state_dict.items():
        if name.endswith('weight') or name.endswith('bias'):
            state_dict[name] = torch.clamp(param, min, max)

def convert_param(param, beta=0, mode='real'):
    '''
    Converts parameter to binary using bitwise nn scheme
    '''
    if mode == 'real':
        return torch.tanh(param)
    elif mode == 'noisy':
        return bitwise_params(param, beta)
    return param

class BitwiseAbstractClass(nn.Module):
    @abstractmethod
    def __init__(self):
        super(BitwiseAbstractClass, self).__init__()

    @abstractmethod
    def forward(self):
        pass

    def update_beta(self, sparsity):
        if sparsity == 0:
            return
        w = self.weight.cpu().data.numpy()
        params = np.abs(w)
        if self.biased:
            b = self.bias.cpu().data.numpy()
            params = np.abs(np.concatenate((w, np.expand_dims(b, axis=1)), axis=1))
        beta = torch.tensor(np.percentile(params, sparsity), dtype=self.weight.dtype,
            device=self.weight.device)
        self.beta = nn.Parameter(beta, requires_grad=False)

    def noisy(self):
        self.mode = 'noisy'
        self.weight = nn.Parameter(torch.tanh(self.weight),
            requires_grad=self.requires_grad)
        if self.biased:
            self.bias = nn.Parameter(torch.tanh(self.bias),
                requires_grad=self.requires_grad)

    def inference(self):
        self.mode = 'inference'
        self.weight = nn.Parameter(bitwise_params(self.weight, self.beta),
            requires_grad=self.requires_grad)
        if self.biased:
            self.bias = nn.Parameter(bitwise_params(self.bias, self.beta),
                requires_grad=self.requires_grad)

class BitwiseLinear(BitwiseAbstractClass):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size, biased=False, requires_grad=True):
        super(BitwiseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.biased = biased
        self.requires_grad = requires_grad
        self.weight = init_weight((output_size, input_size), requires_grad)
        self.bias = None
        if biased:
            self.bias = init_bias(output_size, requires_grad)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = convert_param(self.weight, self.beta, self.mode)
        b = None
        if self.biased:
            b = convert_param(self.bias, self.beta, self.mode)
        return F.linear(x, w, b)

    def __repr__(self):
        return 'BitwiseLinear(%d, %d)' % (self.input_size, self.output_size)

class BitwiseConv1d(BitwiseAbstractClass):
    '''
    1D bitwise (Kim et. al) convolution
    '''
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, biased=False, groups=1, requires_grad=True):
        super(BitwiseConv1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.biased = biased
        self.requires_grad = requires_grad
        weight_size = (output_channels, input_channels//self.groups, kernel_size)
        self.weight = init_weight(weight_size, requires_grad)
        self.bias = None
        if biased:
            self.bias = init_bias(output_channels, requires_grad)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = convert_param(self.weight, self.beta, self.mode)
        b = None
        if self.biased:
            b = convert_param(self.bias, self.beta, self.mode)
        return F.conv1d(x, w, b, stride=self.stride,
            padding=self.padding, groups=self.groups)

    def __repr__(self):
        return 'BitwiseConv1d(%d, %d, kernel_size=%d, stride=%d, padding=%d, self.groups=%d)' % \
            (self.input_channels, self.output_channels, self.kernel_size,
            self.stride, self.padding, self.groups)

class BitwiseConvTranspose1d(BitwiseAbstractClass):
    '''
    Issue: Almost copy paste of BitwiseConv1d. Parameter dimensions may be incorrect
    '''
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, biased=False, groups=1, requires_grad=True):
        super(BitwiseConvTranspose1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.biased = biased
        self.requires_grad = requires_grad
        weight_size = (input_channels, output_channels // groups, kernel_size)
        self.weight = init_weight(weight_size, requires_grad)
        self.bias = None
        if biased:
            self.bias = init_bias(output_channels, requires_grad)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = convert_param(self.weight, self.beta, self.mode)
        b = None
        if self.biased:
            b = convert_param(self.bias, self.beta, self.mode)
        return F.conv_transpose1d(x, w, b, stride=self.stride,
            padding=self.padding, groups=self.groups)

    def __repr__(self):
        return 'BitwiseConvTranspose1d(%d, %d, kernel_size=%d, stride=%d, padding=%d, self.groups=%d)' % \
            (self.input_channels, self.output_channels, self.kernel_size,
            self.stride, self.padding, self.groups)

class BinLinear(nn.Module):
    def __init__(self, input_size, output_size, biased=True):
        super(BinLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.biased = biased
        self.weight = init_weight((output_size, input_size), True)
        self.bias = None
        if biased:
            self.bias = init_bias(output_size, True)

    def forward(self, x):
        w = binarize(self.weight)
        b = None
        if biased:
            b = binarize(self.bias)
        return F.linear(x, w, b)

    def __repr__(self):
        return 'BinLinear(%d, %d)' % (self.input_size, self.output_size)

class BinConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size,
        biased=True, stride=1, padding=0, groups=1):
        super(BinConv1d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.biased = biased
        self.weight = init_weight(weight_size, True)
        self.bias = None
        if biased:
            self.bias = init_bias(output_channels, True)

    def forward(self, x):
        w = binarize(self.weight)
        b = None
        if self.biased:
            b = binarize(self.bias)
        return F.conv1d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)

    def __repr__(self):
        return 'BinConv1d(%d, %d, %d, stride=%d, padding=%d, self.groups=%d)' % (self.input_size,
            self.output_size, self.kernel_size,
            self.stride, self.padding, self.groups)

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size,
        biased=True, stride=1, padding=0, groups=1):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.biased = biased
        if isinstance(kernel_size, int):
            weight_size = (output_channels, input_channels//self.groups, kernel_size, kernel_size)
        else:
            weight_size = (output_channels, input_channels//self.groups, kernel_size[0], kernel_size[1])
        self.weight = init_weight(weight_size, True)
        self.bias = None
        if biased:
            self.bias = init_bias(output_channels, True)

    def forward(self, x):
        w = binarize(self.weight)
        b = None
        if self.biased:
            b = binarize(self.bias)
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)

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

class BitwiseDisperser(BitwiseAbstractClass):
    def __init__(self, channels_per_group, bit_groups, requires_grad=True):
        super(BitwiseDisperser, self).__init__()
        self.scale = 2**(bit_groups+1) + 1
        weight = torch.tensor(
            [2**(-channels_per_group+i+1)/self.scale for i in range(channels_per_group)],
            dtype=torch.float)
        weight = torch.cat([weight for _ in range(bit_groups)]).unsqueeze(1)
        bias = torch.tensor(2**(channels_per_group)*weight - 2/self.scale, dtype=torch.float)
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)
        self.bias = nn.Parameter(bias, requires_grad=requires_grad)
        self.biased = True
        self.mode = 'real'

    def forward(self, x):
        w = self.weight
        b = self.bias
        # w = convert_param(self.weight, 0, self.mode)
        # b = convert_param(self.bias, 0, self.mode)
        return torch.sin(self.scale*math.pi/2 * (x * w + b))

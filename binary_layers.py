import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

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

def init_params(size, biased=True, requires_grad=True):
    w = torch.empty(size)
    nn.init.xavier_uniform_(w)
    w = nn.Parameter(w, requires_grad=requires_grad)
    b = None
    if biased:
        b = torch.zeros(size[0])
        b = nn.Parameter(b, requires_grad=requires_grad)
    return w, b

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

class BitwiseLinear(nn.Module):
    '''
    Linear/affine operation using bitwise (Kim et al.) scheme
    '''
    def __init__(self, input_size, output_size):
        super(BitwiseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight, self.bias = init_params((output_size, input_size), True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = convert_param(self.weight, self.beta, self.mode)
        b = convert_param(self.bias, self.beta, self.mode)
        return F.linear(x, w, b)

    def update_beta(self, sparsity):
        if sparsity == 0:
            return
        w = self.weight.cpu().data.numpy()
        b = self.bias.cpu().data.numpy()
        params = np.abs(np.concatenate((w, np.expand_dims(b, axis=1)), axis=1))
        beta = torch.tensor(np.percentile(params, sparsity), dtype=self.weight.dtype,
            device=self.weight.device)
        self.beta = nn.Parameter(beta, requires_grad=False)

    def noisy(self):
        self.mode = 'noisy'
        self.weight = nn.Parameter(torch.tanh(self.weight), requires_grad=True)
        self.bias = nn.Parameter(torch.tanh(self.bias), requires_grad=True)

    def inference(self):
        self.mode = 'inference'
        self.weight = nn.Parameter(bitwise_params(self.weight, self.beta), requires_grad=False)
        self.bias = nn.Parameter(bitwise_params(self.bias, self.beta), requires_grad=False)

    def __repr__(self):
        return 'BitwiseLinear(%d, %d)' % (self.input_size, self.output_size)

class BitwiseConv1d(nn.Module):
    '''
    1D bitwise (Kim et. al) convolution
    '''
    def __init__(self, input_channels, output_channels, kernel_size,
        stride=1, padding=0, biased=False, groups=1):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        weight_size = (output_channels, input_channels//self.groups, kernel_size)
        self.weight, self.bias = init_params(weight_size, biased, True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        w = convert_param(self.weight, self.beta, self.mode)
        b = convert_param(self.bias, self.beta, self.mode)
        return F.conv1d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)

    def update_beta(self, sparsity):
        if sparsity == 0:
            return
        w = self.weight.cpu().data.numpy().reshape(-1)
        b = self.bias.cpu().data.numpy().reshape(-1)
        params = np.abs(np.concatenate((w, np.expand_dims(b, axis=1)), axis=1))
        beta = torch.tensor(np.percentile(params, sparsity), dtype=self.weight.dtype,
            device=self.weight.device)
        self.beta = nn.Parameter(beta, requires_grad=False)

    def noisy(self):
        self.mode = 'noisy'
        self.weight = nn.Parameter(torch.tanh(self.weight), requires_grad=True)
        self.bias = nn.Parameter(torch.tanh(self.bias), requires_grad=True)

    def inference(self):
        self.mode = 'inference'
        self.weight = nn.Parameter(bitwise_params(self.weight, self.beta), requires_grad=False)
        self.bias = nn.Parameter(bitwise_params(self.bias, self.beta), requires_grad=False)

class BinLinear(nn.Module):
    def __init__(self, input_size, output_size, biased=True):
        super(BinLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.biased = biased
        self.weight, self.bias = init_params((output_size, input_size), biased, True)

    def forward(self, x):
        w = binarize(self.weight)
        b = None
        if self.biased:
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
        weight_size = (output_channels, input_channels//self.groups, kernel_size)
        self.weight, self.bias = init_params(weight_size, biased, True)

    def forward(self, x):
        w = binarize(self.weight)
        b = None
        if self.biased:
            b = binarize(self.bias)
        return F.conv1d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)

    def __repr__(self):
        return 'BinLinear(%d, %d, %d, stride=%d, padding=%d, self.groups=%d)' % (self.input_size,
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
        self.weight, self.bias = init_params(weight_size, biased, True)

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

class Scaler(nn.Module):
    '''
    Batch normalization without shifting
    '''
    def __init__(self, num_features, requires_grad=True):
        super(Scaler, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features), requires_grad=requires_grad)

    def forward(self, x):
        return torch.abs(self.gamma) * x / (torch.std(x, dim=0) + 1e-5)

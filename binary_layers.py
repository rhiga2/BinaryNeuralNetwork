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
        return (x > beta).to(dtype=x.dtype, device=x.device) - (x < -beta).to(dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

bitwise_activation = BitwiseActivation.apply
bitwise_params = BitwiseParams.apply

class BitwiseLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(BitwiseLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        w = torch.empty(output_size, input_size)
        nn.init.xavier_uniform_(w)
        b = torch.zeros(output_size)
        self.weight = nn.Parameter(w, requires_grad=True)
        self.bias = nn.Parameter(b, requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0, dtype=self.weight.dtype), requires_grad=False)
        self.mode = 'real'

    def forward(self, x):
        if self.mode == 'real':
            w = torch.tanh(self.weight)
            b = torch.tanh(self.bias)
        elif self.mode == 'noisy':
            w = bitwise_params(self.weight, self.beta)
            b = bitwise_params(self.bias, self.beta)
        return F.linear(x, w, b)

    def update_beta(self, sparsity):
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

class BLRLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(BLRLinear, self).__init__()
        w = torch.empty(out_size, in_size)
        nn.init.xavier_uniform_(w)
        self.w = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        expect = torch.tanh(self.w)
        mean = F.linear(x, expect)
        var = F.linear(x**2, 1-expect**2)
        return mean, var

class BLRSampler(Function):
    def __init__(self, temp=0.1, eps=1e-5):
        self.temp = temp
        self.eps = eps

    @staticmethod
    def forward(self, mean, var):
        q = Normal(mean, var).cdf(0)
        U = torch.rand_like(mean)
        L = torch.log(U) - torch.log(1 - U)
        return torch.tanh((torch.log(1/(1-q+eps)-1+eps) + L)/temp)

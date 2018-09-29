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
        return torch.tensor(x > beta, dtype=x.dtype, device=x.device) - torch.tensor(x < -beta, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

bitwise_activation = BitwiseActivation.apply

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
        self.beta = 0
        self.mode = 'real'

    def forward(self, x):
        if self.mode == 'real':
            w = torch.tanh(self.weight)
            b = torch.tanh(self.bias)
        elif self.mode == 'noisy':
            w = BitwiseParams.apply(self.weight, self.beta)
            b = BitwiseParams.apply(self.bias, self.beta)
        return F.linear(x, w, b)

    def update_beta(self, sparsity):
        w = self.weight.cpu().data.numpy()
        b = self.bias.cpu().data.numpy()
        params = np.abs(np.concatenate((w, np.expand_dims(b, axis=1)), axis=1))
        self.beta = np.percentile(params, sparsity)

    def noisy(self):
        self.mode = 'noisy'
        self.weight = nn.Parameter(torch.tanh(self.weight), requires_grad=True)
        self.bias = nn.Parameter(torch.tanh(self.bias), requires_grad=True)

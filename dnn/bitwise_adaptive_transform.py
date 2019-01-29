import sys , os
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dnn.binary_layers as binary_layers
import dnn.bitwise_mlp as bitwise_mlp

def _haar_matrix(n):
    '''
    n is a power of 2
    Produce unnormalized haar
    '''
    assert n > 1
    if n == 2:
        return np.array([[1, 1], [1, -1]])
    prev_haar = haar_matrix(n // 2)
    prev_id = np.eye(n // 2)
    haar_top = np.kron(prev_haar, np.array([1, 1]))
    haar_bottom = np.kron(prev_id, np.array([1, -1]))
    return np.concatenate((haar_top, haar_bottom))

def haar_matrix(n):
    '''
    n is a power of 2
    '''
    haar = _haar_matrix(n)
    return haar / np.linalg.norm(haar, axis=1)

class BitwiseAdaptiveTransform(nn.Module):
    '''
    Adaptive transform network inspired by Minje Kim
    '''
    def __init__(self, kernel_size=256, stride=16, in_channels=1,
        out_channels=1, fc_sizes = [], dropout=0, sparsity=95,
        in_bin=binary_layers.identity, weight_bin=binary_layers.identity,
        use_gate=False, adaptive_scaling=True, activation=nn.ReLU(inplace=True),
        weight_init=None, autoencode=False):
        super(BitwiseAutoencoder, self).__init__()

        # Initialize adaptive front end
        self.kernel_size = kernel_size
        self.conv = binary_layers.BitwiseConv1d(1, kernel_size, kernel_size,
            stride=stride, padding=kernel_size, in_bin=binary_layers.identity,
            weight_bin=binary_layers.identity, adaptive_scaling=False,
            use_gate=False
        )

        self.batchnorm = nn.BatchNorm1d(kernel_size)
        self.activation = activation
        self.autoencode = autoencode

        if not autoencode:
            self.mlp = bitwise_mlp.BitwiseMLP(kernel_size, kernel_size,
                fc_sizes=[2048, 2048], dropout=dropout,
                activation=binary_layers.identity,
                in_bin=in_bin, weight_bin=weight_bin, use_batchnorm=True,
                adaptive_scaling=adaptive_scaling, use_gate=use_gate
            )

        # Initialize inverse of front end transform
        self.conv_transpose = binary_layers.BitwiseConvTranspose1d(
            kernel_size, 1, kernel_size, stride=stride,
            in_bin=binary_layers.identity,
            weight_bin=binary_layers.identity,
            adaptive_scaling=False,
            use_gate=False
        )

        # Initialize weights
        if weight_init == 'haar':
            haar = torch.FloatTensor(haar_matrix(kernel_size)).unsqueeze(1)
            self.conv.weight = nn.Parameter(haar, requires_grad=True)

            scale = stride / kernel_size
            self.conv_transpose.weight = nn.Parameter(scale * haar,
                requires_grad=True)

        elif weight_init == 'fft':
            fft = np.fft.fft(np.eye(kernel_size))
            real_fft = np.real(fft)
            im_fft = np.imag(fft)
            basis = torch.FloatTensor(np.concatenate([real_fft[:kernel_size//2], im_fft[:kernel_size//2]], axis=0))
            conv.weight = nn.Parameter(basis.unsqueeze(1), requires_grad=True)
            scale = stride / kernel_size
            invbasis = torch.t(scale * torch.pinverse(basis))
            invbasis = invbasis.contiguous().unsqueeze(1)
            conv_transpose.weight = nn.Parameter(invbasis, requires_grad=True)

        if use_gate:
            self.conv.gate.data[self.conv.weight == 0] = -self.conv.gate.data[self.conv.weight == 0]
            self.conv_transpose.gate.data[self.conv_transpose.weight == 0] = -self.conv_transpose.gate.data[self.conv_transpose.weight == 0]

        self.sparsity = sparsity

    def forward(self, x):
        '''
        Bitwise neural network forward
        * Input is a tensor of shape (batch, channels, time)
        * Output is a tensor of shape (batch, channels, time)
            - batch is the batch size
            - time is the sequence length
            - channels is the number of input channels = num bits in qad
        '''
        time = x.size(2)
        h = self.batchnorm(self.activation(self.conv(x)))

        if not self.autoencode:
            h_size = h.size()
            h = bitwise_mlp.flatten(h)
            h = self.mlp(h)
            h = bitwise_mlp.unflatten(h, h_size[0], h_size[2])

        h = self.conv_transpose(h)[:, :, self.kernel_size:time+self.kernel_size]
        return h

    def clip_weights(self):
        self.conv.clip_weights()
        self.conv_transpose.clip_weights()

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.sparsity == 0:
            return

        self.conv.update_betas(sparsity=args.sparsity)
        self.conv_transpose.update_betas(sparsity=args.sparsity)

    def load_partial_state_dict(self, state_dict):
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

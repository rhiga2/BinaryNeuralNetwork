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
        in_binactiv=None, w_binactiv=None, use_gate=False,
        weight_init=None, autoencode=False, bn_momentum=0.1):
        super(BitwiseAdaptiveTransform, self).__init__()

        # Initialize adaptive front end
        self.kernel_size = kernel_size
        self.in_binactiv = in_binactiv
        if in_binactiv is not None:
            self.in_binfunc = in_binactiv()
        self.conv = binary_layers.BitwiseConv1d(
            1, kernel_size, kernel_size,
            stride=stride, padding=kernel_size, in_binactiv=None,
            w_binactiv=None, use_gate=False, scale_weights=None,
            scale_activations=None
        )

        self.activation = nn.PReLU()
        self.batchnorm = nn.BatchNorm1d(kernel_size, momentum=bn_momentum)
        self.autoencode = autoencode

        if not autoencode:
            self.mlp = bitwise_mlp.BitwiseMLP(
                kernel_size, kernel_size,  fc_sizes=fc_sizes, dropout=dropout,
                bias=False, in_binactiv=in_binactiv, w_binactiv=w_binactiv,
                use_gate=use_gate, scale_weights=None, scale_activations=None,
                bn_momentum=bn_momentum
            )

        # Initialize inverse of front end transform
        self.conv_transpose = binary_layers.BitwiseConvTranspose1d(
            kernel_size, 1, kernel_size, stride=stride,
            in_binactiv=None, w_binactiv=None,
            use_gate=False,
            scale_weights=None, scale_activations=None
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
            self.conv.weight = nn.Parameter(basis.unsqueeze(1), requires_grad=True)
            scale = stride / kernel_size
            invbasis = torch.t(scale * torch.pinverse(basis))
            invbasis = invbasis.contiguous().unsqueeze(1)
            self.conv_transpose.weight = nn.Parameter(invbasis, requires_grad=True)

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
        spec = self.activation(self.conv(x))

        if not self.autoencode:
            h = spec
            if self.in_binfunc is not None:
                h = self.in_binfunc(self.batchnorm(h))

            spec_size = spec.size()
            h = bitwise_mlp.flatten(h)
            h = self.mlp(h)
            h = bitwise_mlp.unflatten(h, spec_size[0], spec_size[2])

            h = torch.sigmoid(h)
            spec = h * spec

        return self.conv_transpose(spec)[:, :, self.kernel_size:time+self.kernel_size]

    def clip_weights(self):
        if not self.autoencode:
            self.mlp.clip_weights()

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.sparsity == 0:
            return

        self.mlp.update_betas(sparsity=args.sparsity)

    def load_partial_state_dict(self, state_dict):
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

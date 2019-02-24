import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import scipy.signal as signal
from sklearn.cluster import KMeans
import argparse
import math

def unit_cube_normalize(x):
    return 2*(x / torch.max(torch.abs(x))) - 1


def quantize_and_disperse(mix_mag, quantizer, disperser):
    mix_mag = 2*(mix_mag / torch.max(torch.abs(mix_mag))) - 1
    qmag = quantizer(mix_mag)
    _, channels, frames = qmag.size()
    bmag = disperser(qmag.view(1, -1))
    bmag = bmag.squeeze(0).contiguous()
    bmag = torch.cat(torch.chunk(bmag, channels, dim=1), dim=0)
    bmag = bmag
    return bmag

def accumulate(x, quantizer, disperser):
    x = torch.FloatTensor(x)
    channels, frames =  x.size()
    x = torch.cat(torch.chunk(x, channels // disperser.num_bits, dim=0), dim=1).unsqueeze(0)
    x = disperser.inverse(x).view(-1, frames)
    x = quantizer.inverse(x)
    return x.numpy()

def uniform_qlevels(x, levels=16):
    '''
    x is flattened array of numbers
    '''
    xmax = np.max(x)
    xmin = np.min(x)
    centers = (xmax - xmin)*(np.arange(levels) + 0.5)/levels + xmin
    bins = get_bins(centers)
    return centers, bins

def kmeans_qlevels(x, levels=16):
    '''
    x is flattened array of numbers
    '''
    km = KMeans(n_clusters=levels)
    km.fit(np.expand_dims(x, axis=1))
    centers = np.sort(km.cluster_centers_.reshape(-1))
    bins = get_bins(centers)
    return centers, bins

def get_bins(centers):
    return (centers[:-1] + centers[1:])/2

def bucketize(x, bins):
    '''
    Quantize x according to bucket bins
    '''
    bucket_x = torch.zeros(x.size())
    for bin in bins:
        bucket_x[x >= bin] += 1
    return bucket_x.to(dtype=torch.long)

class Disperser(nn.Module):
    def __init__(self, num_bits, center=False):
        super(Disperser, self).__init__()
        self.num_bits = num_bits
        self.weight = nn.Parameter(torch.tensor([2**(-i) for i in range(num_bits)]).unsqueeze(1),
            requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(1 + self.weight/2),
            requires_grad=False)
        self.center = center

    def forward(self, x):
        '''
        x has shape (batch, features)
        return has shape (batch, number of bits, features)
        '''
        x = x.unsqueeze(1)
        x = torch.sign(torch.sin(math.pi * (x * self.weight + self.bias)))
        if not self.center:
            x = (x+1)/2
        return x

    def inverse(self, x):
        return torch.sum(x / self.weight, dim=1)

class OneHotTransform(nn.Module):
    def __init__(self, num_bits):
        super(OneHotTransform, self).__init__()
        self.num_bits = num_bits

    def forward(self, x):
        x = x.to(dtype=torch.long)
        y = torch.zeros(x.size(0), 2**self.num_bits, x.size(1))
        y.scatter_(1, x.unsqueeze(1), 1)
        return y

def mu_law(x, mu):
    return torch.sign(x)*torch.log(1 + mu*torch.abs(x))/(np.log(1 + mu))

def inverse_mu_law(x, mu):
    return torch.sign(x)/mu*((1 + mu)**torch.abs(x) - 1)

class Quantizer(nn.Module):
    def __init__(self, min=-1, delta=1/8, num_bits=4, mode='mu_law'):
        super(Quantizer, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits
        self.mode = mode

    def forward(self, x):
        '''
        x has shape (batch, features)
        return has shape (batch, features)
        '''
        if self.mode == 'mu_law':
            x = mu_law(x, 2**self.num_bits-1)
        elif self.mode == 'log':
            x = torch.log(x)
            x = unit_cube_normalize(x)
        x = (x - self.min) / self.delta-1
        x = torch.ceil(x)
        return torch.clamp(x, 0, 2**self.num_bits-1)

    def inverse(self, x):
        x = self.delta * (x + 0.5) + self.min
        if self.mode == 'mu_law':
            x = inverse_mu_law(x, 2**self.num_bits-1)
        elif self.mode == 'log':
            x = torch.exp(x)
        return x

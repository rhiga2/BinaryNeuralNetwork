import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import scipy.signal as signal
from sklearn.cluster import KMeans
import argparse
import math

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
    def __init__(self, min=-1, delta=1/8, num_bits=4, use_mu=True):
        super(Quantizer, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits
        self.use_mu = use_mu

    def forward(self, x):
        '''
        x has shape (batch, features)
        return has shape (batch, 2**num_bits, features)
        '''
        if self.use_mu:
            x = mu_law(x, 2**self.num_bits)
        x = (x - self.min) / self.delta-1
        x = torch.ceil(x)
        return torch.clamp(x, 0, 2**self.num_bits-1)

    def inverse(self, x):
        x = self.delta * (x + 0.5) + self.min
        if self.use_mu:
            x = inverse_mu_law(x, 2**self.num_bits)
        return x

class BinaryDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if data_dir[-1] != '/':
            self.data_dir += '/'
        flist = glob.glob(self.data_dir + 'binary_data*.npz')
        self.length = len(flist)

    def __getitem__(self, i):
        binary_fname = self.data_dir + ('binary_data%d.npz'%i)
        binary_data = np.load(binary_fname)
        return {'bmag': binary_data['bmag'], 'ibm': binary_data['ibm']}

    def __len__(self):
        return self.length

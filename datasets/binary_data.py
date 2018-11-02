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
    bucket_x = torch.zeros(x.size(), dtype=torch.long)
    for bin in bins:
        bucket_x[x >= bin] += 1
    return bucket_x

class Disperser(nn.Module):
    def __init__(self, num_bits, in_features, requires_grad=False, dtype=torch.float):
        super(Disperser, self).__init__()
        self.num_bits = num_bits
        weight = torch.tensor(
            [2**(-i) for i in range(num_bits)],
            dtype=dtype
        )
        weight = weight.unsqueeze(1).unsqueeze(1)
        bias = torch.tensor(2**(num_bits)*weight*in_features + 2, dtype=dtype)
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)
        self.bias = nn.Parameter(bias, requires_grad=requires_grad)

    def forward(self, x):
        '''
        x has shape (batch, channels, frames)
        return has shape (batch, number of bits, channels, frames)
        '''
        x = x.unsqueeze(1)
        return torch.sin(math.pi/2 * (x * self.weight + self.bias))

class Accumulator(nn.Module):
    def __init__(self, num_bits, requires_grad=False):
        super(Accumulator, self).__init__()
        self.num_bits = num_bits
        weight = torch.tensor(
            [2**i for i in range(num_bits)],
            dtype=torch.float
        )
        weight = weight.unsqueeze(1).unsqueeze(1)
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)

    def forward(self, x):
        '''
        x has shape (batch, number of bits, channels, frames)
        return has shape (batch, channels, frames)
        '''
        return torch.sum(x * self.weight, dim=1)

def quantize(x, min, delta, num_bits=4):
    x = (x - min) / delta
    bucket_x = torch.ceil(x)
    return torch.clamp(bucket_x, 0, 2**num_bits-1).to(dtype=torch.long)

def Quantize(nn.Module):
    def __init__(self, min, delta, num_bits=4, dtype=torch.float32):
        super(QuantizeDisperser, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits
        self.dtype = dtype

    def forward(self, x):
        return quantize(x, self.min, self.delta,
            num_bits=self.num_bits).to(self.dtype)

class QuantizeDisperser(nn.Module):
    def __init__(self, min, delta, num_bits=4, dtype=torch.float32):
        super(QuantizeDisperser, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits
        self.disperser = Disperser(num_bits, 1)
        self.dtype = dtype

    def forward(self, x):
        digit_x = quantize(x, self.min, self.delta,
            num_bits=self.num_bits).to(self.dtype)
        digit_x = 2*digit_x - 2**(self.num_bits) + 1
        return torch.sign(self.disperser(digit_x))

class DequantizeAccumulator(nn.Module):
    def __init__(self, min, delta, num_bits=4,
        dtype=torch.float32):
        super(DequantizeAccumulator, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits
        self.accumulator = Accumulator(num_bits, requires_grad=False)

    def forward(self, x):
        x = (x + 1)/2
        return self.delta*(self.accumulator(x) - 0.5) + self.min

def one_hot(x, num_bits=4):
    '''
    x has shape (batch, length)
    return has shape (batch, 2**num_bits, length)
    '''
    y = torch.zeros(x.size(0), 2**num_bits, x.size(1))
    y.scatter_(1, x.unsqueeze(1), 1)
    return y

class QuantizeOneHot(nn.Module):
    def __init__(self, min, delta, num_bits=4, dtype=torch.float32):
        super(QuantizeOneHot, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits

    def forward(self, x):
        '''
        x has shape (batch, length)
        return has shape (batch, 2**num_bits, length)
        '''
        digit_x = quantize(x, self.min, self.delta, num_bits=self.num_bits)
        return one_hot(digit_x, self.num_bits)

def make_binary_mask(premask, dtype=np.float):
    return np.array(premask > 0, dtype=dtype)

def stft(x, window='hann', nperseg=1024, noverlap=768):
    stft_x = signal.stft(x,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap)[2]
    real, imag = np.real(stft_x), np.imag(stft_x)
    mag = np.sqrt(real**2 + imag**2 + 1e-6)
    phase = stft_x / (mag + 1e-6)
    return mag, phase

def istft(mag, phase, window='hann', nperseg=1024, noverlap=768):
    stft_x = mag * phase
    x = signal.istft(stft_x, window=window, nperseg=nperseg, noverlap=noverlap)[1]
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

def crop_length(x, hop):
    return x[:len(x)//hop*hop]

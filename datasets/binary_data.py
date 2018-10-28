import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import scipy.signal as signal
from sklearn.cluster import KMeans
import argparse

unpacked = torch.tensor([
  [0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,0,1],
  [0,0,0,0,0,0,1,0],
  [0,0,0,0,0,0,1,1],
  [0,0,0,0,0,1,0,0],
  [0,0,0,0,0,1,0,1],
  [0,0,0,0,0,1,1,0],
  [0,0,0,0,0,1,1,1],
  [0,0,0,0,1,0,0,0],
  [0,0,0,0,1,0,0,1],
  [0,0,0,0,1,0,1,0],
  [0,0,0,0,1,0,1,1],
  [0,0,0,0,1,1,0,0],
  [0,0,0,0,1,1,0,1],
  [0,0,0,0,1,1,1,0],
  [0,0,0,0,1,1,1,1],
  [0,0,0,1,0,0,0,0],
  [0,0,0,1,0,0,0,1],
  [0,0,0,1,0,0,1,0],
  [0,0,0,1,0,0,1,1],
  [0,0,0,1,0,1,0,0],
  [0,0,0,1,0,1,0,1],
  [0,0,0,1,0,1,1,0],
  [0,0,0,1,0,1,1,1],
  [0,0,0,1,1,0,0,0],
  [0,0,0,1,1,0,0,1],
  [0,0,0,1,1,0,1,0],
  [0,0,0,1,1,0,1,1],
  [0,0,0,1,1,1,0,0],
  [0,0,0,1,1,1,0,1],
  [0,0,0,1,1,1,1,0],
  [0,0,0,1,1,1,1,1],
  [0,0,1,0,0,0,0,0],
  [0,0,1,0,0,0,0,1],
  [0,0,1,0,0,0,1,0],
  [0,0,1,0,0,0,1,1],
  [0,0,1,0,0,1,0,0],
  [0,0,1,0,0,1,0,1],
  [0,0,1,0,0,1,1,0],
  [0,0,1,0,0,1,1,1],
  [0,0,1,0,1,0,0,0],
  [0,0,1,0,1,0,0,1],
  [0,0,1,0,1,0,1,0],
  [0,0,1,0,1,0,1,1],
  [0,0,1,0,1,1,0,0],
  [0,0,1,0,1,1,0,1],
  [0,0,1,0,1,1,1,0],
  [0,0,1,0,1,1,1,1],
  [0,0,1,1,0,0,0,0],
  [0,0,1,1,0,0,0,1],
  [0,0,1,1,0,0,1,0],
  [0,0,1,1,0,0,1,1],
  [0,0,1,1,0,1,0,0],
  [0,0,1,1,0,1,0,1],
  [0,0,1,1,0,1,1,0],
  [0,0,1,1,0,1,1,1],
  [0,0,1,1,1,0,0,0],
  [0,0,1,1,1,0,0,1],
  [0,0,1,1,1,0,1,0],
  [0,0,1,1,1,0,1,1],
  [0,0,1,1,1,1,0,0],
  [0,0,1,1,1,1,0,1],
  [0,0,1,1,1,1,1,0],
  [0,0,1,1,1,1,1,1],
  [0,1,0,0,0,0,0,0],
  [0,1,0,0,0,0,0,1],
  [0,1,0,0,0,0,1,0],
  [0,1,0,0,0,0,1,1],
  [0,1,0,0,0,1,0,0],
  [0,1,0,0,0,1,0,1],
  [0,1,0,0,0,1,1,0],
  [0,1,0,0,0,1,1,1],
  [0,1,0,0,1,0,0,0],
  [0,1,0,0,1,0,0,1],
  [0,1,0,0,1,0,1,0],
  [0,1,0,0,1,0,1,1],
  [0,1,0,0,1,1,0,0],
  [0,1,0,0,1,1,0,1],
  [0,1,0,0,1,1,1,0],
  [0,1,0,0,1,1,1,1],
  [0,1,0,1,0,0,0,0],
  [0,1,0,1,0,0,0,1],
  [0,1,0,1,0,0,1,0],
  [0,1,0,1,0,0,1,1],
  [0,1,0,1,0,1,0,0],
  [0,1,0,1,0,1,0,1],
  [0,1,0,1,0,1,1,0],
  [0,1,0,1,0,1,1,1],
  [0,1,0,1,1,0,0,0],
  [0,1,0,1,1,0,0,1],
  [0,1,0,1,1,0,1,0],
  [0,1,0,1,1,0,1,1],
  [0,1,0,1,1,1,0,0],
  [0,1,0,1,1,1,0,1],
  [0,1,0,1,1,1,1,0],
  [0,1,0,1,1,1,1,1],
  [0,1,1,0,0,0,0,0],
  [0,1,1,0,0,0,0,1],
  [0,1,1,0,0,0,1,0],
  [0,1,1,0,0,0,1,1],
  [0,1,1,0,0,1,0,0],
  [0,1,1,0,0,1,0,1],
  [0,1,1,0,0,1,1,0],
  [0,1,1,0,0,1,1,1],
  [0,1,1,0,1,0,0,0],
  [0,1,1,0,1,0,0,1],
  [0,1,1,0,1,0,1,0],
  [0,1,1,0,1,0,1,1],
  [0,1,1,0,1,1,0,0],
  [0,1,1,0,1,1,0,1],
  [0,1,1,0,1,1,1,0],
  [0,1,1,0,1,1,1,1],
  [0,1,1,1,0,0,0,0],
  [0,1,1,1,0,0,0,1],
  [0,1,1,1,0,0,1,0],
  [0,1,1,1,0,0,1,1],
  [0,1,1,1,0,1,0,0],
  [0,1,1,1,0,1,0,1],
  [0,1,1,1,0,1,1,0],
  [0,1,1,1,0,1,1,1],
  [0,1,1,1,1,0,0,0],
  [0,1,1,1,1,0,0,1],
  [0,1,1,1,1,0,1,0],
  [0,1,1,1,1,0,1,1],
  [0,1,1,1,1,1,0,0],
  [0,1,1,1,1,1,0,1],
  [0,1,1,1,1,1,1,0],
  [0,1,1,1,1,1,1,1],
  [1,0,0,0,0,0,0,0],
  [1,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,1,0],
  [1,0,0,0,0,0,1,1],
  [1,0,0,0,0,1,0,0],
  [1,0,0,0,0,1,0,1],
  [1,0,0,0,0,1,1,0],
  [1,0,0,0,0,1,1,1],
  [1,0,0,0,1,0,0,0],
  [1,0,0,0,1,0,0,1],
  [1,0,0,0,1,0,1,0],
  [1,0,0,0,1,0,1,1],
  [1,0,0,0,1,1,0,0],
  [1,0,0,0,1,1,0,1],
  [1,0,0,0,1,1,1,0],
  [1,0,0,0,1,1,1,1],
  [1,0,0,1,0,0,0,0],
  [1,0,0,1,0,0,0,1],
  [1,0,0,1,0,0,1,0],
  [1,0,0,1,0,0,1,1],
  [1,0,0,1,0,1,0,0],
  [1,0,0,1,0,1,0,1],
  [1,0,0,1,0,1,1,0],
  [1,0,0,1,0,1,1,1],
  [1,0,0,1,1,0,0,0],
  [1,0,0,1,1,0,0,1],
  [1,0,0,1,1,0,1,0],
  [1,0,0,1,1,0,1,1],
  [1,0,0,1,1,1,0,0],
  [1,0,0,1,1,1,0,1],
  [1,0,0,1,1,1,1,0],
  [1,0,0,1,1,1,1,1],
  [1,0,1,0,0,0,0,0],
  [1,0,1,0,0,0,0,1],
  [1,0,1,0,0,0,1,0],
  [1,0,1,0,0,0,1,1],
  [1,0,1,0,0,1,0,0],
  [1,0,1,0,0,1,0,1],
  [1,0,1,0,0,1,1,0],
  [1,0,1,0,0,1,1,1],
  [1,0,1,0,1,0,0,0],
  [1,0,1,0,1,0,0,1],
  [1,0,1,0,1,0,1,0],
  [1,0,1,0,1,0,1,1],
  [1,0,1,0,1,1,0,0],
  [1,0,1,0,1,1,0,1],
  [1,0,1,0,1,1,1,0],
  [1,0,1,0,1,1,1,1],
  [1,0,1,1,0,0,0,0],
  [1,0,1,1,0,0,0,1],
  [1,0,1,1,0,0,1,0],
  [1,0,1,1,0,0,1,1],
  [1,0,1,1,0,1,0,0],
  [1,0,1,1,0,1,0,1],
  [1,0,1,1,0,1,1,0],
  [1,0,1,1,0,1,1,1],
  [1,0,1,1,1,0,0,0],
  [1,0,1,1,1,0,0,1],
  [1,0,1,1,1,0,1,0],
  [1,0,1,1,1,0,1,1],
  [1,0,1,1,1,1,0,0],
  [1,0,1,1,1,1,0,1],
  [1,0,1,1,1,1,1,0],
  [1,0,1,1,1,1,1,1],
  [1,1,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,1],
  [1,1,0,0,0,0,1,0],
  [1,1,0,0,0,0,1,1],
  [1,1,0,0,0,1,0,0],
  [1,1,0,0,0,1,0,1],
  [1,1,0,0,0,1,1,0],
  [1,1,0,0,0,1,1,1],
  [1,1,0,0,1,0,0,0],
  [1,1,0,0,1,0,0,1],
  [1,1,0,0,1,0,1,0],
  [1,1,0,0,1,0,1,1],
  [1,1,0,0,1,1,0,0],
  [1,1,0,0,1,1,0,1],
  [1,1,0,0,1,1,1,0],
  [1,1,0,0,1,1,1,1],
  [1,1,0,1,0,0,0,0],
  [1,1,0,1,0,0,0,1],
  [1,1,0,1,0,0,1,0],
  [1,1,0,1,0,0,1,1],
  [1,1,0,1,0,1,0,0],
  [1,1,0,1,0,1,0,1],
  [1,1,0,1,0,1,1,0],
  [1,1,0,1,0,1,1,1],
  [1,1,0,1,1,0,0,0],
  [1,1,0,1,1,0,0,1],
  [1,1,0,1,1,0,1,0],
  [1,1,0,1,1,0,1,1],
  [1,1,0,1,1,1,0,0],
  [1,1,0,1,1,1,0,1],
  [1,1,0,1,1,1,1,0],
  [1,1,0,1,1,1,1,1],
  [1,1,1,0,0,0,0,0],
  [1,1,1,0,0,0,0,1],
  [1,1,1,0,0,0,1,0],
  [1,1,1,0,0,0,1,1],
  [1,1,1,0,0,1,0,0],
  [1,1,1,0,0,1,0,1],
  [1,1,1,0,0,1,1,0],
  [1,1,1,0,0,1,1,1],
  [1,1,1,0,1,0,0,0],
  [1,1,1,0,1,0,0,1],
  [1,1,1,0,1,0,1,0],
  [1,1,1,0,1,0,1,1],
  [1,1,1,0,1,1,0,0],
  [1,1,1,0,1,1,0,1],
  [1,1,1,0,1,1,1,0],
  [1,1,1,0,1,1,1,1],
  [1,1,1,1,0,0,0,0],
  [1,1,1,1,0,0,0,1],
  [1,1,1,1,0,0,1,0],
  [1,1,1,1,0,0,1,1],
  [1,1,1,1,0,1,0,0],
  [1,1,1,1,0,1,0,1],
  [1,1,1,1,0,1,1,0],
  [1,1,1,1,0,1,1,1],
  [1,1,1,1,1,0,0,0],
  [1,1,1,1,1,0,0,1],
  [1,1,1,1,1,0,1,0],
  [1,1,1,1,1,0,1,1],
  [1,1,1,1,1,1,0,0],
  [1,1,1,1,1,1,0,1],
  [1,1,1,1,1,1,1,0],
  [1,1,1,1,1,1,1,1]
], dtype=torch.uint8)

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

def quantize(x, min, delta, num_bins=16):
    x = (x - min) / delta
    bucket_x = torch.ceil(x)
    return torch.clamp(bucket_x, 0, num_bins-1).to(dtype=torch.long)

class QuantizeDisperser(nn.Module):
    def __init__(self, min, delta, num_bits=4, device=torch.device('cpu'),
        dtype=torch.float32):
        super(QuantizeDisperser, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits
        self.unpacked = unpacked.to(device=device, dtype=dtype)

    def forward(self, x):
        digit_x = quantize(x, self.min, self.delta, num_bins=2**self.num_bits).view(-1)
        qad = torch.index_select(self.unpacked, 0, digit_x)[:, -self.num_bits:]
        return qad.contiguous().view(x.size(0), x.size(1), -1).permute(0, 2, 1)

class DequantizeAccumulator(nn.Module):
    def __init__(self, min, delta, num_bits=4, device=torch.device('cpu'),
        dtype=torch.float32):
        super(DequantizeAccumulator, self).__init__()
        self.min = min
        self.delta = delta
        self.num_bits = num_bits
        self.weights = torch.tensor([2**(num_bits-i-1) for i in range(num_bits)],
            dtype=dtype, device=device).unsqueeze(1)

    def forward(self, x):
        return self.delta*(torch.sum(x * self.weights, dim=1) - 0.5) + self.min

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

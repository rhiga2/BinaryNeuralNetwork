import numpy as np
import torch
import torch.nn.functional as F
import glob
import scipy.signal as signal
from sklearn.cluster import KMeans
import argparse

unpacked = torch.tensor(np.load('unpacked.npy'), dtype=torch.uint8)

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

def quantize_and_disperse(x, min, delta, num_bits=4):
    '''
    Quantize and disperse x into a binary tensor
    x is shape (batch, length)
    returns output of shape (batch, num_bits, length)
    '''
    digit_x = quantize(x, min, delta, num_bins=2**num_bits).view(-1)
    qad = torch.index_select(unpacked, 0, digit_x)[:, -num_bits:]
    return qad.view(x.size(0), x.size(1), -1).permute(0, 2, 1).contiguous()

def dequantize_and_accumulate(x, min, delta, num_bits=4):
    '''
    Dequantize and accumulate a binary tensor
    '''
    weights = torch.tensor([2**(num_bits-i-1) for i in range(num_bits)], dtype=x.dtype).unsqueeze(1)
    return delta*(torch.sum(x * weights, dim=1) - 0.5) + min

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

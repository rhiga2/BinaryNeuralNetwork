import numpy as np
import scipy.signal as signal
from .two_source_mixture import *
from .sinusoidal_data import *
from sklearn.cluster import KMeans
import argparse

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

def binarize(x, bins, num_bits=4):
    '''
    x is shape (F, T)
    F = frequency range
    T = time range
    '''
    assert len(bins)+1 == 2**num_bits
    digit_x = np.digitize(x, bins).astype(np.uint8)
    binary_x = []
    for i in range(digit_x.shape[0]):
        bits = np.unpackbits(np.expand_dims(digit_x[i], axis=0), axis=0)[-num_bits:]
        binary_x.append(bits)
    return np.concatenate(binary_x, axis=0)

def quantize(x, bins, centers):
    digit_x = np.digitize(x, bins).astype(np.int)
    qx = centers[digit_x] # qx = quantized x
    return qx

def make_binary_mask(premask, dtype=np.float):
    return np.array(premask > 0, dtype=dtype)

def stft(x, window='hann', nperseg=1024, noverlap=768):
    stft_x = signal.stft(x,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap)[2]
    real, imag = np.real(stft_x), np.imag(stft_x)
    mag = np.sqrt(real**2 + imag**2)
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

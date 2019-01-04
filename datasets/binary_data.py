import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.make_data import *
from datasets.quantized_data import *
import argparse

def make_binary_mask(x):
    return x > 0

def quantize_and_disperse(mix_mag, quantizer, disperser):
    mix_mag = torch.FloatTensor(mix_mag / np.max(np.abs(mix_mag))).unsqueeze(0)
    qmag = quantizer(mix_mag)
    _, channels, frames = qmag.size()
    bmag = disperser(qmag.view(1, -1))
    bmag = bmag.squeeze(0).contiguous()
    bmag = torch.cat(torch.chunk(bmag, channels, dim=1), dim=0)
    bmag = bmag.numpy()
    return bmag

def accumulate(x, quantizer, disperser):
    x = torch.FloatTensor(x)
    channels, frames =  x.size()
    x = torch.cat(torch.chunk(x, channels // disperser.num_bits, dim=0), dim=1).unsqueeze(0)
    x = disperser.inverse(x).view(-1, frames)
    x = quantizer.inverse(x)
    return x.numpy()

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

def make_binary_data(batchsize, toy=False):
    if toy:
        trainset = DatasetFromDirectory('/media/data/binary_audio/toy_train',
            template='binary_data*.npz')
        valset = DatasetFromDirectory('/media/data/binary_audio/toy_val',
            template='binary_data*.npz')
        rawset = DatasetFromDirectory('/media/data/binary_audio/toy_val',
            template='raw_data*.npz')
    else:
        trainset = DatasetFromDirectory('/media/data/binary_audio/train',
            template='binary_data*.npz')
        valset = DatasetFromDirectory('/media/data/binary_audio/val',
            template='binary_data*.npz')
        rawset = DatasetFromDirectory('/media/data/binary_audio/val',
            template='raw_data*.npz')
    collate = lambda x : collate_and_trim(x, axis=1)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    return train_dl, valset, rawset

class DatasetFromDirectory():
    def __init__(self, data_dir, template='binary_data*.npz'):
        self.data_dir = data_dir
        if data_dir[-1] != '/':
            self.data_dir += '/'
        flist = glob.glob(self.data_dir + template)
        self.template = template.replace('*', '%d')
        self.length = len(flist)

    def __getitem__(self, i):
        fname = self.data_dir + (self.template%i)
        data = np.load(fname)
        return data

    def __len__(self):
        return self.length

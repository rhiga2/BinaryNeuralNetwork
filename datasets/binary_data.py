import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets.two_source_mixture as two_source_mixture
import argparse
import datasets.utils as utils
import scipy.signal as signal
import numpy as np

def make_binary_mask(x):
    return x > 0

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

def get_binary_data(batchsize, toy=False):
    if toy:
        trainset = utils.DatasetFromDirectory(
            '/media/data/binary_audio/toy_train',
            template='binary_data*.npz')
        valset = utils.DatasetFromDirectory(
            '/media/data/binary_audio/toy_val',
            template='binary_data*.npz')
        rawset = utils.DatasetFromDirectory(
            '/media/data/binary_audio/toy_val',
            template='raw_data*.npz')
    else:
        trainset = utils.DatasetFromDirectory(
            '/media/data/binary_audio/train',
            template='binary_data*.npz')
        valset = utils.DatasetFromDirectory(
            '/media/data/binary_audio/val',
            template='binary_data*.npz')
        rawset = utils.DatasetFromDirectory(
            '/media/data/binary_audio/val',
            template='raw_data*.npz')
    collate = lambda x : utils.collate_and_trim(x, axis=1)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    return train_dl, valset, rawset

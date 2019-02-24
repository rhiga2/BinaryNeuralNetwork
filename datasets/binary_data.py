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
    raw_collate = lambda x : utils.collate_and_trim(x, axis=0)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    val_dl = DataLoader(valset, batch_size=batchsize, shuffle=False,
        collate_fn=collate)
    raw_dl = DataLoader(rawset, batch_size=batchsize, shuffle=False,
        collate_fn=raw_collate)
    return train_dl, val_dl, raw_dl

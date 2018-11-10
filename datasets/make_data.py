import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.two_source_mixture import *
import argparse

def make_mixture_set(hop=256, toy=False, max_length=32000):
    speaker_path = '/media/data/timit-wav/train'
    targets = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
        'dr1/fdaw0', 'dr1/fjsp0','dr1/fsjk1', 'dr1/fvmh0']
        # 'dr1/fsma0', 'dr1/ftbr0']
    train_speeches, val_speeches = get_speech_files(speaker_path, targets, num_train=7)

    if toy:
        noise_path = '/media/data/Nonspeech'
        interferences = ['n81.wav', # chimes
                         'n97.wav', # eating chips
                         'n21.wav', # motorcycle
                         'n46.wav', # ocean
                         'n47.wav', # birds
                         'n55.wav', # cicadas?
                         'n59.wav', # jungle?
                         ]
        train_noises, val_noises = get_noise_files(noise_path, interferences)
        trainset = TwoSourceMixtureDataset(train_speeches, train_noises, hop=hop,
            max_length=max_length)
        valset = TwoSourceMixtureDataset(val_speeches, val_noises, hop=hop,
            max_length=max_length)
    else:
        interferences = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0']
            # 'dr1/mwad0', 'dr1/mwar0']
        train_noises, val_noises = get_speech_files(speaker_path, interferences, num_train=7)
        trainset = TwoSourceMixtureDataset(train_speeches, train_noises, hop=hop,
            max_length=max_length)
        valset = TwoSourceMixtureDataset(val_speeches, val_noises, hop=hop,
            max_length=max_length)
    return trainset, valset

def make_data(batchsize, hop=256, toy=False):
    '''
    Make two mixture dataset
    '''
    trainset, valset = make_mixture_set(hop=hop, toy=toy)
    collate = lambda x: collate_and_trim(x, axis=0)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate)
    return train_dl, val_dl

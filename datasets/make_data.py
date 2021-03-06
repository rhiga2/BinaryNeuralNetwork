import sys , os
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal
import datasets.two_source_mixture as two_source_mixture
import argparse

def make_mixture_set(hop=256, toy=False, max_duration=None, transform=None,
    timit=False, random_snr=False):

    if timit:
        directory = '/media/data/timit-wav/train'
        targets = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
            'dr1/fdaw0', 'dr1/fjsp0','dr1/fsjk1', 'dr1/fvmh0',
            'dr1/fsma0', 'dr1/ftbr0']

        inters = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0',
            'dr1/mwad0', 'dr1/mwar0', 'dr1/mtrr0', 'dr1/mtjs0',
            'dr1/mcpm0', 'dr1/mmrp0']
    else:
        directory = '/media/data/wsj/wsj0/11-1.1/wsj0/si_tr_s/'
        targets = ['011', '014', '016', '017', '018', '019', '023', '027', '028', '1a', '1b']
        inters = ['012', '013', '015', '020', '021', '022', '024', '025', '026', '029']

    train_speech, val_speech, test_speech = two_source_mixture.get_speech_files(directory, targets,
                                                                                train_percent=0.6,
                                                                                val_percent=0.2,
                                                                                max_utterances=20)
    train_inter, val_inter, test_inter = two_source_mixture.get_speech_files(directory, inters,
                                                                             train_percent=0.6,
                                                                             val_percent=0.2,
                                                                             max_utterances=20)

    trainset = two_source_mixture.TwoSourceMixtureDataset(train_speech, train_inter, hop=hop,
        max_duration=max_duration, transform=transform, random_snr=random_snr)
    valset = two_source_mixture.TwoSourceMixtureDataset(val_speech, val_inter, hop=hop,
        max_duration=max_duration, transform=transform, random_snr=random_snr)
    testset = two_source_mixture.TwoSourceMixtureDataset(test_speech, test_inter, hop=hop,
        max_duration=max_duration, transform=transform, random_snr=random_snr)

    return trainset, valset, testset

def make_data(batchsize, hop=256, toy=False, max_duration=2, transform=None):
    '''
    Make two mixture dataset
    '''
    trainset, valset, testset = make_mixture_set(hop=hop, toy=toy,
        max_duration=max_duration, transform=transform)
    collate = lambda x: utils.collate_and_trim(x, axis=0)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate)
    test_dl = DataLoader(testset, batch_size=batchsize,
        collate_fn=collate)
    return train_dl, val_dl, test_dl

if __name__=='__main__':
    decimate = lambda x : signal.decimate(x, 2)
    trainset, valset, testset = make_mixture_set(hop=160, max_duration=2,
        transform=decimate, random_snr=False)
    print('Training Size: ', len(trainset))
    print('Validation Size: ', len(valset))
    print('Testing Size: ', len(testset))
    dataset_dir = '/media/data/wsj_mix/decimated2/'
    dirs = ['train/', 'val/', 'test/']
    datasets = [trainset, valset, testset]
    for dir, dataset in zip(dirs, datasets):
        for i, data in enumerate(dataset):
            fname = dataset_dir + dir + 'sample{}.npz'.format(i)
            np.savez(
                fname,
                mixture=data['mixture'],
                target=data['target'],
                interference=data['interference']
            )

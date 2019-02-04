import numpy as np
import glob
import torch
import itertools
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.signal as signal
import random
import soundfile as sf

class TwoSourceMixtureDataset(Dataset):
    def __init__(self, speeches, interferences, fs=16000,
        transform=None, hop=None, max_duration=None):
        self.fs = fs
        self.mixes = list(itertools.product(speeches, interferences))
        self.transform = transform
        self.hop = hop
        self.max_length = 0
        if max_duration:
            self.max_length = max_duration * fs

    def __len__(self):
        return len(self.mixes)

    def _getmix(self, sigf, interf):
        # Read files
        sig, _ = sf.read(sigf)
        if len(sig.shape) != 1:
            sig = np.mean(sig, axis=1)

        if self.hop:
            sig = sig[:len(sig)//self.hop*self.hop]

        if len(sig) > self.max_length:
            start = np.random.randint(len(sig) - self.max_length)
            sig = sig[start:self.max_length+start]

        inter, _ = sf.read(interf, fill_value=0)
        if len(inter.shape) != 1:
            inter = np.mean(inter, axis=1)

        sig_len = len(sig)
        if len(inter) > sig_len:
            start = np.random.randint(len(inter) - sig_len)
            inter = inter[start:sig_len+start]
        elif len(inter) < sig_len:
            inter = np.pad(inter, (0, sig_len - len(inter)))

        # normalize and mix signals
        sig = sig / (np.std(sig) + 1e-5)
        inter = inter / (np.std(inter) + 1e-5)
        mix = sig + inter
        mix = mix / np.max(np.abs(mix)) # for quantization purposes
        sample = {'mixture': mix, 'target': sig, 'interference': inter}

        if self.transform:
            sample = {key: self.transform(value) for key, value in sample.items()}

        return sample

    def __getitem__(self, i):
        sigf, interf = self.mixes[i] # get sig and interference file
        return self._getmix(sigf, interf)

def get_speech_files(speaker_path, speakers, train_percent=0.6, val_percent=0.2,
    max_utterances=None):
    '''
    Assumes that speech files are organized in speaker_path/speakers/*.wav
    speaker path: directory of speakers
    speakers: list of speaker directories
    train_percent: percent of utterances to put in training set
    val_percent: percent of utterances to put in validation set
    max_utterances: number of utterances to put in every set per speaker
    '''
    assert train_percent + val_percent <= 1
    if speaker_path[-1] != '/':
        speaker_path += '/'
    train_speeches = []
    val_speeches = []
    test_speeches = []

    for speaker in speakers:
        if speaker[-1] != '/':
            speaker += '/'
        files = glob.glob(speaker_path + speaker + '*.wav')
        max_sentences = len(files)

        if max_utterances is not None:
            if max_utterances < len(files):
                max_sentences = max_utterances

        num_train = int(train_percent * max_sentences)
        num_val = int(val_percent * max_sentences)
        train_speeches.extend(files[:num_train])
        val_speeches.extend(files[num_train:num_train+num_val])
        test_speeches.extend(files[num_train+num_val:max_sentences])
    return train_speeches, val_speeches, test_speeches

def get_noise_files(noise_path, noises, train_percent=0.6, val_percent=0.2):
    assert train_percent + val_percent <= 1
    if noise_path[-1] != '/':
        noise_path += '/'
    noises = [noise_path + noise for noise in noises]
    train_noises = noises
    val_noises = noises
    if train_percent != 1:
        num_train = len(train_percent * len(noises))
        num_val + len(val_percent * len(noises))
        train_noises = noises[:num_train]
        val_noises = noises[num_train:num_train+num_val]
        test_noises = noises[num_train+num_val:]
    return train_noises, val_noises, test_noises

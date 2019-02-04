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
        random_start=True, transform=None, hop=None, max_duration=None):
        self.fs = fs
        self.mixes = list(itertools.product(speeches, interferences))
        self.transform = transform
        self.hop = hop
        self.max_length = None
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

        inter, _ = sf.read(interf, frames=len(sig), fill_value=0)
        if len(inter.shape) != 1:
            inter = np.mean(inter, axis=1)

        if self.max_length:
            if len(sig) > self.max_length:
                start = np.random.randint(len(sig) - self.max_length)
                sig = sig[start:self.max_length+start]
            if len(inter) > self.max_length:
                start = np.random.randint(len(inter) - self.max_length)
                inter = inter[start:self.max_length+start]

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

def collate_and_trim(batch, axis=0, hop=1, dtype=torch.float):
    keys = list(batch[0].keys())
    outbatch = {key: [] for key in keys}
    min_length = min([sample[keys[0]].shape[axis] for sample in batch])
    for sample in batch:
        length = sample[keys[0]].shape[axis]
        start = (length - min_length) // 2
        for key in keys:
            indices = range(start, start+min_length)
            outbatch[key].append(sample[key].take(indices=indices, axis=axis))

    outbatch = {key: torch.as_tensor(np.stack(values, axis=0), dtype=dtype) for key, values in outbatch.items()}
    return outbatch

def get_speech_files(speaker_path, speakers, train_percent=0.6, val_percent=0.2):
    '''
    Assumes that speech files are organized in speaker_path/speakers/*.wav
    speaker path: directory of speakers
    speakers: list of speaker directories
    num_train: number of utterances to put in training set
    num_val: number of utterances to put in validation set
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
        num_train = int(train_percent * len(files))
        num_val = int(val_percent * len(files))
        train_speeches.extend(files[:num_train])
        end = num_train + num_val
        val_speeches.extend(files[num_train:end])
        test_speeches.extend(files[end:])
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

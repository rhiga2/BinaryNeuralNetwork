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
    def __init__(self, speeches, interferences, fs=16000, snr=0,
        random_start=True, transform=None):
        self.fs = fs
        self.snr = np.power(10, snr/20)
        self.mixes = list(itertools.product(speeches, interferences))
        self.transform = transform

    def __len__(self):
        return len(self.mixes)

    def _getmix(self, sigf, interf):
        # Read files
        sig, _ = sf.read(sigf)
        if len(sig.shape) != 1:
            sig = np.mean(sig, axis=1)
        inter, _ = sf.read(interf, frames=sig.shape[0], fill_value=0)
        if len(inter.shape) != 1:
            inter = np.mean(inter, axis=1)

        # normalize and mix signals
        sig = sig / np.std(sig)
        inter = inter / np.std(inter)
        mix = sig + inter
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
    min_length = min_length // hop * hop
    for sample in batch:
        length = sample[keys[0]].shape[axis]
        start = (length - min_length) // 2
        for key in keys:
            indices = range(start, start+min_length)
            outbatch[key].append(sample[key].take(indices=indices, axis=axis))

    outbatch = {key: torch.as_tensor(np.stack(values, axis=0), dtype=dtype) for key, values in outbatch.items()}
    return outbatch

def get_speech_files(speaker_path, speakers=[], num_train=8):
    assert num_train <= 10 # Assume each speaker has 10 sentences
    if speaker_path[-1] != '/':
        speaker_path += '/'
    train_speeches = []
    val_speeches = []

    for speaker in speakers:
        if speaker[-1] != '/':
            speaker += '/'
        files = glob.glob(speaker_path + speaker + '*.wav')
        # rnadom.shuffle(files)
        train_speeches.extend(files[:num_train])
        val_speeches.extend(files[num_train:])
    return train_speeches, val_speeches

def get_noise_files(noise_path, noises, num_train=2):
    assert num_train <= len(noises)
    if noise_path[-1] != '/':
        noise_path += '/'
    noises = [noise_path + noise for noise in noises]
    # random.shuffle(noises)
    train_noises = noises[:num_train]
    val_noises = noises[num_train:]
    return train_noises, val_noises

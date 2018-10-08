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
        random_start=True, transform=None, hop=None):
        self.fs = fs
        self.snr = np.power(10, snr/20)
        self.mixes = list(itertools.product(speeches, interferences))
        self.transform = transform
        self.hop = hop

    def __len__(self):
        return len(self.mixes)

    def _getmix(self, sigf, interf):
        # Read files
        sig, _ = sf.read(sigf)
        if len(sig.shape) != 1:
            sig = np.mean(sig, axis=1)

        if self.hop:
            sig = sig[:len(sig)//self.hop*self.hop]

        inter, _ = sf.read(interf, frames=sig.shape[0], fill_value=0)
        if len(inter.shape) != 1:
            inter = np.mean(inter, axis=1)

        # normalize and mix signals
        sig = sig / np.std(sig)
        inter = inter / np.std(inter)
        mix = sig + (1 / self.snr) * inter
        sample = {'mixture': mix, 'target': sig, 'interference': inter}

        if self.transform:
            sample = {key: self.transform(value) for key, value in sample.items()}

        return sample

    def __getitem__(self, i):
        sigf, interf = self.mixes[i] # get sig and interference file
        return self._getmix(sigf, interf)

class SineSpeechData(Dataset):
    def __init__(self, speeches, num_sines, fs = 16000, sig_range = [2000, 4000], snr = 0,
        hop=None):
        self.fs = fs
        self.hop = hop
        self.num_sines = num_sines
        self.snr = np.power(10, snr/20)
        self.noise_freqs = np.random.uniform(noise_range[0], noise_range[1], num_sines)
        self.noise_phases = np.random.random(num_sines) * 2*np.pi
        sine_params = list(zip(self.noise_freqs, self.noise_phases))
        self.mixes = list(itertools.product(speeches, sine_params))

    def __getitem__(self, key):
        # return (mixture, target, interference)
        speech_file, sine_params = self.mixes[key]
        freq, phase = sine_params
        speech, _ = sf.read(sigf)
        if len(sig.shape) != 1:
            speech = np.mean(speech, axis=1)

        if self.hop:
            speech = speech[:len(speech)//self.hop*self.hop]

        time = np.arange(len(speech)) * 1 / self.fs
        noise = np.sin(2*np.pi*freq * self.time + phase)
        speech = speech / np.std(sig1)
        noise = noise / np.std(sig2)
        mix = speech + (1 / self.snr) * noise
        mix = mix / np.std(mix)
        return {'mixture': mix, 'target': speech, 'interference': noise}

    def __len__(self):
        return self.size

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

def get_speech_files(speaker_path, speakers, num_train=8):
    assert num_train <= 10 # Assume each speaker has 10 sentences
    if speaker_path[-1] != '/':
        speaker_path += '/'
    train_speeches = []
    val_speeches = []

    for speaker in speakers:
        if speaker[-1] != '/':
            speaker += '/'
        files = glob.glob(speaker_path + speaker + '*.wav')
        train_speeches.extend(files[:num_train])
        val_speeches.extend(files[num_train:])
    return train_speeches, val_speeches

def get_noise_files(noise_path, noises, num_train=None):
    if noise_path[-1] != '/':
        noise_path += '/'
    noises = [noise_path + noise for noise in noises]
    train_noises = noises
    val_noises = noises
    if num_train:
        train_noises = noises[:num_train]
        val_noises = noises[num_train:]
    return train_noises, val_noises

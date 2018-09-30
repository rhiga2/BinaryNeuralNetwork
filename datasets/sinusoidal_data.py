import torch
from torch.utils.data import Dataset
import numpy as np

class SinusoidDataset(Dataset):
    def __init__(self, size, length, samp_freq = 16000, sig_range = [0, 8000],
        noise_range = [0, 8000], snr = 0):
        assert 2 * sig_range[1] <= samp_freq
        assert 2 * noise_range[1] <= samp_freq
        self.size = size
        self.length = length
        self.time = np.arange(length) * 1 / samp_freq
        self.snr = np.power(10, snr/20)
        self.signal_freqs = np.random.uniform(sig_range[0], sig_range[1], self.size)
        self.noise_freqs = np.random.uniform(noise_range[0], noise_range[1], self.size)
        self.signal_phases = np.random.random(self.size) * 2*np.pi
        self.noise_phases = np.random.random(self.size) * 2*np.pi

    def __getitem__(self, key):
        # return (mixture, target, interference)
        sig1 = np.sin(2*np.pi*self.signal_freqs[key] * self.time + self.signal_phases[key])
        sig2 = np.sin(2*np.pi*self.noise_freqs[key] * self.time + self.noise_phases[key])
        sig1 = sig1 / np.std(sig1)
        sig2 = sig2 / np.std(sig2)
        mix = sig1 + (1 / self.snr) * sig2
        mix = mix / np.std(mix)
        return {'mixture': mix, 'target': sig1, 'interference': sig2}

    def __len__(self):
        return self.size

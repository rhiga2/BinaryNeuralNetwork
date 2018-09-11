import numpy as np
import librosa
import glob
import torch
import itertools
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.signal as signal
import random
import stft

class TwoSourceMixtureDataset(Dataset):
    def __init__(self, speeches, interferences, fs=16000, snr=0,
        random_start=True, transform=None, device=torch.device('cpu'),
        dtype=torch.float):
        self.fs = fs
        self.snr = np.power(10, snr/20)
        self.random_start = random_start
        self.mixes = list(itertools.product(speeches, interferences))
        self.transform = transform
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.mixes)

    def _getmix(self, sigf, interf):
        sig_duration = librosa.core.get_duration(filename=sigf)
        inter_duration = librosa.core.get_duration(filename=interf)
        duration = min(sig_duration, inter_duration)

        sig_offset = 0
        inter_offset = 0
        if self.random_start:
            sig_offset =  np.random.random() * (sig_duration - duration)
            inter_offset = np.random.random() * (inter_duration - duration)

        # Read files
        sig, _ = librosa.core.load(sigf, sr=self.fs, mono=True,
                                      duration=duration,
                                      offset=sig_offset,
                                      res_type='kaiser_fast')
        inter, _ = librosa.core.load(interf, sr=self.fs, mono=True,
                                      duration=duration, offset=inter_offset,
                                      res_type='kaiser_fast')

        # normalize and mix signals
        sig = torch.tensor(sig / np.std(sig), dtype=self.dtype, device=self.device)
        inter = torch.tensor(inter / np.std(inter), dtype=self.dtype, device=self.device)
        mix = sig + inter
        sample = {'mixture': mix, 'target': sig, 'interference': inter}

        if self.transform:
            sample = {key: self.transform(value) for key, value in sample.items()}

        return sample

    def __getitem__(self, i):
        sigf, interf = self.mixes[i] # get sig and interference file
        return self._getmix(sigf, interf)


class TwoSourceSpectrogramDataset(Dataset):
    def __init__(self, speeches, interferences, fs=16000, snr=0,
        random_start=True, transform=None, device=torch.device('cpu'),
        fft_size=1024, hop=256):
        self.mixture_set = TwoSourceMixtureDataset(speeches, interferences,
            fs=fs, snr=snr, random_start=random_start, transform=transform,
            device=device, dtype=torch.float)
        self.stft = stft.STFT(fft_size, hop, window_type='hann').to(device)
        self.fft_size = 1024
        self.hop = 256

    def __getitem__(self, i):
        sample = self.mixture_set[i]
        output = {}
        for key in sample:
            x = x.unsqueeze(0)
            mag, phase = stft.transform(x)
            output[key + '_' + 'magnitude'] = mag
            output[key + '_' + 'phase'] = phase
        return output

    def __len__(self):
        return len(self.mixture_set)

def collate_and_trim(batch, dim=0):
    keys = list(batch[0].keys())
    outbatch = {key: [] for key in keys}
    min_length = min([sample[keys[0]].size(dim) for sample in batch])
    for sample in batch:
        length = sample[keys[0]].size(dim)
        start = (length - min_length) // 2
        for key in keys:
            outbatch[key].append(sample[key].narrow(dim, start, min_length))

    outbatch = {key: torch.stack(values) for key, values in outbatch.items()}
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
        random.shuffle(files)
        train_speeches.extend(files[:num_train])
        val_speeches.extend(files[num_train:])
    return train_speeches, val_speeches

def get_noise_files(noise_path, noises, num_train=2):
    assert num_train <= len(noises)
    if noise_path[-1] != '/':
        noise_path += '/'
    noises = [noise_path + noise for noise in noises]
    random.shuffle(noises)
    train_noises = noises[:num_train]
    val_noises = noises[num_train:]
    return train_noises, val_noises

def main():
    # test code
    speaker_path = '/media/data/timit-wav/train'
    noise_path = '/media/data/noises-16k'

    # get training and validation files
    speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0']
    noises = ['car-16k.wav', 'babble-16k.wav', 'street-16k.wav']
    train_speeches, val_speeches = get_speech_files(speaker_path, speakers)
    train_noises, val_noises = get_noise_files(noise_path, noises)

    trainset = TwoSourceMixtureDataset(train_speeches, train_noises)
    valset = TwoSourceMixtureDataset(val_speeches, val_noises)
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))

    # output validation set
    for i in range(len(valset)):
        sample = valset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        mix, target, inter = mix.numpy(), target.numpy(), inter.numpy()
    print('Clean Speech Shape: ', target.shape)
    print('Noisy Speech Shape: ', mix.shape)
    librosa.output.write_wav('results/clean_example.wav', target, 16000, norm = True)
    librosa.output.write_wav('results/noisy_example.wav', mix, 16000, norm = True)

if __name__ == '__main__':
    main()

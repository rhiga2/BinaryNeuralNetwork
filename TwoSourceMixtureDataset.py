import numpy as np
import librosa
import glob
import torch
import itertools
from torch.utils.data import Dataset
import random
import pdb

class TwoSourceMixtureDataset(Dataset):
    def __init__(self, speeches, interferences, samp_freq=16000, snr=0,
        random_start=True, transform=None):
        self.samp_freq = samp_freq
        self.snr = np.power(10, snr/20)
        self.random_start = random_start
        self.mixes = list(itertools.product(speeches, interferences))
        self.transform = None

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
        sig, _ = librosa.core.load(sigf, sr=self.samp_freq, mono=True,
                                      duration=duration,
                                      offset=sig_offset,
                                      res_type='kaiser_fast')
        inter, _ = librosa.core.load(interf, sr=self.samp_freq, mono=True,
                                      duration=duration, offset=inter_offset,
                                      res_type='kaiser_fast')

        # normalize and mix signals
        sig = torch.FloatTensor(sig / np.std(sig))
        inter = torch.FloatTensor(inter / np.std(inter))
        mix = 1/(1 + 1/self.snr) * sig + 1/(1 + self.snr) * inter

        if self.transform:
            sig = self.transform(sig)
            inter = self.transform(inter)
            mix = self.transform(mix)

        return mix, sig, inter

    def __getitem__(self, i):
        sigf, interf = self.mixes[i] # get sig and interference file
        return self._getmix(sigf, interf)

def collate_and_trim(batch, hop):
    outbatch = [[], [], []]
    min_length = min([sample[0].shape[0] for sample in batch])
    min_length = (min_length // hop) * hop
    for sample in batch:
        outbatch[0].append(sample[0][:min_length])
        outbatch[1].append(sample[1][:min_length])
        outbatch[2].append(sample[2][:min_length])

    outbatch = [torch.FloatTensor(np.array(records)) for records in outbatch]
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
        files = glob2.glob(speaker_path + speaker + '*.wav')
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
        mix, target, inter = valset[i]
        mix, target, inter = mix.numpy(), target.numpy(), inter.numpy()
    print('Clean Speech Shape: ', target.shape)
    print('Noisy Speech Shape: ', mix.shape)
    librosa.output.write_wav('results/clean_example.wav', target, 16000, norm = True)
    librosa.output.write_wav('results/noisy_example.wav', mix, 16000, norm = True)

if __name__ == '__main__':
    main()

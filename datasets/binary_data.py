import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.make_data import *
from datasets.quantized_data import *
import argparse

def make_binary_mask(x):
    return x > 0

def quantize_and_disperse(mix_mag, quantizer, disperser):
    mix_mag = torch.FloatTensor(mix_mag / np.max(np.abs(mix_mag))).unsqueeze(0)
    qmag = quantizer(mix_mag)
    _, channels, frames = qmag.size()
    bmag = disperser(qmag.view(1, -1))
    bmag = bmag.squeeze(0).contiguous()
    bmag = torch.cat(torch.chunk(bmag, channels, dim=1), dim=0)
    bmag = bmag.numpy()
    return bmag

def accumulate(x, quantizer, disperser):
    x = torch.FloatTensor(x)
    channels, frames =  x.size()
    x = torch.cat(torch.chunk(x, channels // disperser.num_bits, dim=0), dim=1).unsqueeze(0)
    x = disperser.inverse(x).view(-1, frames)
    x = quantizer.inverse(x)
    return x.numpy()

def stft(x, window='hann', nperseg=1024, noverlap=768):
    stft_x = signal.stft(x,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap)[2]
    real, imag = np.real(stft_x), np.imag(stft_x)
    mag = np.sqrt(real**2 + imag**2 + 1e-6)
    phase = stft_x / (mag + 1e-6)
    return mag, phase

def istft(mag, phase, window='hann', nperseg=1024, noverlap=768):
    stft_x = mag * phase
    x = signal.istft(stft_x, window=window, nperseg=nperseg, noverlap=noverlap)[1]
    return x

class BinaryDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if data_dir[-1] != '/':
            self.data_dir += '/'
        flist = glob.glob(self.data_dir + 'binary_data*.npz')
        self.length = len(flist)

    def __getitem__(self, i):
        binary_fname = self.data_dir + ('binary_data%d.npz'%i)
        binary_data = np.load(binary_fname)
        return {'bmag': binary_data['bmag'], 'ibm': binary_data['ibm']}

    def __len__(self):
        return self.length

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--num_bits', '-nb', type=int, default=4)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    quantizer = Quantizer(min=-1, delta=2/2**(args.num_bits),
        num_bits=args.num_bits, use_mu=True)
    disperser = Disperser(args.num_bits)
    trainset, valset = make_mixture_set(args.toy)
    print('Samples in Trainset: ', len(trainset))
    print('Samples in Valset: ', len(valset))
    dataset_dir = '/media/data/binary_audio/'
    if not args.toy:
        train_dir = dataset_dir + 'train/'
        val_dir = dataset_dir + 'val/'
    else:
        train_dir = dataset_dir + 'toy_train/'
        val_dir = dataset_dir + 'toy_val/'

    for i in range(len(trainset)):
        binary_fname = train_dir + 'binary_data%d.npz' % i
        sample = trainset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        mix_mag, mix_phase = stft(mix)
        targ_mag, targ_phase = stft(target)
        inter_mag, inter_phase = stft(inter)
        ibm = make_binary_mask(targ_mag - inter_mag).astype(np.uint8)
        bmag = quantize_and_disperse(mix_mag, quantizer, disperser).astype(np.uint8)
        np.savez(
            binary_fname,
            bmag=bmag,
            ibm=ibm
        )

    # Output validation binarization
    for i in range(len(valset)):
        binary_fname = val_dir + 'binary_data%d.npz' % i
        sample = valset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        mix_mag, mix_phase = stft(mix)
        targ_mag, targ_phase = stft(target)
        inter_mag, inter_phase = stft(inter)
        ibm = make_binary_mask(targ_mag - inter_mag).astype(np.uint8)
        bmag = quantize_and_disperse(mix_mag, quantizer, disperser).astype(np.uint8)
        np.savez(
            binary_fname,
            bmag=bmag,
            ibm=ibm
        )

if __name__ == '__main__':
    main()

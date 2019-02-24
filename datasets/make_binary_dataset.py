import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.signal as signal
import datasets.make_data as make_data
import datasets.binary_data as binary_data
import datasets.quantized_data as quantized_data
import argparse

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--num_bits', '-nb', type=int, default=4)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    quantizer = quantized_data.Quantizer(min=-1, delta=2/2**(args.num_bits),
        num_bits=args.num_bits, use_mu=False)
    disperser = quantized_data.Disperser(args.num_bits)
    trainset, valset, testset = make_data.make_mixture_set(max_duration=2,
        toy=args.toy, transform=lambda x : signal.decimate(x, 2))
    print('Samples in Trainset: ', len(trainset))
    print('Samples in Valset: ', len(valset))
    print('Samples in Testset: ', len(testset))
    dataset_dir = '/media/data/binary_audio/'
    if not args.toy:
        train_dir = dataset_dir + 'train/'
        val_dir = dataset_dir + 'val/'
        test_dir = dataset_dir + 'test/'
    else:
        train_dir = dataset_dir + 'toy_train/'
        val_dir = dataset_dir + 'toy_val/'
        test_dir = dataset_dir + 'toy_test/'

    dirs = [train_dir, val_dir, test_dir]
    datasets = [trainset, valset, testset]
    stft = stft.STFT(nfft=1024, stride=256, win='hann')

    for directory, dataset in zip(dirs, datasets):
        for i in range(len(dataset)):
            binary_fname = directory + 'binary_data%d.npz' % i
            raw_fname = directory + 'raw_data%d.npz' % i
            sample = dataset[i]
            mix = torch.FloatTensor(sample['mixture'])
            target = torch.FloatTesnor(sample['target'])
            inter = torch.FloatTensor(sample['interference'])
            mix_mag, mix_phase = stft(mix.unsqueeze(0))
            mix_mag = mix_mag.squeeze(0)
            mix_phase = mix_phase.squeeze(0)
            targ_mag, targ_phase = stft(target.unsqueeze(0))
            targ_mag = targ_mag.squeeze(0)
            targ_phase = targ_phase.squeeze(0)
            inter_mag, inter_phase = stft(inter.unsqueeze(0))
            inter_mag = inter_mag.squeeze(0)
            inter_phase = inter_phase.squeeze(0)
            ibm = binary_data.make_binary_mask(targ_mag - inter_mag).to(torch.uint8)

            # uint8 does not convert nicely to torch float tensor
            bmag = 2*(mix_mag / torch.max(torch.abs(mix_mag))) - 1
            bmag = quantized_data.quantize_and_disperse(bmag, quantizer, disperser).to(torch.uint8)
            np.savez(
                binary_fname,
                bmag=bmag.numpy(),
                ibm=ibm.numpy()
            )

            if directory != train_dir:
                np.savez(
                    raw_fname,
                    mix=mix.numpy(),
                    target=target.numpy(),
                    interference=inter.numpy()
                )

if __name__ == '__main__':
    main()

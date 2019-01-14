import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets.make_data import *
from datasets.binary_data import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--num_bits', '-nb', type=int, default=4)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    quantizer = Quantizer(min=-1, delta=2/2**(args.num_bits),
        num_bits=args.num_bits, use_mu=True)
    disperser = Disperser(args.num_bits)
    trainset, valset, testset = make_mixture_set(toy=args.toy)
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

    for directory, dataset in zip(dirs, datasets):
        for i in range(len(dataset)):
            binary_fname = directory + 'binary_data%d.npz' % i
            raw_fname = directory + 'raw_data%d.npz' % i
            sample = dataset[i]
            mix, target, inter = sample['mixture'], sample['target'], sample['interference']
            mix_mag, mix_phase = stft(mix)
            targ_mag, targ_phase = stft(target)
            inter_mag, inter_phase = stft(inter)
            ibm = make_binary_mask(targ_mag - inter_mag).astype(np.uint8)
            bmag = quantize_and_disperse(mix_mag, quantizer, disperser).astype(np.uint8)
            np.savez(
                binary_fname,
                bmag=bmag,
                ibm=ibm,
                spec=mix_mag.astype(np.float32)
            )

            if directory != train_dir:
                np.savez(
                    raw_fname,
                    mix=mix,
                    target=target,
                    interference=inter
                )

if __name__ == '__main__':
    main()

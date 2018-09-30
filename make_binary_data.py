import argparse
import pickle as pkl
import binary_data
import numpy as np
from two_source_mixture import *
from sinusoidal_data import *

def make_mixture_set():
    np.random.seed(0)
    speaker_path = '/media/data/timit-wav/train'
    targ_speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
                    'dr1/fdaw0', 'dr1/fjsp0', 'dr1/fsjk1', 'dr1/fvmh0',
                    'dr1/fsma0', 'dr1/ftbr0']
    inter_speakers = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0',
                    'dr1/mwad0', 'dr1/mwar0']
    train_speeches, val_speeches = get_speech_files(speaker_path, targ_speakers, num_train=7)
    train_noises, val_noises = get_speech_files(speaker_path, inter_speakers, num_train=7)

    crop = lambda x: crop_length(x, 256)
    trainset = TwoSourceMixtureDataset(train_speeches, train_noises, transform=crop)
    valset = TwoSourceMixtureDataset(val_speeches, val_noises, transform=crop)
    return trainset, valset

def main():
    parser = argparse.ArgumentParser(description='binary data')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()
    if not args.toy:
        trainset, valset = make_mixture_set()
        config_name = 'config.npz'
        train_subdir = 'train/'
        val_subdir = 'val/'
    else:
        trainset = SinusoidDataset(size=1000, length=32000,
            sig_range=[0, 4000], noise_range=[4000, 8000])
        valset = SinusoidDataset(size=100, length=32000,
            sig_range=[0, 4000], noise_range=[4000, 8000])
        config_name = 'toy_config.npz'
        train_subdir = 'toy_train/'
        val_subdir = 'toy_val/'
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))

    x = []
    dataset_dir = '/media/data/binary_audio/'
    for i in range(0, len(trainset), 25):
        sample = trainset[i]
        mix_mag, mix_phase = stft(sample['mixture'])
        x.append(mix_mag.reshape(-1))
    centers, bins = kmeans_qlevels(np.concatenate(x, axis=0))
    np.savez(dataset_dir + config_name, centers=centers, bins=bins)
    pkl.dump(trainset, train_dir + 'dataset.pkl')
    pkl.dump(valset, val_dir + 'dataset.pkl')

    # Output training binarization
    for i in range(len(trainset)):
        binary_fname = train_subdir + 'binary_data%d.npz' % i
        sample = trainset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        mix_mag, mix_phase = stft(mix)
        targ_mag, targ_phase = stft(target)
        inter_mag, inter_phase = stft(inter)
        ibm = make_binary_mask(targ_mag - inter_mag, dtype=np.uint8)
        bmag = binarize(mix_mag, bins)
        np.savez(
            dataset_dir + binary_fname,
            bmag=bmag,
            ibm=ibm
        )

    # Output validation binarization
    for i in range(len(valset)):
        binary_fname = val_subdir + 'binary_data%d.npz' % i
        sample = valset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        mix_mag, mix_phase = stft(mix)
        targ_mag, targ_phase = stft(target)
        inter_mag, inter_phase = stft(inter)
        ibm = make_binary_mask(targ_mag - inter_mag, dtype=np.uint8)
        bmag = binarize(mix_mag, bins)
        np.savez(
            dataset_dir + binary_fname,
            bmag=bmag,
            ibm=ibm
        )

if __name__ == '__main__':
    main()

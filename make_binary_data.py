import argparse
import pickle as pkl
from datasets.binary_data import *
import numpy as np
from datasets.two_source_mixture import *
from datasets.sinusoidal_data import *
from torch.utils.data import Dataset, DataLoader

def make_dataset(batchsize, seed=0, toy=False):
    np.random.seed(seed)

    train_dir = '/media/data/binary_audio/train'
    val_dir = '/media/data/binary_audio/val'
    if toy:
        train_dir = '/media/data/binary_audio/toy_train'
        val_dir = '/media/data/binary_audio/toy_val'

    trainset = BinaryDataset(train_dir)
    valset = BinaryDataset(val_dir)
    collate_fn = lambda x: collate_and_trim(x, axis=1)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)
    return train_dl, val_dl

def make_mixture_set(toy=False):
    speaker_path = '/media/data/timit-wav/train'
    targets = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
        'dr1/fdaw0', 'dr1/fjsp0', 'dr1/fsjk1', 'dr1/fvmh0',
        'dr1/fsma0', 'dr1/ftbr0']
    train_speeches, val_speeches = get_speech_files(speaker_path, targets, num_train=7)

    if toy:
        # trainset = SineSpeechData(train_speeches, 10, hop=256)
        # valset = SineSpeechData(val_speeches, 10, hop=256)
        noise_path = '/media/data/Nonspeech'
        interferences = ['n81.wav', # chimes
                         'n97.wav', # eating chips
                         'n21.wav', # motorcycle
                         'n46.wav', # ocean
                         'n47.wav', # birds
                         'n55.wav', # cicadas?
                         'n59.wav', # jungle?
                         ]
        train_noises, val_noises = get_noise_files(noise_path, interferences)
        trainset = TwoSourceMixtureDataset(train_speeches, train_noises, hop=256)
        valset = TwoSourceMixtureDataset(val_speeches, val_noises, hop=256)
    else:
        interferences = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0',
                    'dr1/mwad0', 'dr1/mwar0']
        train_noises, val_noises = get_speech_files(speaker_path, interferences, num_train=7)
        trainset = TwoSourceMixtureDataset(train_speeches, train_noises, hop=256)
        valset = TwoSourceMixtureDataset(val_speeches, val_noises, hop=256)

    return trainset, valset

def main():
    parser = argparse.ArgumentParser(description='binary data')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--denoising', action='store_true')
    args = parser.parse_args()
    dataset_dir = '/media/data/binary_audio/'
    if not args.toy:
        config_name = 'config.npz'
        train_dir = dataset_dir + 'train/'
        val_dir = dataset_dir + 'val/'
        if args.denoising:
            mode = 'denoising'
    else:
        config_name = 'toy_config.npz'
        train_dir = dataset_dir + 'toy_train/'
        val_dir = dataset_dir + 'toy_val/'
    trainset, valset = make_mixture_set(args.toy)
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))

    x = []
    for i in range(0, len(trainset), 25):
        sample = trainset[i]
        mix_mag, mix_phase = stft(sample['mixture'])
        x.append(mix_mag.reshape(-1))
    centers, bins = kmeans_qlevels(np.concatenate(x, axis=0))
    np.savez(dataset_dir + config_name, centers=centers, bins=bins)
    with open(train_dir + 'dataset.pkl', 'wb') as f:
        pkl.dump(trainset, f)
    with open(val_dir + 'dataset.pkl', 'wb') as f:
        pkl.dump(valset, f)

    # Output training binarization
    for i in range(len(trainset)):
        binary_fname = train_dir + 'binary_data%d.npz' % i
        sample = trainset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        mix_mag, mix_phase = stft(mix)
        targ_mag, targ_phase = stft(target)
        inter_mag, inter_phase = stft(inter)
        ibm = make_binary_mask(targ_mag - inter_mag, dtype=np.uint8)
        bmag = binarize_stft(mix_mag, bins)
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
        ibm = make_binary_mask(targ_mag - inter_mag, dtype=np.uint8)
        bmag = binarize(mix_mag, bins)
        np.savez(
            binary_fname,
            bmag=bmag,
            ibm=ibm
        )

if __name__ == '__main__':
    main()

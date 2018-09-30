import numpy as np
import scipy.signal as signal
import json
from .two_source_mixture import *
from .sinusoidal_data import *
from sklearn.cluster import KMeans
import argparse

def uniform_qlevels(x, levels=16):
    '''
    x is flattened array of numbers
    '''
    xmax = np.max(x)
    xmin = np.min(x)
    centers = (xmax - xmin)*(np.arange(levels) + 0.5)/levels + xmin
    bins = get_bins(centers)
    return centers, bins

def kmeans_qlevels(x, levels=16):
    '''
    x is flattened array of numbers
    '''
    km = KMeans(n_clusters=levels)
    km.fit(np.expand_dims(x, axis=1))
    centers = np.sort(km.cluster_centers_.reshape(-1))
    bins = get_bins(centers)
    return centers, bins

def get_bins(centers):
    return (centers[:-1] + centers[1:])/2

def binarize(x, bins, num_bits=4):
    '''
    x is shape (F, T)
    F = frequency range
    T = time range
    '''
    assert len(bins)+1 == 2**num_bits
    digit_x = np.digitize(x, bins).astype(np.uint8)
    binary_x = []
    for i in range(digit_x.shape[0]):
        bits = np.unpackbits(np.expand_dims(digit_x[i], axis=0), axis=0)[-num_bits:]
        binary_x.append(bits)
    return np.concatenate(binary_x, axis=0)

def quantize(x, bins, centers):
    digit_x = np.digitize(x, bins).astype(np.int)
    qx = centers[digit_x] # qx = quantized x
    return qx

def make_binary_mask(premask, dtype=np.float):
    return np.array(premask > 0, dtype=dtype)

def stft(x, window='hann', nperseg=1024, noverlap=768):
    stft_x = signal.stft(x,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap)[2]
    real, imag = np.real(stft_x), np.imag(stft_x)
    mag = np.sqrt(real**2 + imag**2)
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

def crop_length(x, hop):
    return x[:len(x)//hop*hop]

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

import numpy as np
import scipy.signal as signal
import json
from two_source_mixture import *
from sklearn.cluster import KMeans

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
    phase = np.arctan2(imag, real)
    return mag, phase

def istft(mag, phase, window='hann', nperseg=1024, noverlap=768):
    stft_x = mag*np.exp(1j*phase)
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

class RawDataset():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if data_dir[-1] != '/':
            self.data_dir += '/'
        flist = glob.glob(self.data_dir + 'raw_data*.npz')
        self.length = len(flist)

    def __getitem__(self, i):
        raw_fname = self.data_dir + ('raw_data%d.npz'%i)
        raw_data = np.load(raw_fname)
        return {'mix': raw_data['mix'], 'target': raw_data['target']}

    def __len__(self):
        return self.length

def main():
    np.random.seed(0)
    speaker_path = '/media/data/timit-wav/train'
    targ_speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
                    'dr1/fdaw0', 'dr1/fjsp0', 'dr1/fsjk1', 'dr1/fvmh0']
    inter_speakers = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0',
                    'mwad0']

    train_speeches, val_speeches = get_speech_files(speaker_path, targ_speakers, num_train=6)
    train_noises, val_noises = get_speech_files(speaker_path, inter_speakers, num_train=6)

    trainset = TwoSourceMixtureDataset(train_speeches, train_noises)
    valset = TwoSourceMixtureDataset(val_speeches, val_noises)
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))

    dataset_dir = '/media/data/binary_audio/'
    x = []
    for i in range(0, len(trainset), 10):
        sample = trainset[i]
        mix_mag, mix_phase = stft(sample['mixture'])
        x.append(mix_mag.reshape(-1))
    centers, bins = kmeans_qlevels(np.concatenate(x, axis=0))

    # Output training binarization
    for i in range(len(trainset)):
        binary_fname = 'train/binary_data%d.npz' % i
        raw_fname = 'train/raw_data%d.npz' % i
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
        np.savez(
            dataset_dir + raw_fname,
            mix=mix,
            target=target
        )

    # Output validation binarization
    for i in range(len(valset)):
        binary_fname = 'val/binary_data%d.npz' % i
        raw_fname = 'val/raw_data%d.npz' % i
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
        np.savez(
            dataset_dir + raw_fname,
            mix=mix,
            target=target
        )

if __name__ == '__main__':
    main()

import numpy as np
import scipy.signal as signal
import json
from two_source_mixture import *

def main():
    window = 'hann'
    nperseg = 1024
    noverlap = 768
    np.random.seed(0)
    speaker_path = '/media/data/timit-wav/train'
    targ_speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0']
    inter_speakers = ['dr1/mdpk0', 'dr1/mjwt0']

    train_speeches, val_speeches = get_speech_files(speaker_path, targ_speakers)
    train_noises, val_noises = get_speech_files(speaker_path, inter_speakers)
    transform = lambda x: signal.stft(x, window=window, nperseg=nperseg, noverlap=noverlap)[2]

    trainset = TwoSourceMixtureDataset(train_speeches, train_noises, transform=transform)
    valset = TwoSourceMixtureDataset(val_speeches, val_noises, transform=transform)
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))

    # out trainset
    dataset_dir = '/media/data/binary_audio/'
    config = {'window': window, 'nperseg': nperseg, 'noverlap': noverlap}
    json_out = json.dumps(config)

    with open(dataset_dir + 'config.json', 'w') as f:
        f.write(json_out)

    for i in range(len(trainset)):
        fname = 'train/%d.npz' % i
        sample = trainset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        np.savez(dataset_dir + fname, mix=mix, target=target)

    # output validation set
    for i in range(len(valset)):
        fname = 'val/%d.npz % i'
        sample = valset[i]
        mix, target, inter = sample['mixture'], sample['target'], sample['interference']
        np.savez(dataset_dir + fname, mix=mix, target=target)

    print('Clean Speech Shape: ', target.shape)
    print('Noisy Speech Shape: ', mix.shape)

if __name__ == '__main__':
    main()

import unittest
import torch
import numpy as np
from bss_eval import *

def bss_eval_test( sep, sources, i=0):
    # Current target
    from numpy import dot, linalg, log10
    min_len = min([len(sep), len(sources[i])])
    sources = sources[:,:min_len]
    sep = sep[:min_len]
    target = sources[i]

    # Target contribution
    s_target = target * dot( target, sep.T) / dot( target, target.T)

    # Interference contribution
    pse = dot( dot( sources, sep.T), \
    linalg.inv( dot( sources, sources.T))).T.dot( sources)
    e_interf = pse - s_target

    # Artifact contribution
    e_artif= sep - pse;

    # Interference + artifacts contribution
    e_total = e_interf + e_artif;

    # Computation of the log energy ratios
    sdr = 10*log10( sum( s_target**2) / sum( e_total**2));
    sir = 10*log10( sum( s_target**2) / sum( e_interf**2));
    sar = 10*log10( sum( (s_target + e_interf)**2) / sum( e_artif**2));

    # Done!
    return BSSMetrics(sdr, sir, sar)

class TestBssEval(unittest.TestCase):
    def setUp(self):
        # Create dataset
        speaker_path = '/media/data/timit-wav/train'
        noise_path = '/media/data/noises-16k'

        # get training and validation files
        speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0']
        noises = ['car-16k.wav', 'babble-16k.wav', 'street-16k.wav']
        speeches, _ = get_speech_files(speaker_path, speakers)
        noises, _ = get_noise_files(noise_path, noises)

        self.trainset = TwoSourceMixtureDataset(speeches, noises)

    def test_bsseval(self):
        '''
        Compare my pytorch implementation of bss eval with Shrikant's numpy implementation
        '''
        for i in range(len(self.trainset)):
            sample = self.trainset[i]
            pred = sample['target'] + \
                np.random.random(sample['mixture'].shape)*0.1 + \
                sample['interference']*0.1
            sources = np.stack([sample['target'], sample['interference']], axis=0)
            metric = bss_eval(torch.FloatTensor(pred), torch.FloatTensor(sources))
            metric_test = bss_eval_test(pred, sources)
            print('SDR Error: ', (metric.sdr - metric_test.sdr)**2, metric.sdr, metric_test.sdr)
            print('SIR Error: ', (metric.sir - metric_test.sir)**2, metric.sir, metric_test.sir)
            print('SAR Error: ', (metric.sar - metric_test.sar)**2, metric.sar, metric_test.sar)

if __name__ == '__main__':
    unittest.main()

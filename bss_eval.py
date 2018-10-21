import numpy as np
from datasets.two_source_mixture import *

class BSSMetrics:
    def __init__(self, sdr=0, sir=0, sar=0):
        self.sdr = sdr
        self.sir = sir
        self.sar = sar

class BSSMetricsList:
    def __init__(self):
        self.sdrs = []
        self.sirs = []
        self.sars = []

    def append(self, metric):
        self.sdrs.append(metric.sdr)
        self.sirs.append(metric.sir)
        self.sars.append(metric.sar)

    def extend(self, metrics):
        self.sdrs.extend(metrics.sdrs)
        self.sirs.extend(metrics.sirs)
        self.sars.extend(metrics.sars)

    def mean(self):
        sdrs = torch.FloatTensor(self.sdrs)
        sirs = torch.FloatTensor(self.sirs)
        sars = torch.FloatTensor(self.sars)
        sdr = torch.mean(sdrs[torch.isfinite(sdrs)])
        sir = torch.mean(sirs[torch.isfinite(sirs)])
        sar = torch.mean(sars[torch.isfinite(sars)])
        return sdr, sir, sar

def compute_s_target(pred, target):
    '''
    pred (T)
    target (T)
    '''
    return torch.mean(target*pred)/\
        torch.mean(target**2)*target

def compute_source_projection(pred, sources):
    '''
    pred (T)
    sources (T, S)
    '''
    pinv_pred = torch.matmul(torch.pinverse(sources), pred)
    return torch.matmul(sources, pinv_pred)

def compute_sdr(pred, s_target):
    e_total = pred - s_target
    return 10*torch.log10(torch.mean(s_target**2)/torch.mean(e_total**2))

def compute_sir(s_target, e_inter):
    return 10*torch.log10(torch.mean(s_target**2)/torch.mean(e_inter**2))

def compute_sar(s_target, e_inter, e_art):
    source_projection = s_target + e_inter
    return 10*torch.log10(torch.mean(source_projection**2)/torch.mean(e_art**2))

def bss_eval(pred, sources, target_idx=0):
    '''
    BSS eval metric calculation.
    pred (T) s.t. T is the number of time steps
    sources (S, T) s.t. S is the number of sources in mixture
    target_idx (int) index of target in sources
    '''
    sources = torch.t(sources)
    target = sources[:, target_idx]
    s_target = compute_s_target(pred, target)
    source_proj = compute_source_projection(pred, sources)
    e_inter = source_proj - s_target
    e_art = pred - source_proj
    sdr = compute_sdr(pred, s_target)
    sir = compute_sir(s_target, e_inter)
    sar = compute_sar(s_target, e_inter, e_art)
    metric = BSSMetrics(sdr, sir, sar)
    return metric

def bss_eval_batch(preds, source_tensor, target_idx=0):
    '''
    preds (N, T)
    source_tensor (N, S, T)
    '''
    metrics = BSSMetricsList()
    sdrs, sirs, sars = [], [], []
    for i in range(preds.size()[0]):
        pred = preds[i]
        sources = source_tensor[i]
        metric = bss_eval(pred, sources, target_idx)
        metrics.append(metric)
    return metrics

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

def test_metrics():
    # test code
    speaker_path = '/media/data/timit-wav/train'
    noise_path = '/media/data/noises-16k'

    # get training and validation files
    speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0']
    noises = ['car-16k.wav', 'babble-16k.wav', 'street-16k.wav']
    speeches, _ = get_speech_files(speaker_path, speakers)
    noises, _ = get_noise_files(noise_path, noises)

    trainset = TwoSourceMixtureDataset(speeches, noises)
    for i in range(len(trainset)):
        sample = trainset[i]
        pred = sample['mixture'] + np.random.random(sample['mixture'].shape)*0.01
        sources = np.stack([sample['target'], sample['interference']], axis=0)
        metric = bss_eval(torch.FloatTensor(pred), torch.FloatTensor(sources))
        metric_test = bss_eval_test(pred, sources)
        print('SDR Error: ', (metric.sdr - metric_test.sdr)**2)
        print('SIR Error: ', (metric.sir - metric_test.sir)**2)
        print('SAR Error: ', (metric.sar - metric_test.sar)**2)

if __name__=='__main__':
    test_metrics()

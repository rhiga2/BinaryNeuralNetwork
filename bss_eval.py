import mir_eval
from TwoSourceMixtureDataset import *

def compute_s_target(pred, target):
    '''
    pred (T)
    target (T)
    '''
    return torch.mean(target*pred)/torch.mean(target**2)*target

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
    '''
    target = sources[:, target_idx]
    s_target = compute_s_target(pred, target)
    source_proj = compute_source_projection(pred, sources)
    e_inter = source_proj - s_target
    e_art = pred - source_proj
    sdr = compute_sdr(pred, s_target)
    sir = compute_sir(s_target, e_inter)
    sar = compute_sar(s_target, e_inter, e_art)
    return sdr, sir, sar

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
    return (sdr, sir, sar)

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
        pred = sample['mixture']
        sources = torch.stack([sample['target'], sample['interference']], dim=1)
        sdr_est, sir_est, sar_est = bss_eval(pred, sources)
        sdr, sir, sar = bss_eval_test(pred.numpy(), sources.numpy().T)
        print('SDR Error: ', (sdr - sdr_est.numpy())**2, sdr, sdr_est.numpy())
        print('SIR Error: ', (sir - sir_est.numpy())**2, sir, sir_est.numpy())
        print('SAR Error: ', (sar - sar_est.numpy())**2, sar, sar_est.numpy())

if __name__=='__main__':
    test_metrics()

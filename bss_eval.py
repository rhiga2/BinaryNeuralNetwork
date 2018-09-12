import mir_eval
from TwoSourceMixtureDataset import *

def compute_s_target(pred, target):
    print(target.size())
    return torch.mean(target*pred, dim=1)/torch.mean(target**2, dim=1)*target

def compute_e_interference(pred, inter):
    return torch.mean(inter*pred, dim=1)/torch.mean(inter**2, dim=1)*inter

def compute_e_artifact(pred, s_target, e_inter):
    return pred - s_target - e_inter

def compute_sdr(s_target, e_inter, e_art):
    error = e_inter + e_art
    return 10*torch.log10(torch.mean(s_target**2, dim=1)/torch.mean(error**2, dim=1))

def compute_sir(s_target, e_inter):
    return 10*torch.log10(torch.mean(s_target**2, dim=1)/torch.mean(e_inter**2, dim=1))

def compute_sar(s_target, e_inter, e_art):
    source_projection = s_target + e_inter
    return 10*torch.log10(torch.mean(source_projection**2, dim=1)/torch.mean(e_art**2, dim=1))

def bss_eval(pred, target, inter):
    '''
    BSS eval for two sources
    '''
    s_target = compute_s_target(pred, target)
    e_inter = compute_e_interference(pred, inter)
    e_art = compute_e_artifact(pred, s_target, e_inter)
    sdr = compute_sdr(s_target, e_inter, e_art)
    sir = compute_sir(s_target, e_inter)
    sar = compute_sar(s_target, e_inter, e_art)
    return sdr, sir, sar

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
        pred = sample['mixture'].unsqueeze(0)
        target = sample['target'].unsqueeze(0)
        interference = sample['interference'].unsqueeze(0)
        sdr_est, sir_est, sar_est = bss_eval(pred, target, interference)
        refs = np.stack([sample['target'].numpy(), sample['interference'].numpy()], axis=1).T
        ests = np.stack([sample['mixture'].numpy(), sample['interference'].numpy()], axis=1).T
        sdrs, sirs, sars, perm = mir_eval.separation.bss_eval_sources(refs, ests)
        print('SDR Error: ', (sdrs[0] - sdr_est.numpy())**2)
        print('SDR Error: ', (sirs[0] - sir_est.numpy())**2)
        print('SDR Error: ', (sars[0] - sar_est.numpy())**2)

if __name__=='__main__':
    test_metrics()

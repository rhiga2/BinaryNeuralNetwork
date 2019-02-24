import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf

def stft(x, window='hann', nperseg=1024, noverlap=768):
    stft_x = signal.stft(x,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap)[2]
    real, imag = np.real(stft_x), np.imag(stft_x)
    mag = np.sqrt(real**2 + imag**2)
    phase = np.angle(stft_x)
    # phase = stft_x / (mag + 1e-6)
    return mag, phase

def istft(mag, phase, window='hann', nperseg=1024, noverlap=768):
    stft_x = mag * (np.cos(phase) + 1j*np.sin(phase))
    x = signal.istft(stft_x, window=window, nperseg=nperseg, noverlap=noverlap)[1]
    return x

def get_fourier_basis(nfft):
    cutoff = nfft//2 + 1
    fft = np.fft.fft(np.eye(nfft))
    real_fft = np.real(fft)[:cutoff]
    im_fft = np.imag(fft)[:cutoff]
    basis = torch.FloatTensor(np.concatenate([real_fft, im_fft], axis=0))
    return basis

class STFT(nn.Module):
    def __init__(self, nfft=1024, stride=256, win='hann'):
        super().__init__()
        self.cutoff = nfft//2+1
        self.conv = nn.Conv1d(1, 2*self.cutoff, nfft, stride=stride,
                             padding=nfft, bias=False)
        basis = get_fourier_basis(nfft)
        if win is not None:
            window = torch.FloatTensor(signal.get_window(win, nfft))
            basis = window * basis
        self.conv.weight = nn.Parameter(basis.unsqueeze(1), requires_grad=False)

    def forward(self, x):
        spec = self.conv(x)
        real = spec[:, :self.cutoff]
        imag = spec[:, self.cutoff:]
        mag = torch.sqrt(real**2 + imag**2)
        angle = torch.atan2(imag, real)
        return mag, angle

class ISTFT(nn.Module):
    def __init__(self, nfft=1024, stride=256, win='hann'):
        super().__init__()
        assert nfft % stride == 0
        self.nfft = nfft
        self.cutoff = nfft//2 + 1
        self.conv_transpose = nn.ConvTranspose1d(
            2*self.cutoff, 1, nfft, stride=stride, bias=False
        )
        basis = get_fourier_basis(nfft)
        window = torch.ones(nfft)
        if win is not None:
            window = torch.FloatTensor(signal.get_window(win, nfft))
        norm = window**2
        for i in range(nfft//stride-1):
            idx = (i+1) * stride
            norm[:idx] += window[-idx:]**2
            norm[idx:] += window[:-idx]**2
        norm[norm < 1e-10] = 1.
        invbasis = torch.t(torch.pinverse(basis)) * window / norm
        invbasis = invbasis.contiguous().unsqueeze(1)
        self.conv_transpose.weight = nn.Parameter(invbasis, requires_grad=False)

    def forward(self, mag, angle):
        real = mag * torch.cos(angle)
        imag = mag * torch.sin(angle)
        spec = torch.cat([real, imag], dim=1)
        x = self.conv_transpose(spec)
        return x[:, :, self.nfft:x.size(2)-self.nfft]

def main():
    # Test multiple windows
    wins = [None, 'hann', 'hamming', 'blackman', 'parzen']
    nfft = 1024
    stride = 256
    x, samplerate = sf.read('example.wav')
    x = torch.FloatTensor(x)
    x = x.mean(-1)[:(len(x)//stride * stride)]
    x = x.unsqueeze(0).unsqueeze(0)
    for win in wins:
        torch_stft = STFT(nfft=nfft, stride=stride, win=win)
        torch_istft = ISTFT(nfft=nfft, stride=stride, win=win)
        mag, phase = torch_stft(x)
        print(torch.min(mag), torch.max(mag))
        x_hat = torch_istft(mag, phase)
        print('{} mean-square error:'.format(win),
            torch.mean((x - x_hat)**2).item())

if __name__ == '__main__':
    main()

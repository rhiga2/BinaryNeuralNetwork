import torch
import torch.nn as nn
import numpy as np
import scipy.signal as signal

class STFT(nn.Module):
    def __init__(self, nfft, hop, dtype=torch.float, window_type=None):
        super(STFT, self).__init__()
        fft = np.fft.fft(np.eye(nfft))
        if window_type == None:
            window = np.ones(nfft)
        else:
            window = signal.get_window(window_type, nfft)
        fft = fft * window
        scale = nfft / hop
        self.nfft = nfft
        self.hop = hop
        self.cutoff = nfft // 2 + 1
        fft_matrix = np.concatenate([np.real(fft[:self.cutoff]),
            np.imag(fft[:self.cutoff])], axis=0)
        fft_tensor = torch.tensor(fft_matrix, dtype=dtype)
        ifft_tensor = torch.t(torch.pinverse(scale * fft_tensor))
        fft_tensor = nn.Parameter(fft_tensor.unsqueeze(1), requires_grad=False)
        ifft_tensor = nn.Parameter(ifft_tensor.unsqueeze(1), requires_grad=False)
        self._transform = nn.Conv1d(1, 2*self.cutoff, nfft, stride=hop, bias=False,
             padding=nfft//2)
        self._inverse = nn.ConvTranspose1d(2*self.cutoff, 1, nfft, stride=hop, bias=False)
        self._transform.weight = fft_tensor
        self._inverse.weight = ifft_tensor

    def transform(self, x):
        x = x.unsqueeze(1)
        X = self._transform(x)
        real_X = X[:, :self.cutoff, :]
        imag_X = X[:, self.cutoff:, :]
        mag = torch.sqrt(real_X**2 + imag_X**2) * 2/self.nfft
        phase = torch.atan2(imag_X, real_X)
        return mag, phase

    def inverse(self, mag, phase):
        X = torch.cat([mag*torch.cos(phase), mag*torch.sin(phase)], dim=1) * self.nfft/2
        x = self._inverse(X).squeeze(1)
        x = x[:, self.nfft//2:x.size(1)-self.nfft//2]
        return x

    def forward(self, x):
        return self._transform(x)

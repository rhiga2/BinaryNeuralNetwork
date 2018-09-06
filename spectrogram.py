import torch
import torch.nn as nn
import numpy as np
import scipy.signal as signal
from TwoSourceMixtureDataset import *

class MakeSpectrogram(nn.Module):
    def __init__(self, fft_size, hop):
        super(MakeSpectrogram, self).__init__()
        self.fft_size = fft_size
        fft = np.fft.fft(np.eye(fft_size)) * np.hanning(fft_size)
        real_fft = nn.Parameter(torch.FloatTensor(np.real(fft)).unsqueeze(1), requires_grad=False)
        imag_fft = nn.Parameter(torch.FloatTensor(np.imag(fft)).unsqueeze(1), requires_grad=False)
        self.real_conv = nn.Conv1d(1, fft_size, fft_size, stride=hop, bias=False)
        self.imag_conv = nn.Conv1d(1, fft_size, fft_size, stride=hop, bias=False)
        self.real_conv.weight = real_fft
        self.imag_conv.weight = imag_fft

    def forward(self, x):
        assert len(x.size()) == 2 # 2D time series
        x = x.unsqueeze(1)
        real_x = self.real_conv(x)
        imag_x = self.imag_conv(x)
        mag = (real_x**2 + imag_x**2)[:, :self.fft_size // 2 + 1, :]
        return mag

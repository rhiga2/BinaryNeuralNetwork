import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np
import librosa
import matplotlib.pyplot as plt

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target, interference=None):
        mse = self.mse(prediction, target)
        return mse

class SignalDistortionRatio(nn.Module):
    def __init__(self, l1_penalty=0, epsilon = 2e-7):
        super(SignalDistortionRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target, interference=None):
        #sdr = torch.mean(prediction**2) / torch.mean(prediction * target)**2
        sdr = -torch.mean(prediction * target)**2 / (torch.mean(prediction**2) + self.epsilon)
        return sdr

class SignalInterferenceRatio(nn.Module):
    def __init__(self, epsilon=2e-7):
        super(SignalInterferenceRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target, interference):
        # prediction = prediction / (torch.std(prediction, dim=1, keepdim=True) + self.epsilon)
        sir = torch.mean(prediction * interference)**2 / (torch.mean(prediction * target)**2 + self.epsilon)
        # sir = -torch.mean(prediction * target)**2 / (torch.mean(prediction * interference)**2 + self.epsilon)
        return sir

class SignalArtifactRatio(nn.Module):
    def __init__(self, epsilon=2e-7):
        super(SignalArtifactRatio, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target, interference):
        # prediction = prediction / (torch.std(prediction, dim=1, keepdim=True) + self.epsilon)
        inter_norm = torch.mean(interference**2, dim = 1, keepdim=True)
        target_norm = torch.mean(target**2, dim = 1, keepdim=True)
        ref_correlation = torch.mean(prediction * target, dim = 1, keepdim=True)
        inter_correlation = torch.mean(prediction * interference, dim = 1, keepdim=True)
        project = inter_norm * ref_correlation * target + target_norm * inter_correlation * interference
        #sar = -torch.mean(project**2) / (torch.mean(prediction**2) + self.epsilon)
        sar = torch.mean(prediction**2) / (torch.mean(project**2) + self.epsilon)
        return sar

class ShortTimeObjectiveIntelligibility(nn.Module):
    def __init__(self):
        super(ShortTimeObjectiveIntelligibility, self).__init__()
        self.fs = 16000
        self.num_bands = 15
        self.center_freq = 150
        self.min_energy = 40
        self.fft_size = 512
        self.fft_in_frame_size = 256
        self.hop = 128
        self.num_frames = 30
        self.beta =  1 + 10**(15 / 20)
        self.fft_pad = (self.fft_size - self.fft_in_frame_size) // 2

        scale = self.fft_size / self.hop
        window = np.hanning(self.fft_in_frame_size)
        zero_pad = np.zeros(self.fft_pad)
        window = np.concatenate([zero_pad, window, zero_pad])
        fft = np.fft.fft(np.eye(self.fft_size))
        self.rows = self.fft_size // 2 + 1
        fft = np.vstack((np.real(fft[:self.rows,:]), np.imag(fft[:self.rows,:])))
        fft = window * fft
        self.fftmat = nn.Parameter(torch.FloatTensor(fft).unsqueeze(1), requires_grad=False)
        self.octmat, _ = self._get_octave_mat(self.fs, self.fft_size,
                                              self.num_bands, self.center_freq)
        self.octmat = nn.Parameter(torch.FloatTensor(self.octmat), requires_grad=False)

    def forward(self, prediction, target, inteference):
        # pred, targ = self._remove_silent_frames(prediction, target)

        # (batch, 1, time) to (batch, fft_size, frames)
        pred_mag, pred_phase = self._stft(prediction)
        targ_mag, targ_phase = self._stft(target)

        # (batch, fft_size, frames) to (batch, frames, fft_size)
        pred_mag = pred_mag.permute(0, 2, 1).contiguous()
        targ_mag = targ_mag.permute(0, 2, 1).contiguous()

        # (batch, frames, fft_size) to (batch, frames, num_bands)
        x = torch.sqrt(F.linear(targ_mag**2, self.octmat))
        y = torch.sqrt(F.linear(pred_mag**2, self.octmat))

        # (batch, frames, num_bands) to (batch, num_bands, frames)
        x = x.permute(0, 2, 1).contiguous()
        y = y.permute(0, 2, 1).contiguous()

        corr = 0
        for i, m in enumerate(range(self.num_frames, x.size()[2])):
            # segment (batch, num_bands, frames) to (batch, num_bands, new_frames)
            x_seg = x[:, :, m - self.num_frames : m]
            y_seg = y[:, :, m - self.num_frames : m]
            alpha = torch.sqrt(torch.sum(x_seg**2, dim=2, keepdim=True) / (torch.sum(y_seg**2, dim=2, keepdim=True) + 1e-7))
            y_prime = torch.min(alpha * y_seg, self.beta * x_seg)
            corr += self._correlation(x_seg, y_prime)

        return -corr / (i + 1)

    def _stft(self, seq):
        seq = seq.unsqueeze(1)
        stft = F.conv1d(seq, self.fftmat, stride=self.hop, padding=self.fft_pad)
        real = stft[:, :self.rows, :]
        imag = stft[:, self.rows:, :]
        mag = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)
        return mag, phase

    def _get_octave_mat(self, fs, nfft, numBands, mn):
        f = np.linspace(0, fs, nfft+1)
        f = f[:int(nfft/2)+1]
        k = np.arange(float(numBands))
        cf = 2**(k/3)*mn;
        fl = np.sqrt((2.**(k/3)*mn) * 2**((k-1.)/3)*mn)
        fr = np.sqrt((2.**(k/3)*mn) * 2**((k+1.)/3)*mn)
        A = np.zeros((numBands, len(f)) )

        for i in range(len(cf)) :
            b = np.argmin((f-fl[i])**2)
            fl[i] = f[b]
            fl_ii = b

            b = np.argmin((f-fr[i])**2)
            fr[i] = f[b]
            fr_ii = b
            A[i, np.arange(fl_ii,fr_ii)] = 1

        rnk = np.sum(A, axis=1)
        numBands = np.where((rnk[1:] >= rnk[:-1]) & (rnk[1:] != 0))[-1][-1]+1
        A = A[:numBands+1,:];
        cf = cf[:numBands+1];
        return A, cf

    def _remove_silent_frames(self, x, y):
        pass

    def _correlation(self, x, y):
        '''
        Input shape is (batch, bands, time dimension)
        '''
        xn = x - torch.mean(x, dim=2, keepdim=True)
        xn /= torch.sqrt(torch.sum(xn**2, dim=2, keepdim=True))
        yn = y - torch.mean(y, dim=2, keepdim=True)
        yn /= torch.sqrt(torch.sum(yn**2, dim=2, keepdim=True))
        r = torch.mean(torch.sum(xn * yn, dim=2))
        return r

class DiscreteWasserstein(nn.Module):
    def __init__(self, num_classes, mode='one_hot',
        dist_matrix=None, device=torch.device('cpu')):
        '''
        Input
        * num_classes: number of classes in dataset
        * mode is one of the following options
            - 'one_hot': targets are one hot vectors
            - 'integer': targets are integers
        *
        '''
        super(DiscreteWasserstein, self).__init__()
        self.mode = mode
        self.dist_matrix = dist_matrix
        if dist_matrix:
            col = torch.arange(0, num_classes).unsqueeze(1)
            row = torch.arange(0, num_classes)
            self.dist_matrix = torch.abs(col - row)

    def forward(self, x, y):
        '''
        Inputs:
        x is shape (batch, number of classes, time)
        y is shape (batch, number of classes, time) if one hot
            or (batch, time) otherwise
        '''
        # make input shape (batch x time, number of classes)
        batch, classes, time = x.size()
        x = x.permute(0, 2, 1).contiguous().view(-1, classes)
        if self.mode == 'one_hot':
            y = torch.argmax(y, dim=1)
        y = y.view(-1, 1)
        dists = self.dist_matrix[y]
        costs = torch.sum(x * dists, dim=1)
        return torch.mean(costs)

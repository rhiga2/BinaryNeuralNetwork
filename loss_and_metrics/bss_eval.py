import sys , os
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import loss_and_metrics.sepcosts as sepcosts

class BSSMetricsList:
    def __init__(self):
        self.sdrs = []
        self.sirs = []
        self.sars = []
        self.stois = []

    def extend(self, metrics):
        self.sdrs.extend(metrics.sdrs)
        self.sirs.extend(metrics.sirs)
        self.sars.extend(metrics.sars)
        self.stois.extend(metrics.stois)

    def mean(self):
        mean_sdr = torch.mean(torch.FloatTensor(self.sdrs)).item()
        mean_sir = torch.mean(torch.FloatTensor(self.sirs)).item()
        mean_sar = torch.mean(torch.FloatTensor(self.sars)).item()
        mean_stoi = torch.mean(torch.FloatTensor(self.stois)).item()
        return mean_sdr, mean_sir, mean_sar, mean_stoi

def compute_s_target(pred, target):
    '''
    pred (length)
    target (length)
    '''
    return torch.sum(target*pred)/\
        torch.sum(target**2)*target

def compute_source_projection(pred, sources):
    '''
    pred (length)
    sources (length, sources)
    '''
    pinv_pred = torch.matmul(torch.pinverse(sources), pred)
    return torch.matmul(sources, pinv_pred)

def compute_sdr(pred, s_target):
    '''
    pred (length)
    s_target (length)
    '''
    e_total = pred - s_target
    return 10*torch.log10(torch.sum(s_target**2)/torch.sum(e_total**2))

def compute_sir(s_target, e_inter):
    '''
    pred (length)
    e_inter (length)
    '''
    return 10*torch.log10(torch.sum(s_target**2)/torch.sum(e_inter**2))

def compute_sar(s_target, e_inter, e_art):
    '''
    pred (length)
    e_art (length)
    '''
    source_projection = s_target + e_inter
    return 10*torch.log10(torch.sum(source_projection**2)/torch.sum(e_art**2))

def bss_eval(pred, sources):
    '''
    BSS eval metric calculation.
    pred (length)
    sources (sources, length)
    target_idx (int) index of target in sources
    '''
    sources = torch.t(sources)
    target = sources[:, 0]
    s_target = compute_s_target(pred, target)
    source_proj = compute_source_projection(pred, sources)
    e_inter = source_proj - s_target
    e_art = pred - source_proj
    sdr = compute_sdr(pred, s_target).item()
    sir = compute_sir(s_target, e_inter).item()
    sar = compute_sar(s_target, e_inter, e_art).item()
    return sdr, sir, sar

def bss_eval_batch(preds, source_tensor, target_idx=0):
    '''
    preds (batch, length)
    source_tensor (batch, sources, length)
    '''
    metrics = BSSMetricsList()
    for i in range(preds.size(0)):
        pred = preds[i]
        sources = source_tensor[i]
        sdr, sir, sar = bss_eval(pred, sources)
        metrics.sdrs.append(sdr)
        metrics.sirs.append(sir)
        metrics.sars.append(sar)
    return metrics

class BSSEvaluate(nn.Module):
    def __init__(self, fs=16000):
        super().__init__()
        self.stoi = sepcosts.ShortTimeObjectiveIntelligibility(
            fs=fs,
            return_loss=False
        )

    def forward(self, preds, source_tensor, target_idx=0):
        metrics = bss_eval_batch(preds, source_tensor).cpu()
        metrics.stois = list(self.stoi(preds, source_tensor[:, 0],
                            source_tensor[:, 1]).cpu())
        return metrics

class LossMetrics():
    '''
    Data struct that keeps track of all losses and metrics during the training process
    '''
    def __init__(self, bss=True):
        self.time = []
        self.train_loss = []
        self.val_loss = []
        self.bss = bss
        if bss:
            self.sdrs = []
            self.sirs = []
            self.sars = []
            self.stois = []

    def update(self, train_loss, val_loss, sdr=None, sir=None, sar=None,
        stoi=None, period=1):
        if self.time:
            self.time.append(self.time[-1] + period)
        else:
            self.time = [0]
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        if self.bss:
            self.sdrs.append(sdr)
            self.sirs.append(sir)
            self.sars.append(sar)
            self.stois.append(stoi)

def train_plot(vis, loss_metrics, eid=None, win=[None, None]):
    '''
    Plots loss and metrics during the training process
    '''
    # Loss plots
    data1 = [
        dict(
            x=loss_metrics.time, y=loss_metrics.train_loss, name='Training Loss',
            hoverinfo='y', line=dict(width=1), mode='lines', type='scatter'),
        dict(
            x=loss_metrics.time, y=loss_metrics.val_loss, name='Validation Loss',
            hoverinfo='y', line=dict( width=1), mode='lines', type='scatter')
    ]
    layout1 = dict(
        showlegend=True,
        legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)'),
        margin=dict(r=30, b=40, l=50, t=50),
        font=dict(family='Bell Gothic Std'),
        xaxis=dict(autorange=True, title='Training Epochs'),
        yaxis=dict(autorange=True, title='Loss'),
        title=win[0]
    )
    vis._send(dict(data=data1, layout=layout1, win=win[0], eid=eid))

    if loss_metrics.bss:
        # BSS_EVAL plots
        data2 = [
            # SDR
            dict(
                x=loss_metrics.time, y=loss_metrics.sdrs, name='SDR',
                hoverinfo='name+y+lines', line=dict(width=1), mode='lines',
                type='scatter'
            ),
            # SIR
            dict(
                x=loss_metrics.time, y=loss_metrics.sirs, name='SIR',
                hoverinfo='name+y+lines', line=dict(width=1), mode='lines',
                type='scatter'
            ),
            # SAR
            dict(
                x=loss_metrics.time, y=loss_metrics.sars, name='SAR',
                hoverinfo='name+y+lines', line=dict(width=1), mode='lines',
                type='scatter'
            ),
            # STOI
            dict(
                x=loss_metrics.time, y=loss_metrics.stois, name='STOI',
                hoverinfo='name+y+lines', line=dict(width=1), mode='lines',
                type='scatter', yaxis='y2'
            ),

        ]
        layout2 = dict(
            showlegend=True,
            legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
            margin=dict(r=30, b=40, l=50, t=50),
            font=dict(family='Bell Gothic Std'),
            xaxis=dict(autorange=True, title='Training samples'),
            yaxis=dict(autorange=True, title='dB'),
            yaxis2=dict(autorange=True, title='STOI', overlaying='y', side='right'),
            title=win[1]
        )
        vis._send(dict(data=data2, layout=layout2, win=win[1], eid=eid))

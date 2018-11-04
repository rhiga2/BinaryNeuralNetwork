import sys , os
sys.path.append('../')

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
        sdr = np.mean(self.sdrs)
        sir = np.mean(self.sirs)
        sar = np.mean(self.sars)
        return sdr, sir, sar

def compute_s_target(pred, target):
    '''
    pred (T)
    target (T)
    '''
    return torch.sum(target*pred)/\
        torch.sum(target**2)*target

def compute_source_projection(pred, sources):
    '''
    pred (T)
    sources (T, S)
    '''
    pinv_pred = torch.matmul(torch.pinverse(sources), pred)
    return torch.matmul(sources, pinv_pred)

def compute_sdr(pred, s_target):
    e_total = pred - s_target
    return 10*torch.log10(torch.sum(s_target**2)/torch.sum(e_total**2))

def compute_sir(s_target, e_inter):
    return 10*torch.log10(torch.sum(s_target**2)/torch.sum(e_inter**2))

def compute_sar(s_target, e_inter, e_art):
    source_projection = s_target + e_inter
    return 10*torch.log10(torch.sum(source_projection**2)/torch.sum(e_art**2))

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
    sdr = compute_sdr(pred, s_target).item()
    sir = compute_sir(s_target, e_inter).item()
    sar = compute_sar(s_target, e_inter, e_art).item()
    metric = BSSMetrics(sdr, sir, sar)
    return metric

def bss_eval_batch(preds, source_tensor, target_idx=0):
    '''
    preds (N, T)
    source_tensor (N, S, T)
    '''
    metrics = BSSMetricsList()
    for i in range(preds.size()[0]):
        pred = preds[i]
        sources = source_tensor[i]
        metric = bss_eval(pred, sources, target_idx)
        metrics.append(metric)
    return metrics

class LossMetrics():
    '''
    Data struct that keeps track of all losses and metrics during the training process
    '''
    def __init__(self):
        self.time = []
        self.train_loss = []
        self.val_loss = []
        self.sdrs = []
        self.sirs = []
        self.sars = []

    def update(self, train_loss, val_loss, sdr, sir, sar, output_period=1):
        if self.time:
            self.time.append(self.time[-1] + output_period)
        else:
            self.time = [0]
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.sdrs.append(sdr)
        self.sirs.append(sir)
        self.sars.append(sar)

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

    # BSS_EVAL plots
    data2 = [
        # SDR
        dict(
            x=loss_metrics.time, y=loss_metrics.sdrs, name='SDR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
        # SIR
        dict(
            x=loss_metrics.time, y=loss_metrics.sirs, name='SIR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
        # SAR
        dict(
            x=loss_metrics.time, y=loss_metrics.sars, name='SAR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
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

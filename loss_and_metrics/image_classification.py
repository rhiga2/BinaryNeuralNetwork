import sys , os
sys.path.append('../')

import numpy as np

class LossMetrics():
    '''
    Data struct that keeps track of all losses and metrics during the training process
    '''
    def __init__(self, bss=True):
        self.time = []
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

    def update(self, train_loss, train_accuracy, val_loss,
        val_accuracy, period=1):
        if self.time:
            self.time.append(self.time[-1] + period)
        else:
            self.time = [0]
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_accuracy.append(train_accuracy)
        self.val_accuracy.append(val_accuracy)

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

    data2 = [
        # SDR
        dict(
            x=loss_metrics.time, y=loss_metrics.train_accuracy, name='Train Accuracy',
            hoverinfo='name+y+lines', line=dict(width=1), mode='lines', type='scatter'
        ),
        dict(
            x=loss_metrics.time, y=loss_metrics.val_accuracy, name='Validation Accuracy',
            hoverinfo='name+y+lines', line=dict(width=1), mode='lines', type='scatter'
        )
    ]
    layout2 = dict(
        showlegend=True,
        legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
        margin=dict(r=30, b=40, l=50, t=50),
        font=dict(family='Bell Gothic Std'),
        xaxis=dict(autorange=True, title='Training Epochs'),
        yaxis=dict(autorange=True, title='Accuracy'),
        title=win[1]
    )
    vis._send(dict(data=data2, layout=layout2, win=win, eid=eid))

def plot_weights(vis, weights, numbins=30, win='temp', title='temp'):
    vis.histogram(X=weights, opts=dict(title=title, numbins=numbins), win=win)

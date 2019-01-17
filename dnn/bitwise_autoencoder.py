import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from datasets.make_data import *
from datasets.quantized_data import *
from datasets.two_source_mixture import *
from loss_and_metrics.sepcosts import *
from loss_and_metrics.bss_eval import *
from dnn.binary_layers import *
import visdom
import argparse

class BitwiseAutoencoder(nn.Module):
    '''
    Adaptive transform network inspired by Minje Kim
    '''
    def __init__(self, kernel_size=256, stride=16, in_channels=1,
        out_channels=1, fc_sizes = [], dropout=0, sparsity=95,
        adapt=True, autoencode=False, use_gate=True, scale=1.0,
        activation=torch.tanh):
        super(BitwiseAutoencoder, self).__init__()

        # Initialize adaptive front end
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(1, kernel_size, kernel_size, stride=stride,
            padding=kernel_size)
        self.conv1.weight = nn.Parameter(self.conv1.weight * scale,
            requires_grad=True)
        self.autoencode = autoencode
        self.activation = activation
        self.batchnorm = nn.BatchNorm1d(kernel_size)

        # Initialize inverse of front end transform
        self.conv1_transpose = nn.ConvTranspose1d(kernel_size, 1, kernel_size, stride=stride)
        self.sparsity = sparsity
        self.mode = 'real'

    def forward(self, x):
        '''
        Bitwise neural network forward
        * Input is a tensor of shape (batch, channels, time)
        * Output is a tensor of shape (batch, channels, time)
            - batch is the batch size
            - time is the sequence length
            - channels is the number of input channels = num bits in qad
        '''
        time = x.size(2)
        h = self.activation(self.batchnorm(self.conv1(x)))
        h = self.conv1_transpose(h)[:, :, self.kernel_size:time+self.kernel_size]
        return h

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.mode != 'noisy':
            return

        if not self.autoencode:
            for layer in self.linear_list:
                layer.update_beta(sparsity=self.sparsity)

def train(model, dl, optimizer, loss=F.mse_loss, device=torch.device('cpu'),
    autoencode=False, quantizer=None, transform=None, dtype=torch.float):
    running_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        mix, target = batch['mixture'], batch['interference']
        if autoencode:
            mix = target
        if quantizer:
            mix = quantizer(mix).to(device=device, dtype=dtype) / 255
            target = quantizer(target).to(device=device, dtype=torch.long)
        if transform:
            mix = transform(mix)
        else:
            mix = mix.unsqueeze(1)
        mix = mix.to(device=device)
        target = target.to(device=device)
        estimate = model(mix)
        estimate = estimate.squeeze(1)
        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss.item() * mix.size(0)
        reconst_loss.backward()
        optimizer.step()
    return running_loss / len(dl.dataset)

def val(model, dl, loss=F.mse_loss, autoencode=False,
    quantizer=None, transform=None, device=torch.device('cpu'),
    dtype=torch.float):
    running_loss = 0
    bss_metrics = BSSMetricsList()
    for batch in dl:
        mix, target = batch['mixture'], batch['target']
        if autoencode:
            mix = target
        if quantizer:
            mix = quantizer(mix).to(device=device, dtype=dtype) / 255
            target = quantizer(target).to(device=device, dtype=torch.long)
        if transform:
            mix = transform(mix)
        else:
            mix = mix.unsqueeze(1)
        mix = mix.to(device=device)
        target = target.to(device=device)
        estimate = model(mix)
        estimate = estimate.squeeze(1)
        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss.item() * mix.size(0)
        if quantizer:
            estimate = quantizer.inverse(estimate)
        estimate = estimate.to(device='cpu')
        sources = torch.stack([batch['target'], batch['interference']], dim=1)
        metrics = bss_eval_batch(estimate, sources)
        bss_metrics.extend(metrics)
    return running_loss / len(dl.dataset), bss_metrics

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--period', '-p', type=int, default=1)
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--toy', '-toy', action='store_true')
    parser.add_argument('--autoencode', '-autoencode', action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)

    parser.add_argument('--kernel', '-k', type=int, default=256)
    parser.add_argument('--stride', '-s', type=int, default=16)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--activation', '-a', default='tanh')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    activation = pick_activation(args.activation)

    # Initialize quantizer and dequantizer
    quantizer = None
    transform = None

    # Make model and dataset
    train_dl, val_dl, _ = make_data(args.batchsize, hop=args.stride, toy=args.toy)
    model = BitwiseAutoencoder(args.kernel, args.stride, fc_sizes=[2048, 2048],
        in_channels=1, out_channels=1, dropout=args.dropout,
        sparsity=args.sparsity, autoencode=args.autoencode,
        scale=1.0, activation=activation)
    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))
    model.to(device=device, dtype=dtype)
    print(model)

    loss = SignalDistortionRatio()
    loss_metrics = LossMetrics()

    # Initialize optimizer
    vis = visdom.Visdom(port=5800)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = train(model, train_dl, optimizer, loss=loss, device=device,
            autoencode=args.autoencode, quantizer=quantizer, transform=transform,
            dtype=dtype)

        if epoch % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, loss=loss, device=device,
                autoencode=args.autoencode, quantizer=quantizer,
                transform=transform, dtype=dtype)
            sdr, sir, sar = bss_metrics.mean()
            loss_metrics.update(train_loss, val_loss, sdr, sir, sar,
                output_period=args.period)
            train_plot(vis, loss_metrics, eid='Ryley', win=['Loss', 'BSS Eval'])
            print('Validation Cost: ', val_loss)
            print('Val SDR: ', sdr)
            print('Val SIR: ', sir)
            print('Val SAR: ', sar)
            torch.save(model.state_dict(), '../models/' + args.exp + '.model')
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

    with open('../results/' + args.exp + '.pkl', 'wb') as f:
        pkl.dump(loss_metrics, f)

if __name__ == '__main__':
    main()

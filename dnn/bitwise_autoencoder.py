import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import datasets.make_data as make_data
import datasets.quantized_data as quantized_data
import datasets.two_source_mixture as two_source_mixture
import loss_and_metrics.sepcosts as sepcosts
import loss_and_metrics.bss_eval as bss_eval
import dnn.binary_layers as binary_layers
import visdom
import argparse

def _haar_matrix(n):
    '''
    n is a power of 2
    Produce unnormalized haar
    '''
    assert n > 1
    if n == 2:
        return np.array([[1, 1], [1, -1]])
    prev_haar = haar_matrix(n // 2)
    prev_id = np.eye(n // 2)
    haar_top = np.kron(prev_haar, np.array([1, 1]))
    haar_bottom = np.kron(prev_id, np.array([1, -1]))
    return np.concatenate((haar_top, haar_bottom))

def haar_matrix(n):
    '''
    n is a power of 2
    '''
    haar = _haar_matrix(n)
    return haar / np.linalg.norm(haar, axis=1)

class BitwiseAutoencoder(nn.Module):
    '''
    Adaptive transform network inspired by Minje Kim
    '''
    def __init__(self, kernel_size=256, stride=16, in_channels=1,
        out_channels=1, fc_sizes = [], dropout=0, sparsity=95, adapt=True,
        autoencode=False, use_gate=True, in_bin=binary_layers.identity,
        weight_bin=binary_layers.identity):
        super(BitwiseAutoencoder, self).__init__()

        # Initialize adaptive front end
        self.kernel_size = kernel_size
        self.autoencode = autoencode
        self.conv = binary_layers.BitwiseConv1d(1, kernel_size, kernel_size,
            stride=stride, padding=kernel_size,
            in_bin=in_bin, weight_bin=weight_bin, adaptive_scaling=True, use_gate=True)

        # Initialize conv weights
        haar = torch.FloatTensor(haar_matrix(kernel_size))
        self.conv.weight = nn.Parameter(haar.unsqueeze(1), requires_grad=True)

        self.batchnorm = nn.BatchNorm1d(kernel_size)
        self.activation = nn.ReLU(inplace=True)

        # Initialize inverse of front end transform
        self.conv_transpose = binary_layers.BitwiseConvTranspose1d(
            kernel_size, 1, kernel_size, stride=stride, in_bin=in_bin,
            weight_bin=weight_bin, adaptive_scaling=True, use_gate=True
        )

        # Initialize conv transpose weights to FFT
        scale = kernel_size/stride
        invhaar = torch.t(haar)
        invhaar = invhaar.contiguous().unsqueeze(1)
        self.conv_transpose.weight = nn.Parameter(invhaar, requires_grad=True)

        self.sparsity = sparsity

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
        h = self.batchnorm(self.activation(self.conv(x)))
        h = self.conv_transpose(h)[:, :, self.kernel_size:time+self.kernel_size]
        return h

    def clip_weights(self):
        self.conv.clip_weights()
        self.conv_transpose.clip_weights()

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.sparsity == 0:
            return

        self.conv.update_betas(sparsity=args.sparsity)
        self.conv_transpose.update_betas(sparsity=args.sparsity)

def train(model, dl, optimizer, loss=F.mse_loss, device=torch.device('cpu'),
    autoencode=False, quantizer=None, transform=None, dtype=torch.float,
    clip_weights=False):
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
        if clip_weights:
            model.clip_weights()
    return running_loss / len(dl.dataset)

def val(model, dl, loss=F.mse_loss, autoencode=False,
    quantizer=None, transform=None, device=torch.device('cpu'),
    dtype=torch.float):
    running_loss = 0
    bss_metrics = bss_eval.BSSMetricsList()
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
        metrics = bss_eval.bss_eval_batch(estimate, sources)
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
    parser.add_argument('--autoencode', '-ae', action='store_true')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--clip_weights', '-cw', action='store_true')

    parser.add_argument('--kernel', '-k', type=int, default=256)
    parser.add_argument('--stride', '-s', type=int, default=16)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--in_bin', '-ib', default='identity')
    parser.add_argument('--weight_bin', '-wb', default='identity')
    parser.add_argument('--loss', '-l', default='mse')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Initialize quantizer and dequantizer
    quantizer = None
    transform = None

    in_bin = binary_layers.pick_activation(args.in_bin)
    weight_bin = binary_layers.pick_activation(args.weight_bin)

    # Make model and dataset
    train_dl, val_dl, _ = make_data.make_data(args.batchsize, hop=args.stride,
        toy=args.toy)
    model = BitwiseAutoencoder(args.kernel, args.stride, fc_sizes=[2048, 2048],
        in_channels=1, out_channels=1, dropout=args.dropout,
        sparsity=args.sparsity, autoencode=args.autoencode,
        in_bin=in_bin, weight_bin=weight_bin)
    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))
    model.to(device=device, dtype=dtype)
    print(model)

    if args.loss == 'mse':
        loss = nn.MSELoss()
    else:
        loss = sepcosts.SignalDistortionRatio()
    loss_metrics = bss_eval.LossMetrics()

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
            dtype=dtype, clip_weights=args.clip_weights)

        if epoch % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, loss=loss, device=device,
                autoencode=args.autoencode, quantizer=quantizer,
                transform=transform, dtype=dtype)
            sdr, sir, sar = bss_metrics.mean()
            loss_metrics.update(train_loss, val_loss, sdr, sir, sar,
                output_period=args.period)
            bss_eval.train_plot(vis, loss_metrics, eid='Ryley', win=['Loss', 'BSS Eval'])
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

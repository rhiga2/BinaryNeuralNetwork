import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import datasets.quantized_data as quantized_data
import datasets.two_source_mixture as two_source_mixture
import datasets.make_data as make_data
import dnn.binary_layers as binary_layers
import dnn.bitwise_adaptive_transform as adaptive_transform
import dnn.bitwise_tasnet as bitwise_tasnet
import dnn.bitwise_wavenet as bitwise_wavenet
import loss_and_metrics.bss_eval as bss_eval
import loss_and_metrics.sepcosts as sepcosts
import visdom
import argparse
import pickle as pkl

def train(model, dl, optimizer, loss=F.mse_loss, device=torch.device('cpu'),
    autoencode=False, quantizer=None, dtype=torch.float,
    clip_weights=False, classification=False):
    running_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        mix, target = batch['mixture'], batch['target']
        if autoencode:
            mix = target
        if quantizer:
            mix = quantizer(mix).to(device=device, dtype=dtype) / 255
            target = quantizer(target).to(device=device, dtype=torch.long).view(-1)
        mix = mix.unsqueeze(1)
        mix = mix.to(device=device)
        target = target.to(device=device)
        estimate = model(mix)

        if classification:
            estimate = estimate.permute(0, 2, 1).contiguous().view(-1, 256)
        else:
            estimate = estimate.squeeze(1)

        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss * mix.size(0)
        reconst_loss.backward()
        optimizer.step()
        if clip_weights:
            model.clip_weights()
    optimizer.zero_grad()
    return running_loss.item() / len(dl.dataset)

def val(model, dl, loss=F.mse_loss, autoencode=False,
    quantizer=None, device=torch.device('cpu'),
    dtype=torch.float, classification=False):
    running_loss = 0
    bss_metrics = bss_eval.BSSMetricsList()
    for batch in dl:
        mix, target, inter = batch['mixture'], batch['target'], batch['interference']
        features = mix
        labels = target
        if autoencode:
            features = target
        if quantizer:
            features = quantizer(features).to(device=device, dtype=dtype) / 255
            labels = quantizer(labels).to(device=device, dtype=torch.long).view(-1)
        features = features.unsqueeze(1)

        with torch.no_grad():
            features = features.to(device=device)
            labels = labels.to(device=device)
            estimate = model(features)
            estimate_size = estimate.size()

            if classification:
                estimate = estimate.permute(0, 2, 1).contiguous().view(-1, 256)
            else:
                estimate = estimate.squeeze(1)

            reconst_loss = loss(estimate, labels)
            running_loss += reconst_loss * mix.size(0)

            if classification:
                estimate = estimate.view(estimate_size[0], estimate_size[2], 256).contiguous().permute(0, 2, 1)

            if quantizer:
                estimate = torch.argmax(estimate, dim=1).to(dtype=dtype)
                estimate = quantizer.inverse(estimate)

        sources = torch.stack([target, inter], dim=1).to(device=device)
        metrics = bss_eval.bss_eval_batch(estimate, sources)
        bss_metrics.extend(metrics)
    return running_loss.item() / len(dl.dataset), bss_metrics

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--model', '-model', default='adaptive_transform')
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
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--in_bin', '-ib', default='identity')
    parser.add_argument('--weight_bin', '-wb', default='identity')
    parser.add_argument('--loss', '-l', default='mse')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--adaptive_scaling', '-as', action='store_true')
    parser.add_argument('--activation', '-a', default='tanh')
    parser.add_argument('--use_batchnorm', '-ub', action='store_true')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    activation = binary_layers.pick_activation(args.activation)
    in_bin = binary_layers.pick_activation(args.in_bin)
    weight_bin = binary_layers.pick_activation(args.weight_bin)

    classification = False
    stride=1
    if args.model == 'wavenet':
        filter_activation = binary_layers.pick_activation(args.activation)
        gate_activation = binary_layers.pick_activation(args.activation,
            bipolar_shift=False)
        model = bitwise_wavenet.BitwiseWavenet(1, 256,
            kernel_size=2, filter_activation=torch.tanh,
            gate_activation=torch.sigmoid, in_bin=in_bin, weight_bin=weight_bin,
            adaptive_scaling=args.adaptive_scaling, use_gate=args.use_gate,
            use_batchnorm=args.use_batchnorm)
    elif args.model == 'tasnet':
        stride=10
        bitwise_tasnet.BitwiseTasnet(1, 256,
            256, 512, blocks=4, front_kernel_size=20, front_stride=10,
            kernel_size=3, layers=8, in_bin=in_bin, weight_bin=weight_bin,
            adaptive_scaling=args.adaptive_scaling, use_gate=args.use_gate,
            use_batchnorm=args.use_batchnorm)
    else:
        stride=16
        model = adaptive_transform.BitwiseAdaptiveTransform(1024, 16,
            fc_sizes=[2048, 2048], in_channels=1, out_channels=1,
            dropout=args.dropout, sparsity=args.sparsity,
            autoencode=args.autoencode, in_bin=in_bin, weight_bin=weight_bin,
            adaptive_scaling=args.adaptive_scaling, use_gate=args.use_gate,
            activation=activation, weight_init='fft')

    # Make model and dataset
    train_dl, val_dl, _ = make_data.make_data(args.batchsize, hop=stride,
        toy=args.toy, max_duration=2, transform=lambda x : signal.decimate(x, 2))

    if args.load_file:
        model.load_partial_state_dict(torch.load('../models/' + args.load_file))
    model.to(device=device, dtype=dtype)
    print(model)

    quantizer = None
    if args.loss == 'mse':
        loss = nn.MSELoss()
    elif args.loss == 'sdr':
        loss = sepcosts.SignalDistortionRatio()
    elif args.loss == 'cel':
        quantizer = quantized_data.Quantizer()
        loss = nn.CrossEntropyLoss()
        classification = True
    elif args.loss == 'sisnr':
        loss = nn.SISNRLoss()
    loss_metrics = bss_eval.LossMetrics()

    # Initialize optimizer
    vis = visdom.Visdom(port=5801)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = 0
        train_loss = train(model, train_dl, optimizer, loss=loss, device=device,
            autoencode=args.autoencode, quantizer=quantizer,
            dtype=dtype, clip_weights=args.clip_weights,
            classification=classification)

        if epoch % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, loss=loss, device=device,
                autoencode=args.autoencode, quantizer=quantizer,
                dtype=dtype, classification=classification)
            sdr, sir, sar = bss_metrics.mean()
            loss_metrics.update(train_loss, val_loss, sdr, sir, sar,
                output_period=args.period)
            bss_eval.train_plot(vis, loss_metrics, eid='Ryley',
                win=['{} Loss'.format(args.exp), '{} BSS Eval'.format(args.exp)])
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

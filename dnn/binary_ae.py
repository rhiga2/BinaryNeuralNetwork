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
import dnn.bitwise_autoencoder as bitwise_autoencoder
import dnn.bitwise_wavenet as bitwise_wavenet
import loss_and_metrics.bss_eval as bss_eval
import visdom
import argparse
import pickle as pkl

def train(model, dl, optimizer, loss=F.mse_loss, device=torch.device('cpu'),
    autoencode=False, quantizer=None, transform=None, dtype=torch.float,
    clip_weights=False, classification=False):
    running_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        mix, target = batch['mixture'], batch['interference']
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
        running_loss += reconst_loss.item() * mix.size(0)
        reconst_loss.backward()
        optimizer.step()
        if clip_weights:
            model.clip_weights()
    optimizer.zero_grad()
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
            target = quantizer(target).to(device=device, dtype=torch.long).view(-1)
        mix = mix.unsqueeze(1)
        mix = mix.to(device=device)
        target = target.to(device=device)
        with torch.no_grad():
            estimate = model(mix)
            estimate_size = estimate.size()
            if classification:
                estimate = estimate.permute(0, 2, 1).contiguous().view(-1, 256)
            else:
                estimate = estimate.squeeze(1)

            reconst_loss = loss(estimate, target)
            running_loss += reconst_loss.item() * mix.size(0)
            estimate = estimate.view(estimate_size[0], estimate_size[2], 256).contiguous().permute(0, 2, 1)

            if quantizer:
                estimate = torch.argmax(estimate, dim=1).to(dtype=dtype)
                estimate = quantizer.inverse(estimate)

        estimate = estimate.to(device='cpu')
        sources = torch.stack([batch['target'], batch['interference']], dim=1)
        metrics = bss_eval.bss_eval_batch(estimate, sources)
        bss_metrics.extend(metrics)
    return running_loss / len(dl.dataset), bss_metrics

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--wavenet', '-wavenet', action='store_true')
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
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--adaptive_scaling', '-as', action='store_true')
    parser.add_argument('--activation', '-a', default='tanh')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    in_bin = binary_layers.pick_activation(args.in_bin)
    weight_bin = binary_layers.pick_activation(args.weight_bin)

    # Make model and dataset
    train_dl, val_dl, _ = make_data.make_data(args.batchsize, hop=args.stride,
        toy=args.toy, max_length=24000)
    classification = False
    if args.wavenet:
        filter_activation = binary_layers.pick_activation(args.activation)
        gate_activation = binary_layers.pick_activation(args.activation,
            bipolar_shift=False)
        model = bitwise_wavenet.BitwiseWavenet(1, 256,
            kernel_size=args.kernel, filter_activation=torch.tanh,
            gate_activation=torch.sigmoid, in_bin=in_bin, weight_bin=weight_bin,
            adaptive_scaling=args.adaptive_scaling, use_gate=args.use_gate)
        classification = True
    else:
        model = bitwise_autoencoder.BitwiseAutoencoder(args.kernel, args.stride,
            fc_sizes=[2048, 2048], in_channels=1, out_channels=1,
            dropout=args.dropout, sparsity=args.sparsity,
            autoencode=args.autoencode, in_bin=in_bin, weight_bin=weight_bin,
            adaptive_scaling=args.adaptive_scaling, use_gate=args.use_gate,
            activation=args.activation)

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
    loss_metrics = bss_eval.LossMetrics()

    # Initialize optimizer
    vis = visdom.Visdom(port=5801)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = train(model, train_dl, optimizer, loss=loss, device=device,
            autoencode=args.autoencode, quantizer=quantizer, transform=transform,
            dtype=dtype, clip_weights=args.clip_weights,
            classification=classifcation)

        if epoch % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, loss=loss, device=device,
                autoencode=args.autoencode, quantizer=quantizer,
                transform=transform, dtype=dtype, classifcation=classification)
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

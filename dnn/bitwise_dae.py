import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import scipy.signal as signal
import datasets.quantized_data as quantized_data
import datasets.two_source_mixture as two_source_mixture
import datasets.utils as utils
import dnn.binary_layers as binary_layers
import dnn.bitwise_adaptive_transform as adaptive_transform
import dnn.bitwise_tasnet as bitwise_tasnet
import dnn.bitwise_wavenet as bitwise_wavenet
import loss_and_metrics.bss_eval as bss_eval
import loss_and_metrics.sepcosts as sepcosts
import visdom
import argparse
import pickle as pkl

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
    parser.add_argument('--lr_decay', '-lrd', type=float, default=0.9)
    parser.add_argument('--decay_period', '-dp', type=int, default=10)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--clip_weights', '-cw', action='store_true')
    parser.add_argument('--decimate', '-decimate', action='store_true')

    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--in_binactiv', '-ib', default='identity')
    parser.add_argument('--w_binactiv', '-wb', default='identity')
    parser.add_argument('--loss', '-l', default='mse')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    in_binactiv = binary_layers.pick_activation(args.in_binactiv)
    w_binactiv = binary_layers.pick_activation(args.w_binactiv)

    classification = False
    stride=1
    if args.model == 'wavenet':
        filter_activation = binary_layers.pick_activation(args.activation)
        gate_activation = binary_layers.pick_activation(args.activation,
            bipolar_shift=False)
        model = bitwise_wavenet.BitwiseWavenet(
            1, 256, kernel_size=2, filter_activation=torch.tanh,
            gate_activation=torch.sigmoid, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv,
            use_gate=args.use_gate,
        )
    elif args.model == 'tasnet':
        stride=10
        model = bitwise_tasnet.BitwiseTasNet(
            1, 256, 512,
            blocks=4, front_kernel_size=20, front_stride=10,
            kernel_size=3, layers=8, in_binactiv=in_binactiv, w_binactiv=w_binactiv,
            use_gate=args.use_gate
        )
    else:
        stride=32
        model = adaptive_transform.BitwiseAdaptiveTransform(
            1024, stride, fc_sizes=[2048, 2048, 2048],
            in_channels=1, out_channels=1,
            dropout=args.dropout, sparsity=args.sparsity,
            autoencode=args.autoencode, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv,
            use_gate=args.use_gate, weight_init='fft'
        )

    # Make model and dataset
    train_dl, val_dl, _ = utils.get_data_from_directory(args.batchsize,
        '/media/data/wsj_mix/decimated2/', template='sample*.npz',
        return_dls=True)

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
        loss = sepcosts.SISNRLoss()
    loss_metrics = bss_eval.LossMetrics()

    # Initialize optimizer
    vis = visdom.Visdom(port=5801)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    def forward(model, dl, train=False, autoencode=False, clip_weights=False,
        classification=False):
        running_loss = 0
        bss_metrics = bss_eval.BSSMetricsList()
        for batch in dl:
            if train:
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
            running_loss += reconst_loss.item() * mix.size(0)
            if train:
                reconst_loss.backward()
                optimizer.step()
                if clip_weights:
                    model.clip_weights()
            else:
                sources = torch.stack([target, inter], dim=1).to(device=device)
                metrics = bss_evaluate(estimate, sources)
                bss_metrics.extend(metrics)
        if train:
            optimizer.zero_grad()
        return running_loss / len(dl.dataset), bss_metrics

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = 0
        train_loss, _ = forward(model, train_dl, train=True,
            autoencode=args.autoencode, clip_weights=args.clip_weights,
            classification=classification)

        if epoch % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, train=False,
                autoencode=args.autoencode, classification=classification)
            sdr, sir, sar, stoi = bss_metrics.mean()
            loss_metrics.update(train_loss, val_loss, sdr, sir, sar,
                period=args.period)
            bss_eval.train_plot(vis, loss_metrics, eid='Ryley',
                win=['{} Loss'.format(args.exp), '{} BSS Eval'.format(args.exp)])
            print('Validation Cost: ', val_loss)
            print('Val SDR: ', sdr)
            print('Val SIR: ', sir)
            print('Val SAR: ', sar)
            print('Val SAR: ', stoi)
            torch.save(model.state_dict(), '../models/' + args.exp + '.model')
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

        if (epoch+1) % args.decay_period == 0 and args.lr_decay != 1:
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    with open('../results/' + args.exp + '.pkl', 'wb') as f:
        pkl.dump(loss_metrics, f)

if __name__ == '__main__':
    main()

import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import datasets.binary_data as binary_data
import datasets.stft as stft
import loss_and_metrics.bss_eval as bss_eval
import dnn.binary_layers as binary_layers
import dnn.bitwise_mlp as bitwise_mlp
import soundfile as sf
from dnn.solvers import BinarySTFTSolver
import visdom
import pickle as pkl
import argparse

def mean_squared_error(estimate, target, weight=None):
    if weight is not None:
        return 0.5 * torch.mean(weight*(estimate - target)**2)
    else:
        return 0.5 * torch.mean((estimate - target)**2)

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=128,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--load_file', '-lf', type=str, default=None)

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--decay_period', '-dp', type=int, default=10)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0)
    parser.add_argument('--period', '-p', type=int, default=1)
    parser.add_argument('--loss', '-l', type=str, default='bce')
    parser.add_argument('--weighted', '-w', action='store_true')

    parser.add_argument('--sparsity', '-s', type=float, default=0)
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--activation', '-a', default='tanh')
    parser.add_argument('--in_binactiv', '-ib', default='identity')
    parser.add_argument('--w_binactiv', '-wb', default='identity')
    parser.add_argument('--clip_weights', '-cw', action='store_true')
    parser.add_argument('--bn_momentum', '-bnm', type=float, default=0.1)
    args = parser.parse_args()

    # Initialize device
    if torch.cuda.is_available():
        device = torch.device('cuda:'+ str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Initialize loss function
    if args.loss == 'mse':
        loss = mean_squared_error
    else:
        loss = F.binary_cross_entropy_with_logits
    loss_metrics = bss_eval.LossMetrics()

    in_binactiv = binary_layers.pick_activation(args.in_binactiv)
    w_binactiv = binary_layers.pick_activation(args.w_binactiv)

    # Make model and dataset
    train_dl, val_dl, raw_dl = binary_data.get_binary_data(args.batchsize, toy=args.toy)
    model = bitwise_mlp.BitwiseMLP(
        in_size=2052,
        out_size=513,
        fc_sizes=[2048, 2048, 2048],
        dropout=args.dropout,
        sparsity=args.sparsity,
        use_gate=args.use_gate,
        in_binactiv=in_binactiv,
        w_binactiv=w_binactiv,
        bn_momentum=args.bn_momentum,
        bias=False,
        scale_weights=None,
        scale_activations=None
    )

    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))

    model.to(device=device)
    print(model)

    # Initialize optimizer
    vis = visdom.Visdom(port=5801)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr,
        weight_decay=args.weight_decay)
    solver = BinarySTFTSolver(model, loss, optimizer, args.weighted)
    scheduler = optim.lr_scheduler.StepLR(solver.optimizer, args.decay_period,
        gamma=args.lr_decay)

    max_sdr = 0
    for epoch in range(args.epochs):
        scheduler.step()
        total_cost = 0
        model.update_betas()
        train_loss = solver.train(train_dl, clip_weights=args.clip_weights)

        if (epoch+1) % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            val_loss, val_metrics = solver.eval(val_dl, raw_dl=raw_dl)
            print('Val Cost: %f' % val_loss)
            sdr, sir, sar, stoi = val_metrics.mean()
            print('SDR: ', sdr)
            print('SIR: ', sir)
            print('SAR: ', sar)
            print('STOI: ', stoi)
            loss_metrics.update(train_loss, val_loss,
                sdr, sir, sar, stoi, period=args.period)
            bss_eval.train_plot(vis, loss_metrics, eid='Ryley',
                win=['{} Loss'.format(args.exp),
                '{} BSS Eval'.format(args.exp)])
            if sdr > max_sdr:
                max_sdr = sdr
                torch.save(model.state_dict(), '../models/' + args.exp + '.model')

    with open('../results/' + args.exp + '.pkl', 'wb') as f:
        pkl.dump(loss_metrics, f)

if __name__ == '__main__':
    main()

import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import datasets.binary_data as binary_data
import loss_and_metrics.bss_eval as bss_eval
import dnn.binary_layers as binary_layers
import dnn.bitwise_mlp as bitwise_mlp
import soundfile as sf
import visdom
import pickle as pkl
import argparse

def train(model, dl, optimizer=None, loss=F.mse_loss, device=torch.device('cpu'),
    dtype=torch.float, weighted=False, clip_weights=False):
    running_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        bmag = batch['bmag'].to(device=device)
        ibm = batch['ibm'].to(device=device)
        bmag = bmag.to(device=device)
        bmag_size = bmag.size()
        bmag = 2*bmag - 1 
        bmag = bitwise_mlp.flatten(bmag)
        estimate = model(bmag)
        estimate = bitwise_mlp.unflatten(estimate, bmag_size[0], bmag_size[2])
        if weighted:
            spec = batch['spec'].to(device=device)
            spec = spec / torch.std(spec)
            cost = loss(estimate, ibm, weight=spec)
        else:
            cost = loss(estimate, ibm)
        running_loss += cost.item() * bmag_size[0]
        cost.backward()
        optimizer.step()
        if clip_weights:
            model.clip_weights()
    optimizer.zero_grad()
    return running_loss / len(dl.dataset)

def evaluate(model, dataset, rawset, loss=F.mse_loss, max_samples=400,
    device=torch.device('cpu'), weighted=False):
    bss_metrics = bss_eval.BSSMetricsList()
    running_loss = 0
    for i in range(len(dataset)):
        if i >= max_samples:
            return running_loss / max_samples, bss_metrics

        raw_sample = rawset[i]
        bin_sample = dataset[i]
        mix = raw_sample['mix']
        target = raw_sample['target']
        interference = raw_sample['interference']
        mix_mag, mix_phase = binary_data.stft(mix)
        with torch.no_grad():
            bmag = torch.FloatTensor(bin_sample['bmag']).unsqueeze(0).to(device)
            ibm = torch.FloatTensor(bin_sample['ibm']).unsqueeze(0).to(device)
            bmag_size = bmag.size()
            bmag = 2*bmag - 1
            bmag = bitwise_mlp.flatten(bmag)
            premask = model(bmag)
            premask = bitwise_mlp.unflatten(premask, bmag_size[0], bmag_size[2])
            if weighted:
                spec =  bin_sample['spec']
                spec = torch.FloatTensor(spec).to(device)
                spec = spec / torch.std(spec)
                cost = loss(premask, ibm, weight=spec)
            else:
                cost = loss(premask, ibm)
            running_loss += cost.item()
        mask = binary_data.make_binary_mask(premask).squeeze(0).cpu()
        estimate = binary_data.istft(mix_mag * mask.numpy(), mix_phase)
        sources = np.stack([target, interference], axis=0)
        metric = bss_eval.bss_eval_np(estimate, sources)
        bss_metrics.append(metric)
    return running_loss / len(dataset), bss_metrics

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
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--period', '-p', type=int, default=1)
    parser.add_argument('--loss', '-l', type=str, default='bce')
    parser.add_argument('--weighted', '-w', action='store_true')

    parser.add_argument('--sparsity', '-s', type=float, default=0)
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--use_batchnorm', '-ub', action='store_true')
    parser.add_argument('--activation', '-a', default='tanh')
    parser.add_argument('--weight_bin', '-wb', default='tanh')
    parser.add_argument('--in_bin', '-ib', default='tanh')
    parser.add_argument('--clip_weights', '-cw', action='store_true')
    parser.add_argument('--bn_momentum', '-bnm', type=float, default=0.1)
    parser.add_argument('--adaptive_scaling', '-as', action='store_true')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
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

    activation = binary_layers.pick_activation(args.activation)
    weight_bin = binary_layers.pick_activation(args.weight_bin)
    in_bin = binary_layers.pick_activation(args.in_bin)

    # Make model and dataset
    train_dl, valset, rawset = binary_data.make_binary_data(args.batchsize, toy=args.toy)
    model = bitwise_mlp.BitwiseMLP(
        in_size=2052,
        out_size=513,
        fc_sizes=[2048, 2048],
        dropout=args.dropout,
        sparsity=args.sparsity,
        use_gate=args.use_gate,
        use_batchnorm=args.use_batchnorm,
        activation=activation,
        weight_bin=weight_bin,
        in_bin=in_bin,
        bn_momentum=args.bn_momentum,
        adaptive_scaling=args.adaptive_scaling
    )

    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))

    model.to(device=device)
    print(model)

    # Initialize optimizer
    vis = visdom.Visdom(port=5801)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = train(model, train_dl, optimizer, loss=loss,
            device=device, weighted=args.weighted, clip_weights=args.clip_weights)

        if (epoch+1) % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, val_metrics = evaluate(model, valset, rawset, loss=loss,
                weighted=args.weighted, device=device)
            print('Val Cost: %f' % val_loss)
            sdr, sir, sar = val_metrics.mean()
            print('SDR: ', sdr)
            print('SIR: ', sir)
            print('SAR: ', sar)
            loss_metrics.update(train_loss, val_loss,
                sdr, sir, sar, output_period=args.period)
            bss_eval.train_plot(vis, loss_metrics, eid='Ryley', win=['{} Loss'.format(args.exp), 
                '{} BSS Eval'.format(args.exp)])
            torch.save(model.state_dict(), '../models/' + args.exp + '.model')

    with open('../results/' + args.exp + '.pkl', 'wb') as f:
        pkl.dump(loss_metrics, f)

if __name__ == '__main__':
    main()

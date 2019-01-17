import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from datasets.binary_data import *
from datasets.quantized_data import *
from datasets.two_source_mixture import *
from loss_and_metrics.sepcosts import *
from loss_and_metrics.bss_eval import *
from dnn.bitwise_mlp import *
from dnn.binarized_network import *
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
        spec = batch['spec'].to(device=device)
        spec = spec / torch.std(spec)
        bmag = bmag.to(device=device)
        model_in = flatten(bmag)
        estimate = model(model_in)
        estimate = unflatten(estimate, bmag.size(0), bmag.size(2))
        if weighted:
            cost = loss(estimate, ibm, weight=spec)
        else:
            cost = loss(estimate, ibm)
        running_loss += cost.item() * bmag.size(0)
        cost.backward()
        optimizer.step()
        if clip_weights:
            model.clip_weights()
    return running_loss / len(dl.dataset)

def evaluate(model, dataset, rawset, loss=F.mse_loss, max_samples=400,
    device=torch.device('cpu'), weighted=False):
    bss_metrics = BSSMetricsList()
    running_loss = 0
    for i in range(len(dataset)):
        if i >= max_samples:
            return bss_metrics

        raw_sample = rawset[i]
        bin_sample = dataset[i]
        mix = raw_sample['mix']
        target = raw_sample['target']
        interference = raw_sample['interference']
        mix_mag, mix_phase = stft(mix)

        bmag = torch.FloatTensor(bin_sample['bmag']).unsqueeze(0).to(device)
        ibm = torch.FloatTensor(bin_sample['ibm']).unsqueeze(0).to(device)
        spec = bin_sample['spec']
        weights = torch.FloatTensor(spec).to(device)
        weights = weights / torch.std(weights)

        model_in = flatten(bmag)
        premask = model(model_in)
        premask = unflatten(premask, bmag.size(0), bmag.size(2))
        if weighted:
            cost = loss(premask, ibm, weight=weights)
        else:
            cost = loss(premask, ibm)
        running_loss += cost.item()
        mask = make_binary_mask(premask).squeeze(0).cpu()
        estimate = istft(mix_mag * mask.numpy(), mix_phase)
        sources = np.stack([target, interference], axis=0)
        metric = bss_eval_np(estimate, sources)
        bss_metrics.append(metric)
    sf.write('estimate.wav', estimate, 16000)
    sf.write('target.wav', target, 16000)
    return running_loss / len(dataset), bss_metrics

def flatten(x):
    batch, channels, time = x.size()
    x = x.permute(0, 2, 1).contiguous().view(-1, channels)
    return x

def unflatten(x, batch, time, permutation=(0, 2, 1)):
    x = x.view(batch, time, -1)
    x = x.permute(*permutation).contiguous()
    return x

def mean_squared_error(estimate, target, weight=None):
    if weight is not None:
        return 0.5 * torch.mean(weight*(estimate - target)**2)
    else:
        return 0.5 * torch.mean((estimate - target)**2)

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=256,
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

    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--use_batchnorm', '-ub', action='store_true')
    parser.add_argument('--activation', '-a', default='tanh')
    parser.add_argument('--clip_weights', '-cw', action='store_true')
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
    loss_metrics = LossMetrics()

    # Initialize activation
    activation = torch.tanh
    if args.activation == 'ste':
        activation = ste
    elif args.activation == 'clipped_ste':
        activation = clipped_ste
    elif args.activation == 'bitwise_activation':
        activation = bitwise_activation
    elif args.activation == 'relu':
        activation = nn.ReLU()

    # Make model and dataset
    train_dl, valset, rawset = make_binary_data(args.batchsize, toy=args.toy)
    model = BitwiseMLP(in_size=2052, out_size=513, fc_sizes=[2048, 2048],
        dropout=args.dropout, sparsity=args.sparsity, use_gate=args.use_gate,
        use_batchnorm=args.use_batchnorm, activation=activation)
    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))
    else:
        print('Real Network Training')
    model.to(device=device)
    print(model)

    # Initialize optimizer
    vis = visdom.Visdom(port=5800)
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
            train_plot(vis, loss_metrics, eid='Ryley', win=['Loss', None])
            torch.save(model.state_dict(), '../models/' + args.exp + '.model')

    with open('../results/' + args.exp + '.pkl', 'wb') as f:
        pkl.dump(loss_metrics, f)

if __name__ == '__main__':
    main()

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
import visdom
import argparse

def evaluate(model, dl, optimizer=None, loss=F.mse_loss, device=torch.device('cpu'),
    dtype=torch.float, train=True, weighted=False):
    running_loss = 0
    for batch in dl:
        if optimizer:
            optimizer.zero_grad()
        mix = batch['bmag'].to(device=device)
        target = batch['ibm'].to(device=device)
        spec = batch['spec'].to(device=device)
        spec = spec / torch.std(spec)
        mix = mix.to(device=device)
        model_in = flatten(mix)
        estimate = model(model_in)
        estimate = unflatten(estimate, mix.size(0), mix.size(2))
        if weighted:
            cost = loss(estimate, target, weight=spec)
        else:
            cost = loss(estimate, target)
        running_loss += cost.item() * mix.size(0)
        if optimizer:
            cost.backward()
            optimizer.step()
    return running_loss / len(dl.dataset)

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
        return torch.mean(weight*(estimate - target)**2)
    else:
        return torch.mean((estimate - target)**2)

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=256,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--train_noisy', '-tn',  action='store_true')
    parser.add_argument('--output_period', '-op', type=int, default=1)
    parser.add_argument('--update_temp', '-ut', action='store_true')
    parser.add_argument('--update_period', '-up', type=int, default=32)
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--l1_reg', '-l1r', type=float, default=0)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--model_file', '-mf', default='temp_model.model')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--use_noise', '-noise', action='store_true')
    parser.add_argument('--nobatchnorm', '--nobn', action='store_true')
    parser.add_argument('--loss', '-l', type=str, default='bce')
    parser.add_argument('--weighted', '-w', action='store_true')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:'+ str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Initialize loss function
    output_activation = False
    if args.loss == 'mse':   
        loss = mean_squared_error
        output_activation = True
    else:
        loss = F.binary_cross_entropy_with_logits
    loss_metrics = LossMetrics()

    # Make model and dataset
    train_dl, val_dl = make_binary_data(args.batchsize, toy=args.toy)
    model = BitwiseMLP(in_size=2052, out_size=513, fc_sizes=[2048, 2048],
        dropout=args.dropout, sparsity=args.sparsity, use_gate=args.use_gate,
        use_noise=args.use_noise, use_batchnorm=not args.nobatchnorm, 
        output_activation=output_activation)
    if args.train_noisy:
        print('Noisy Network Training')
        if args.load_file:
            model.load_state_dict(torch.load('../models/' + args.load_file))
        model.noisy()
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
        train_loss = evaluate(model, train_dl, optimizer, loss=loss, device=device, weighted=args.weighted)

        if (epoch+1) % args.output_period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss = evaluate(model, val_dl, loss=loss, device=device, weighted=args.weighted)
            print('Val Cost: ', val_loss)
            loss_metrics.update(train_loss, val_loss,
                output_period=args.output_period)
            train_plot(vis, loss_metrics, eid='Ryley', win=['Loss', None])
            torch.save(model.state_dict(), '../models/' + args.model_file)
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

        if (epoch+1) % args.update_period == 0 and args.update_temp:
            model.update_temp(model.temp*10)

if __name__ == '__main__':
    main()

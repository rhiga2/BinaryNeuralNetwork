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
from dnn.binary_layers import *
import visdom
import argparse

class BitwiseMLP(nn.Module):
    def __init__(self, in_size=2052, out_size=512, fc_sizes=[], dropout=0, sparsity=95):
        super(BitwiseMLP, self).__init__()
        self.activation = torch.tanh
        self.in_size = in_size
        self.out_size = out_size

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [out_size,]
        in_size = in_size
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        for i, out_size in enumerate(fc_sizes):
            self.linear_list.append(BitwiseLinear(in_size, out_size))
            in_size = out_size
            self.bn_list.append(nn.BatchNorm1d(out_size))
            if i < self.num_layers - 1:
                self.dropout_list.append(nn.Dropout(dropout))

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
        batch, channels, time = x.size()
        x = x.permute(0, 2, 1).contiguous().view(-1, channels)

        for i in range(self.num_layers):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.dropout_list[i](x)

        x = x.view(-1, time, self.out_size).permute(0, 2, 1)
        return x

    def noisy(self):
        '''
        Converts real network to noisy training network
        '''
        self.mode = 'noisy'
        self.activation = bitwise_activation
        for layer in self.linear_list:
            layer.noisy()

    def inference(self):
        '''
        Converts noisy training network to bitwise network
        '''
        self.mode = 'inference'
        self.activation = bitwise_activation
        for layer in self.linear_list:
            layer.inference()
        for bn in self.bn_list:
            sign_weight = torch.sign(bn.weight)
            bias = -sign_weight * bn.running_mean
            bias += bn.bias * bn.running_var / torch.abs(bn.weight)
            bn.bias = nn.Parameter(bias, requires_grad=False)
            bn.weight = nn.Parameter(sign_weight, requires_grad=False)
            bn.running_var = torch.ones_like(running_var)
            bn_running_mean = torch.zeros_like(running_mean)

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.mode != 'noisy':
            return

        for layer in self.linear_list:
            layer.update_beta(sparsity=self.sparsity)


def evaluate(model, dl, optimizer=None, loss=F.mse_loss, device=torch.device('cpu'),
    dtype=torch.float, train=True):
    running_loss = 0
    for batch in dl:
        if optimizer:
            optimizer.zero_grad()
        mix = batch['bmag'].to(device=device)
        target  = batch['ibm'].to(device=device)
        mix = mix.to(device=device)
        estimate = model(mix)
        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss.item() * mix.size(0)
        if optimizer:
            reconst_loss.backward()
            optimizer.step()
    return running_loss / len(dl.dataset)

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--train_noisy', '-tn',  action='store_true')
    parser.add_argument('--output_period', '-op', type=int, default=1)
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--l1_reg', '-l1r', type=float, default=0)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--model_file', '-mf', default='temp_model.model')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Make model and dataset
    train_dl, val_dl = make_binary_data(args.batchsize, toy=args.toy)
    model = BitwiseMLP(in_size=2052, out_size=513, fc_sizes=[2048, 2048],
        dropout=args.dropout, sparsity=args.sparsity)
    if args.train_noisy:
        print('Noisy Network Training')
        if args.load_file:
            model.load_state_dict(torch.load('../models/' + args.load_file))
        model.noisy()
    else:
        print('Real Network Training')
    model.to(device=device)
    print(model)

    # Initialize loss function
    loss = nn.BCEWithLogitsLoss()
    loss = loss.to(device=device)
    loss_metrics = LossMetrics()

    # Initialize optimizer
    vis = visdom.Visdom(port=5800)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = evaluate(model, train_dl, optimizer, loss=loss, device=device)

        if epoch % args.output_period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss = evaluate(model, val_dl, loss=loss, device=device)
            print('Val Cost: ', val_loss)
            loss_metrics.update(train_loss, val_loss,
                output_period=args.output_period)
            train_plot(vis, loss_metrics, eid='Ryley', win=['Loss', None])
            torch.save(model.state_dict(), '../models/' + args.model_file)
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.binary_data import *
from make_binary_data import *
from binary_layers import *
import argparse

class BinarizedNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc_sizes = [], biased=False, dropout=0):
        super(BinarizedNetwork, self).__init__()
        self.params = {}
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [output_size,]
        in_size = input_size
        self.linear_list = nn.ModuleList()
        self.batchnorm_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        self.activation = binarize
        self.conv1 = BinConv1d(input_size, input_size, 5, padding=2, groups=input_size)
        for i, out_size in enumerate(fc_sizes):
            self.linear_list.append(BinLinear(in_size, out_size, biased=biased))
            in_size = out_size
            if i < self.num_layers - 1:
                self.batchnorm_list.append(nn.BatchNorm1d(out_size))
                self.dropout_list.append(nn.Dropout(dropout))
        self.scaler = Scaler(output_size)

    def forward(self, x):
        '''
        * Input is a tensor of shape (N, F, T)
        * Output is a tensor of shape (N, F, T)
        '''
        h = self.activation(self.conv1(x))

        # Flatten (N, F, T) -> (NT, F)
        h = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        for i in range(self.num_layers):
            h = self.linear_list[i](h)
            if i < self.num_layers - 1:
                h = self.batchnorm_list[i](h)
                h = self.activation(h)
                h = self.dropout_list[i](h)
        h = self.scaler(h)
        # Unflatten (NT, F) -> (N, F, T)
        y = h.view(x.size(0), x.size(2), -1).permute(0, 2, 1)
        return y

def make_model(dropout=0, toy=False):
    if toy:
        model = BinarizedNetwork(2052, 513, fc_sizes=[1024], dropout=dropout)
        model_name = 'models/toy_bin_network.model'
    else:
        model = BinarizedNetwork(2052, 513, fc_sizes=[2048, 2048], dropout=dropout)
        model_name = 'models/bin_network.model'

    return model, model_name

def model_loss(model, batch, mse=False, device=torch.device('cpu')):
    bmag, ibm = batch['bmag'].cuda(device), batch['ibm'].cuda(device)
    premask = model(2*bmag-1)
    if mse:
        loss = F.mse_loss(premask, 2*ibm - 1)
    else:
        loss = F.binary_cross_entropy_with_logits(premask, ibm)
    return loss

def clip(state_dict, min, max):
    pass

def main():
    parser = argparse.ArgumentParser(description='binarized network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=0.95)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--output_period', '-op', type=int, default=8)
    parser.add_argument('--l1_reg', '-l1r', type=float, default=0)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dl, val_dl = make_dataset(args.batchsize, toy=args.toy)
    model, model_name = make_model(args.dropout, toy=args.toy)
    model.to(device)
    print(model)
    loss = nn.BCEWithLogitsLoss()
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.train()
        for count, batch in enumerate(train_dl):
            optimizer.zero_grad()
            cost = model_loss(model, batch)
            total_cost += cost.data
            cost.backward()
            optimizer.step()
            clip_params(model)
        avg_cost = total_cost / (count + 1)

        if epoch % args.output_period == 0:
            print('Epoch %d Training Cost: ' % epoch, avg_cost)
            total_cost = 0
            model.eval()
            for count, batch in enumerate(val_dl):
                cost = model_loss(model, batch)
                total_cost += cost.data
            avg_cost = total_cost / (count + 1)
            print('Validation Cost: ', avg_cost)
            torch.save(model.state_dict(), model_name)
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

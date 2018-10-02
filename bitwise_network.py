import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.two_source_mixture import *
from datasets.binary_data import *
from binary_layers import *
import argparse

class BitwiseNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc_sizes = [], dropout=0, sparsity=95):
        super(BitwiseNetwork, self).__init__()
        self.params = {}
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [output_size,]
        in_size = input_size
        self.linear_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        self.activation = torch.tanh
        for i, out_size in enumerate(fc_sizes):
            self.linear_list.append(BitwiseLinear(in_size, out_size))
            in_size = out_size
            if i < self.num_layers - 1:
                self.dropout_list.append(nn.Dropout(dropout))
        self.sparsity = sparsity

    def forward(self, x):
        '''
        * Input is a tensor of shape (N, F, T)
        * Output is a tensor of shape (N, F, T)
        '''
        # Flatten (N, F, T) -> (NT, F)
        h = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        for i in range(self.num_layers): 
            h = self.linear_list[i](h) 
            # if self.linear_list[i].mode == 'noisy':
            #    h /= (self.linear_list[i].input_size * (1 - self.sparsity/100.0))
            if i < self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout_list[i](h) 
        # Unflatten (NT, F) -> (N, F, T)
        y = h.view(x.size(0), x.size(2), -1).permute(0, 2, 1)
        return y

    def noisy(self):
        self.activation = bitwise_activation
        for layer in self.linear_list:
            layer.noisy()

    def update_betas(self):
        for layer in self.linear_list:
            if layer.mode == 'noisy':
                layer.update_beta(sparsity=self.sparsity)

def make_dataset(batchsize, seed=0, toy=False):
    np.random.seed(seed)

    train_dir = '/media/data/binary_audio/train'
    val_dir = '/media/data/binary_audio/val'
    if toy:
        train_dir = '/media/data/binary_audio/toy_train'
        val_dir = '/media/data/binary_audio/toy_val'

    trainset = BinaryDataset(train_dir)
    valset = BinaryDataset(val_dir)
    collate_fn = lambda x: collate_and_trim(x, axis=1)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)
    return train_dl, val_dl

def make_model(dropout=0, sparsity=0, train_noisy=False, toy=False):
    if toy:
        model = BitwiseNetwork(2052, 513, fc_sizes=[1024], dropout=dropout)
        real_model = 'models/toy_real_network.model'
        bitwise_model = 'models/toy_bitwise_network.model'
    else:
        model = BitwiseNetwork(2052, 513, fc_sizes=[2048, 2048],
            dropout=dropout, sparsity=sparsity)
        real_model = 'models/real_network.model'
        bitwise_model = 'models/bitwise_network.model'

    if not train_noisy:
        print('Real Network Training')
        model_name = real_model
    else:
        print('Noisy Network Training')
        model_name = bitwise_model
        model.load_state_dict(torch.load(real_model))
        model.noisy()

    return model, model_name

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=0.95)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--train_noisy', action='store_true')
    parser.add_argument('--output_period', '-op', type=int, default=8)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=95.0)
    parser.add_argument('--l1_reg', '-l1r', type=float, default=0)
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dl, val_dl = make_dataset(args.batchsize, toy=args.toy)
    model, model_name = make_model(args.dropout, args.sparsity, args.train_noisy, toy=args.toy)
    model.to(device)
    print(model)
    loss = nn.BCEWithLogitsLoss()
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    def model_loss(model, binary_batch, compute_bss=False):
        bmag, ibm = batch['bmag'].cuda(device), batch['ibm'].cuda(device)
        premask = model(2*bmag-1)
        cost = loss(premask, ibm)
        if args.l1_reg:
            for p in model.parameters():
                cost += args.l1_reg * torch.norm(p, 1)
        return cost

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        for count, batch in enumerate(train_dl):
            optimizer.zero_grad()
            cost = model_loss(model, batch)
            total_cost += cost.data
            cost.backward()
            optimizer.step()
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

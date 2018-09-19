import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import numpy as np
from bss_eval import *
from torch.utils.data import Dataset, DataLoader
from two_source_mixture import *
from binary_data import *
import argparse

class RealNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc_sizes = [], dropout=0, activation=torch.tanh):
        super(RealNetwork, self).__init__()
        self.params = {}
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [output_size,]
        in_size = input_size
        self.dropout_list = nn.ModuleList()
        for i, out_size in enumerate(fc_sizes):
            wname, bname = 'weight%d' % (i+1,), 'bias%d' % (i+1,)
            w = torch.empty(out_size, in_size)
            nn.init.xavier_uniform_(w)
            b = torch.zeros(out_size)
            in_size = out_size
            setattr(self, wname, nn.Parameter(w, requires_grad=True))
            setattr(self, bname, nn.Parameter(b, requires_grad=True))
            self.dropout_list.append(nn.Dropout(dropout))
        self.activation = activation

    def forward(self, x):
        '''
        * Input is a tensor of shape (N, F, T)
        * Output is a tensor of shape (N, F, T)
        '''
        # Flatten (N, F, T) -> (NT, F)
        h = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        for i in range(self.num_layers):
            wname, bname = 'weight%d' % (i+1,), 'bias%d' % (i+1,)
            modified_w = torch.tanh(getattr(self, wname))
            modified_b = torch.tanh(getattr(self, bname))
            h = F.linear(h, modified_w, modified_b)
            if i != self.num_layers:
                h = self.activation(h)
        # Unflatten (NT, F) -> (N, F, T)
        y = h.view(x.size(0), x.size(2), -1).permute(0, 2, 1)
        return y

def make_dataset(batchsize, seed=0):
    np.random.seed(seed)
    trainset = BinaryDataset('/media/data/binary_audio/train')
    valset = BinaryDataset('/media/data/binary_audio/val')
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)
    return train_dl, val_dl

def main():
    parser = argparse.ArgumentParser(description='real network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dl, val_dl = make_dataset(args.batchsize)
    model = RealNetwork(2052, 513, fc_sizes=[1024, 1024]).to(device)
    print(model)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def model_loss(model, binary_batch, compute_bss=False):
        bmag, ibm = batch['bmag'].cuda(device), batch['ibm'].cuda(device)
        premask = model(bmag)
        cost = loss(premask, ibm)
        return cost

    for epoch in range(args.epochs):
        total_cost = 0
        count = 0
        bss_metrics = BSSMetricsList()
        model.train()
        for count, batch in enumerate(train_dl):
            optimizer.zero_grad()
            cost = model_loss(model, batch)
            total_cost += cost.data
            cost.backward()
            optimizer.step()
        avg_cost = total_cost / (count + 1)

        if epoch % 8 == 0:
            print('Epoch %d Training Cost: ' % epoch, avg_cost)

            total_cost = 0
            bss_metrics = BSSMetricsList()
            model.eval()
            for count, batch in enumerate(val_dl):
                cost = model_loss(model, batch)
                total_cost += cost.data
            avg_cost = total_cost / (count + 1)
            print('Validation Cost: ', avg_cost)
            torch.save(model.state_dict(), 'models/real_network.model')

if __name__ == '__main__':
    main()

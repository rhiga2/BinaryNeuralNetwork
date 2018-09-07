import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from TwoSourceMixtureDataset import *
import argparse

class RealNetwork(nn.Module):
    def __init__(self, fft_size, fc_sizes = [], activation=F.relu):
        super(RealNetwork, self).__init__()
        input_size = fft_size
        self.linear_layers = nn.ModuleList()
        fc_sizes = fc_sizes + [fft_size,]
        for output_size in fc_sizes:
            self.linear_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size
        self.activation = F.relu

    def forward(self, x):
        '''
        * Input is a tensor of shape (N, F, T)
        * Output is a tensor of shape (N, F, T)
        '''
        # Flatten (N, F, T) -> (NT, F)
        h = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        for layer in self.linear_layers[:-1]:
            h = self.activation(layer(h))
        h = self.linear_layers[-1](h)
        # Unflatten (NT, F) -> (N, F, T)
        y = h.view(-1, x.size(2), x.size(1)).permute(0, 2, 1)
        return y

def make_dataset(batchsize):
    np.random.seed(0)
    speaker_path = '/media/data/timit-wav/train'
    targ_speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
                    'dr1/fdaw0', 'dr1/fjsp0', 'dr1/fsjk1', 'dr1/fvmh0']
    inter_speakers = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0',
                    'mwad0']
    train_speeches, val_speeches = get_speech_files(speaker_path, targ_speakers)
    train_inters, val_inters = get_speech_files(speaker_path, inter_speakers)
    trainset = TwoSourceSpectrogramDataset(train_speeches, train_inters)
    valset = TwoSourceSpectrogramDataset(val_speeches, val_inters)
    collate_fn = lambda x: collate_and_trim(x, dim=1)
    train_dl = DataLoader(trainset, batch_size=batchsize,
        shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(valset, batch_size=1, collate_fn=collate_fn)
    return train_dl, val_dl

def make_model(device):
    real_net = RealNetwork(513, fc_sizes=[1024, 1024]).to(device)
    return real_net

def main():
    parser = argparse.ArgumentParser(description='real network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--disable_cuda', type=bool, default=False)
    args = parser.parse_args()

    if not args.disable_cuda or torch.cuda.is_available():
        print('Using device 0')
        device = torch.device('cuda:0')
        dtype = torch.cuda.FloatTensor
    else:
        print('Using CPU')
        device = torch.device('cpu')
        dtype = torch.FloatTensor

    train_dl, val_dl = make_dataset(args.batchsize, device)
    real_net = make_model(device)
    print(real_net)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(real_net.parameters(), lr=1e-3)

    try:
        for epoch in range(args.epochs):
            total_cost = 0
            real_net.train()
            for count, batch in enumerate(train_dl):
                optimizer.zero_grad()
                output = real_net(batch['mixture'].to(device))
                ibm = ((batch['target'] - batch['interference']) >= 0).type(dtype) # ideal binary mask
                cost = loss(output, ibm)
                total_cost += cost.cpu().numpy()[0]
                cost.backward()
                optimizer.step()
            avg_cost = total_cost / (count + 1)
            print('Epoch %d Training Cost: ' % epoch, avg_cost, end=' ')

            total_cost = 0
            real_net.eval()
            for count, batch in enumerate(val_dl):
                output = real_net(batch['mixture'].to(device))
                ibm = ((batch['target'] - batch['interference']) >= 0).type(dtype) # ideal binary mask
                cost = loss(output, ibm)
                total_cost += cost.cpu().numpy()[0]
            avg_cost = total_cost / (count + 1)
            print('Validation Cost: ', avg_cost)

    finally:
        torch.save(real_net.state_dict(), 'models/real_network.model')

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from TwoSourceMixtureDataset import *
from spectrogram import *
import argparse

class RealNetwork(nn.Module):
    def __init__(self, fft_size, fc_sizes = [], activation=F.relu):
        input_size = fft_size
        self.linear_layers = nn.ModuleList()
        fc_sizes = fc_sizes + fft_size
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
        y = h.view(-1, x.size(0), x.size(1)).permute(0, 2, 1)
        return y

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
    else:
        print('Using CPU')
        device = torch.device('cpu')

    np.random.seed(0)
    targ_path = '/media/data/timit-wav/train'
    inter_path = targ_path
    targ_speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
                    'dr1/fdaw0', 'dr1/fjsp0', 'dr1/fsjk1', 'dr1/fvmh0']
    inter_speakers = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0',
                    'mwad0']
    trainset = TwoSourceMixtureDataset(train_speeches, train_noises)
    valset = TwoSourceMixtureDataset(val_speeches, val_noises)
    print('Train Length: ', len(trainset))
    print('Validation Length: ', len(valset))
    collate_fn = lambda x: collate_fn(x, 256)
    train_dl = DataLoader(trainset, batch_size=args.batchsize,
        shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(valset, batch_size=len(valset), collate_fn=collate_fn)

    make_spectrogram = MakeSpectrogram(fft_size=1024, hop=256).to(device)
    real_net = RealNetwork(513, fc_sizes=[1024, 1024]).to(device)
    print(real_net)
    loss = torch.nn.BCEWithLogitsLoss()
    optim = optim.Adam(real_net.parameters(), lr=1e-3)

    try:
        for epoch in range(args.epochs):
            total_cost = 0
            real_net.train()
            for count, batch in enumerate(train_dl):
                optim.zero_grad()
                mixture_spectrogram = make_spectrogram(batch['mixture'].to(device))
                target_spectrogram = make_spectrogram(batch['target'].to(device))
                interference_spectrogram = make_spectrogram(batch['interference'].to(device))
                output = real_net(mixture_spectrogram)
                ibm = (target_spectrogram - inter_spectrogram) > 0 # ideal binary mask
                cost = loss(output, ibm)
                total_cost += cost
                cost.backward()
                optim.step()
            avg_cost = cost / (count + 1)
            print('Epoch %d Training Cost: ' % epoch, avg_cost, sep=' ')

            total_cost = 0
            real_net.eval()
            for batch in val_dl:
                mixture_spectrogram = make_spectrogram(batch['mixture'].to(device))
                target_spectrogram = make_spectrogram(batch['target'].to(device))
                interference_spectrogram = make_spectrogram(batch['interference'].to(device))
                output = real_net(mixture_spectrogram)
                ibm = (target_spectrogram - interference_spectrogram) > 0
                cost = loss(output, ibm)
                total_cost += cost
            avg_cost = cost / (count + 1)
            print('Validation Cost: ', avg_cost)

    finally:
        torch.save(real_net.state_dict(), 'real_network.model')

if __name__ == '__main__':
    main()

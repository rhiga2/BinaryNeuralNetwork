import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from bss_eval import *
from torch.utils.data import Dataset, DataLoader
from TwoSourceMixtureDataset import *
from stft import *
import argparse

class RealNetwork(nn.Module):
    def __init__(self, input_size, fc_sizes = [], activation=F.tanh):
        super(RealNetwork, self).__init__()
        self.params = {}
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [input_size,]
        for i, output_size in enumerate(fc_sizes):
            wname, bname = 'weight%d' % (i+1,), 'bias%d' % (i+1,)
            w = torch.empty(output_size, input_size)
            nn.init.xavier_uniform_(w)
            b = torch.zeros(output_size)
            input_size = output_size
            setattr(self, wname, nn.Parameter(w, requires_grad=True))
            setattr(self, bname, nn.Parameter(b, requires_grad=True))
        self.activation = F.relu

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
        y = h.view(-1, x.size(2), x.size(1)).permute(0, 2, 1)
        return y

def make_dataset(batchsize, device=torch.device('cpu'), seed=0):
    np.random.seed(seed)
    speaker_path = '/media/data/timit-wav/train'
    targ_speakers = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
                    'dr1/fdaw0', 'dr1/fjsp0', 'dr1/fsjk1', 'dr1/fvmh0']
    inter_speakers = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0', 'dr1/mrws0',
                    'mwad0']
    train_speeches, val_speeches = get_speech_files(speaker_path, targ_speakers)
    train_inters, val_inters = get_speech_files(speaker_path, inter_speakers)
    trainset = TwoSourceMixtureDataset(train_speeches, train_inters,
        device=device)
    valset = TwoSourceMixtureDataset(val_speeches, val_inters,
        device=device)
    collate_fn = lambda x: collate_and_trim(x, dim=0, hop=256)
    train_dl = DataLoader(trainset, batch_size=batchsize,
        shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)
    return train_dl, val_dl

def make_model(device=torch.device('cpu')):
    real_net = RealNetwork(513, fc_sizes=[1024, 1024]).to(device)
    return real_net

def make_binary_mask(premask, device=torch.device('cpu'), dtype=torch.float):
    return torch.tensor(premask > 0, dtype=dtype, device=device)

def main():
    parser = argparse.ArgumentParser(description='real network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dl, val_dl = make_dataset(args.batchsize, device=device)
    stft = STFT(1024, 256).to(device)
    model = make_model(device)
    print(model)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def evaluate_model(model, batch):
        mix, targ, inter = batch['mixture'], batch['target'], batch['interference']
        mix_mag, mix_phase = stft.transform(mix)
        targ_mag, targ_phase = stft.transform(targ)
        inter_mag, inter_phase = stft.transform(inter)
        premask = model(mix_mag)
        ibm = make_binary_mask(targ_mag - inter_mag, device)
        cost = loss(premask, ibm)

        # calculate bss eval metrics
        mask = make_binary_mask(premask, device)
        mag_pred = mask*mix_mag
        sig_pred = stft.inverse(mask*mix_mag, mix_phase)
        sources = torch.stack([targ, inter], dim=2)
        metrics = bss_eval_batch(sig_pred, sources)

        return cost, metrics

    for epoch in range(args.epochs):
        total_cost = 0
        bss_metrics = BSSMetricsList()
        model.train()
        for count, batch in enumerate(train_dl):
            optimizer.zero_grad()
            cost, metrics = evaluate_model(model, batch)
            bss_metrics.extend(metrics)
            total_cost += cost.data
            cost.backward()
            optimizer.step()
        avg_cost = total_cost / (count + 1)

        if epoch % 8 == 0:
            print('Epoch %d Training Cost: ' % epoch, avg_cost, end=', ')
            sdr, sir, sar = bss_metrics.mean()
            print('SDR: %f, SIR: %f, SAR: %f' % (sdr, sir, sar))

            total_cost = 0
            bss_metrics = BSSMetricsList()
            model.eval()
            for count, batch in enumerate(val_dl):
                cost, metrics = evaluate_model(model, batch)
                total_cost += cost.data
                bss_metrics.extend(metrics)
            avg_cost = total_cost / (count + 1)
            print('Validation Cost: ', avg_cost, end = ', ')
            sdr, sir, sar = bss_metrics.mean()
            print('SDR: %f, SIR: %f, SAR: %f' % (sdr, sir, sar))
    torch.save(model.state_dict(), 'models/real_network.model')

if __name__ == '__main__':
    main()

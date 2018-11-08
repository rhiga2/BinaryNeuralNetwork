import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from datasets.quantized_data import *
from datasets.two_source_mixture import *
from loss_and_metrics.sepcosts import *
from loss_and_metrics.bss_eval import *
from dnn.binary_layers import *
import visdom
import argparse

def make_mixture_set(hop=256, toy=False):
    speaker_path = '/media/data/timit-wav/train'
    targets = ['dr1/fcjf0', 'dr1/fetb0', 'dr1/fsah0', 'dr1/fvfb0',
        'dr1/fdaw0', 'dr1/fjsp0'] #,'dr1/fsjk1', 'dr1/fvmh0',
        # 'dr1/fsma0', 'dr1/ftbr0']
    train_speeches, val_speeches = get_speech_files(speaker_path, targets, num_train=7)

    if toy:
        noise_path = '/media/data/Nonspeech'
        interferences = ['n81.wav', # chimes
                         'n97.wav', # eating chips
                         'n21.wav', # motorcycle
                         'n46.wav', # ocean
                         'n47.wav', # birds
                         'n55.wav', # cicadas?
                         'n59.wav', # jungle?
                         ]
        train_noises, val_noises = get_noise_files(noise_path, interferences)
        trainset = TwoSourceMixtureDataset(train_speeches, train_noises, hop=hop)
        valset = TwoSourceMixtureDataset(val_speeches, val_noises, hop=hop)
    else:
        interferences = ['dr1/mdpk0', 'dr1/mjwt0', 'dr1/mrai0']#, 'dr1/mrws0',
        #    'dr1/mwad0', 'dr1/mwar0']
        train_noises, val_noises = get_speech_files(speaker_path, interferences, num_train=7)
        trainset = TwoSourceMixtureDataset(train_speeches, train_noises, hop=hop)
        valset = TwoSourceMixtureDataset(val_speeches, val_noises, hop=hop)
    return trainset, valset

def make_data(batchsize, hop=256, toy=False):
    '''
    Make two mixture dataset
    '''
    trainset, valset = make_mixture_set(hop=hop, toy=toy)
    collate = lambda x: collate_and_trim(x, axis=0)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate)
    return train_dl, val_dl

class BitwiseNetwork(nn.Module):
    '''
    Adaptive transform network inspired by Minje Kim
    '''
    def __init__(self, kernel_size=1024, stride=256, in_channels=1,
        out_channels=256, combine_hidden=8, fc_sizes = [], dropout=0,
        sparsity=95, adapt=True, autoencode=False, groups=1):
        super(BitwiseNetwork, self).__init__()
        # Initialize adaptive front end
        self.kernel_size = kernel_size
        self.cutoff = kernel_size // 2 + 1
        self.transform_channels = 2*self.cutoff
        self.conv1 = BitwiseConv1d(in_channels, self.transform_channels,
            kernel_size, stride=stride, padding=kernel_size, groups=groups)
        self.in_scaler = nn.BatchNorm1d(self.transform_channels)
        self.autoencode = autoencode
        self.activation = torch.tanh

        # dense layers for denoising
        if not self.autoencode:
            self.combine1 = nn.Conv2d(2, combine_hidden, 1)
            self.combine2 = nn.Conv2d(combine_hidden, 1, 1)

            # Initialize linear layers
            self.num_layers = len(fc_sizes) + 1
            fc_sizes = fc_sizes + [self.cutoff,]
            in_size = self.cutoff
            self.linear_list = nn.ModuleList()
            self.scaler_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i, out_size in enumerate(fc_sizes):
                self.linear_list.append(BitwiseLinear(in_size, out_size))
                in_size = out_size
                self.scaler_list.append(Scaler(out_size))
                if i < self.num_layers - 1:
                    self.dropout_list.append(nn.Dropout(dropout))


        # Initialize inverse of front end transform
        self.conv1_transpose = BitwiseConvTranspose1d(self.transform_channels,
            out_channels, kernel_size, stride=stride, groups=groups)
        self.output_activation = nn.Softmax(dim=1)
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
        time = x.size(2)
        x = self.activation(self.in_scaler(self.conv1(x)))

        if not self.autoencode:
            # (batch, channels_per_group * transform_size, time)
            real_x = transformed_x[:, :self.cutoff, :]
            imag_x = transformed_x[:, self.cutoff:, :]
            spec_x = torch.stack([real_x, imag_x], dim=1)
            spec_x = F.relu(self.combine1(spec_x))
            spec_x = F.relu(self.combine2(spec_x)).squeeze(1)

            # Flatten (batch, bands, frames) -> (batch * frames, bands)
            h = spec_x.permute(0, 2, 1).contiguous().view(-1, spec_x.size(1))
            for i in range(self.num_layers):
                h = self.linear_list[i](h)
                if self.mode != 'inference':
                    h = self.scaler_list[i](h)
                if i < self.num_layers - 1:
                    h = self.activation(h)
                    h = self.dropout_list[i](h)
            h = (self.output_transform(h) + 1) / 2

            # Unflatten (batch * frames, bands) -> (batch, bands, frames)
            mask = h.view(spec_x.size(0), spec_x.size(2), -1).permute(0, 2, 1)
            mask = torch.cat([mask, mask], dim=1)
            transformed_x = transformed_x * mask

        x = self.conv1_transpose(x)[:, :, self.kernel_size:time+self.kernel_size]
        x = self.output_activation(x)
        return x

    def noisy(self):
        '''
        Converts real network to noisy training network
        '''
        self.mode = 'noisy'
        self.activation = bitwise_activation
        self.conv1.noisy()
        self.conv1_transpose.noisy()
        if not self.autoencode:
            for layer in self.linear_list:
                layer.noisy()

    def inference(self):
        '''
        Converts noisy training network to bitwise network
        '''
        self.mode = 'inference'
        self.activation = bitwise_activation
        if not self.autoencode:
            for layer in self.linear_list:
                layer.inference()

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        if self.mode != 'noisy':
            return

        self.conv1.update_beta(sparsity=self.sparsity)
        self.conv1_transpose.update_beta(sparsity=self.sparsity)
        if not self.autoencode:
            for layer in self.linear_list:
                layer.update_beta(sparsity=self.sparsity)

def train(model, dl, optimizer, loss=F.mse_loss, device=torch.device('cpu'), autoencode=False,
    quantizer=None, transform=None, dtype=torch.float):
    running_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        mix, target = batch['mixture'], batch['interference']
        if autoencode:
            mix = target
        if quantizer:
            mix = quantizer(mix).to(device=device, dtype=dtype)
            target = quantizer(target).to(device=device, dtype=torch.long)
        if transform:
            mix = transform(mix)
        else:
            mix = mix.unsqueeze(1)
        mix = mix.to(device=device)
        estimate = model(mix)
        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss.item() * mix.size(0)
        reconst_loss.backward()
        optimizer.step()
    return running_loss / len(dl.dataset)

def val(model, dl, loss=F.mse_loss, autoencode=False,
    quantizer=None, transform=None, device=torch.device('cpu'),
    dtype=torch.float):
    running_loss = 0
    bss_metrics = BSSMetricsList()
    for batch in dl:
        mix, target = batch['mixture'], batch['target']
        if autoencode:
            mix = target
        if quantizer:
            mix = quantizer(mix).to(device=device, dtype=dtype)
            target = quantizer(target).to(device=device, dtype=torch.long)
        if transform:
            mix = transform(mix)
        else:
            mix = mix.unsqueeze(1)
        mix = mix.to(device=device)
        estimate = model(mix)
        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss.item() * mix.size(0)
        estimate = torch.argmax(estimate, dim=1).to(torch.float)
        if quantizer:
            estimate = quantizer.inverse(estimate)
        estimate = estimate.to(device='cpu')
        sources = torch.stack([batch['target'], batch['interference']], dim=1)
        metrics = bss_eval_batch(estimate, sources)
        bss_metrics.extend(metrics)
    return running_loss / len(dl.dataset), bss_metrics

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--kernel', '-k', type=int, default=512)
    parser.add_argument('--stride', '-s', type=int, default=128)
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--no_adapt', '-no_adapt', action='store_true')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--train_noisy', '-tn',  action='store_true')
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--output_period', '-op', type=int, default=1)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--l1_reg', '-l1r', type=float, default=0)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--autoencode', action='store_true')
    parser.add_argument('--model_file', '-mf', default='temp_model.model')
    parser.add_argument('--num_bits', '-nb', type=int, default=8)
    parser.add_argument('--groups', '-g', type=int, default=1)
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Initialize quantizer and dequantizer
    delta = 2/(2**args.num_bits)
    quantizer = Quantizer(-1, delta, num_bits=args.num_bits, use_mu=True)
    # transform = Disperser(num_bits=args.num_bits, center=True)
    # transform = transform.to(device=device, dtype=dtype)
    transform = None

    # Make model and dataset
    train_dl, val_dl = make_data(args.batchsize, hop=args.stride, toy=args.toy)
    model = BitwiseNetwork(args.kernel, args.stride, fc_sizes=[2048, 2048],
        in_channels=args.num_bits if transform else 1, out_channels=2**args.num_bits,
        dropout=args.dropout, sparsity=args.sparsity, adapt=not args.no_adapt,
        autoencode=args.autoencode, groups=args.groups)
    if args.train_noisy:
        print('Noisy Network Training')
        if args.load_file:
            model.load_state_dict(torch.load('../models/' + args.load_file))
        model.noisy()
    else:
        print('Real Network Training')
    model.to(device=device, dtype=dtype)
    print(model)

    # Initialize loss function
    col = torch.arange(0, 2**args.num_bits).to(torch.float)
    col = 100*quantizer.inverse(col)
    dist_matrix = torch.abs(col.unsqueeze(1)-col)
    loss = DiscreteWasserstein(2**args.num_bits, mode='interger',
        default_dist=False, dist_matrix=dist_matrix)
    loss = loss.to(device=device, dtype=dtype)
    loss_metrics = LossMetrics()

    # Initialize optimizer
    vis = visdom.Visdom(port=5800)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = train(model, train_dl, optimizer, loss=loss, device=device,
            autoencode=args.autoencode, quantizer=quantizer, transform=transform,
            dtype=dtype)

        if epoch % args.output_period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, loss=loss, device=device,
                autoencode=args.autoencode, quantizer=quantizer,
                transform=transform, dtype=dtype)
            sdr, sir, sar = bss_metrics.mean()
            loss_metrics.update(train_loss, val_loss, sdr, sir, sar,
                output_period=args.output_period)
            train_plot(vis, loss_metrics, eid='Ryley', win=['Loss', 'BSS Eval'])
            print('Validation Cost: ', val_loss)
            print('Val SDR: ', sdr)
            print('Val SIR: ', sir)
            print('Val SAR: ', sar)
            torch.save(model.state_dict(), '../models/' + args.model_file)
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

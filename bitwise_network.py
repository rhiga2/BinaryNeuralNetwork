import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import numpy as np
from datasets.binary_data import *
from make_binary_data import *
from sepcosts import *
from binary_layers import *
from bss_eval import *
import visdom
import argparse

class BitwiseNetwork(nn.Module):
    '''
    Adaptive transform network inspired by Minje Kim
    '''
    def __init__(self, kernel_size=1024, stride=256, combine_hidden=8, fc_sizes = [], dropout=0, sparsity=95,
        adapt=True):
        super(BitwiseNetwork, self).__init__()
        # Initialize adaptive front end
        self.kernel_size = kernel_size
        self.cutoff = kernel_size // 2 + 1
        self.conv1 = nn.Conv1d(1, 2*self.cutoff, kernel_size, stride=stride,
            bias=False, padding=2*self.cutoff)
        fft = np.fft.fft(np.eye(kernel_size))
        real_fft = np.real(fft)
        im_fft = np.imag(fft)
        basis = torch.FloatTensor(
            np.concatenate([real_fft[:self.cutoff], im_fft[:self.cutoff]], axis=0)
        )
        self.conv1.weight = nn.Parameter(basis.unsqueeze(1), requires_grad=adapt)

        self.combine1 = nn.Conv2d(2, combine_hidden, 1)
        self.combine2 = nn.Conv2d(combine_hidden, 1, 0)

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [self.cutoff,]
        in_size = self.cutoff
        self.linear_list = nn.ModuleList()
        self.scaler_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        self.activation = torch.tanh
        for i, out_size in enumerate(fc_sizes):
            self.linear_list.append(BitwiseLinear(in_size, out_size))
            in_size = out_size
            self.scaler_list.append(Scaler(out_size))
            if i < self.num_layers - 1:
                self.dropout_list.append(nn.Dropout(dropout))
        self.output_transform = bitwise_activation

        # Initialize inverse of front end transform
        self.scale = kernel_size / stride
        inv_basis = torch.t(torch.pinverse(self.scale*basis))
        self.conv1_transpose = nn.ConvTranspose1d(2*self.cutoff, 1, kernel_size,
            stride=stride, bias=False)
        self.conv1_transpose.weight = nn.Parameter(inv_basis.unsqueeze(1),
            requires_grad=adapt)

        self.sparsity = sparsity
        self.mode = 'real'

    def forward(self, x):
        '''
        Bitwise neural network forward
        * Input is a tensor of shape (N, T)
        * Output is a tensor of shape (N, T)
            * N is the batch size
            * T is the sequence length
        '''
        # (N, T) -> (N, 1, T)
        transformed_x = x.unsqueeze(1)
        transformed_x = self.conv1(transformed_x)

        real_x = transformed_x[:, :self.cutoff, :]
        imag_x = transformed_x[:, self.cutoff:, :]
        spec_x = torch.stack([real_x, imag_x], dim=1)
        spec_x = F.relu(self.combine1(spec_x))
        spec_x = F.relu(self.combine2(spec_x)).squeeze(1)

        # Flatten (N, F, T') -> (NT', F)
        h = spec_x.permute(0, 2, 1).contiguous().view(-1, spec_x.size(1))
        for i in range(self.num_layers):
            h = self.linear_list[i](h)
            if self.mode != 'inference':
                h = self.scaler_list[i](h)
            if i < self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout_list[i](h)
        h = (self.output_transform(h) + 1) / 2

        # Unflatten (NT', F) -> (N, F, T')
        mask = h.view(spec_x.size(0), spec_x.size(2), -1).permute(0, 2, 1)
        mask = torch.cat([mask, mask], dim=1)
        reconstructed_x = transformed_x * mask
        y_hat = self.conv1_transpose(reconstructed_x)
        y_hat = y_hat.squeeze(1)
        return y_hat[:, 2*self.cutoff:x.size(1)+2*self.cutoff]

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

    def update_betas(self):
        '''
        Updates sparsity parameter beta
        '''
        for layer in self.linear_list:
            if layer.mode == 'noisy':
                layer.update_beta(sparsity=self.sparsity)

def make_data(batchsize, toy=False):
    '''
    Make two mixture dataset
    '''
    trainset, valset = make_mixture_set(toy=toy)
    collate = lambda x: collate_and_trim(x, axis=0)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate)
    return train_dl, val_dl

def make_model(dropout=0, sparsity=0, train_noisy=None, toy=False, adapt=True):
    '''
    Creates model dataset
    '''
    if toy:
        model = BitwiseNetwork(1024, 256, fc_sizes=[1024, 1024],
            dropout=dropout, adapt=adapt, sparsity=sparsity)
        real_model = 'models/toy_real_network.model'
        bitwise_model = 'models/toy_bitwise_network.model'
    else:
        model = BitwiseNetwork(1024, 256, fc_sizes=[2048, 2048],
            dropout=dropout, sparsity=sparsity, adapt=adapt)
        real_model = 'models/real_network.model'
        bitwise_model = 'models/bitwise_network.model'

    if not train_noisy:
        print('Real Network Training')
        model_name = real_model
    else:
        print('Noisy Network Training')
        model_name = bitwise_model
        model.load_state_dict(torch.load('models/' + train_noisy))
        model.noisy()

    return model, model_name

def biortho_loss(w, winv):
    '''
    Forces the back end transform to be a biorthogonal projection of the front
    end transform
    '''
    return torch.mse_loss(torch.mm(winv, w), torch.eye(w.size(1)))

@DeprecationWarning
def model_loss(model, batch, biortho_reg=0, loss=F.mse_loss, device=torch.device('cpu')):
    mix = batch['mixture'].cuda(device)
    targ = batch['target'].cuda(device)
    interference = batch['interference'].cuda(device)
    estimate = model(mix)
    reconstruction_loss = loss(estimate, targ, interference)
    biortho_loss = 0
    if biortho_reg != 0:
        w = model.conv1.weight.squeeze(1)
        winv = model.scale * torch.t(model.conv2.weight.squeeze(1))
        inv_loss = biortho_reg * biortho_loss(winv, w)
    return reconstruction_loss + biortho_loss

def get_data_from_batch(batch, device=torch.device('cpu')):
    mix = batch['mixture'].cuda(device)
    target = batch['target'].cuda(device)
    inter = batch['interference'].cuda(device)
    return mix, target, inter

def train(model, dl, optimizer, loss=F.mse_loss, device=torch.device('cpu')):
    running_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        mix, target, inter = get_data_from_batch(batch, device)
        estimates = model(mix)
        reconst_loss = loss(estimates, target)
        running_loss += reconst_loss.item() * mix.size(0)
        reconst_loss.backward()
        optimizer.step()
    return running_loss / len(dl)

def val(model, dl, loss=F.mse_loss, device=torch.device('cpu')):
    running_loss = 0
    bss_metrics = BSSMetricsList()
    for batch in dl:
        mix, target, inter = get_data_from_batch(batch, device)
        estimates = model(mix)
        reconst_loss = loss(estimates, target)
        running_loss += reconst_loss.item() * mix.size(0)
        sources = torch.stack([target, inter], dim=1)
        metrics = bss_eval_batch(estimates, sources)
        bss_metrics.extend(metrics)
    return running_loss / len(dl), bss_metrics

class LossMetrics(nn.Module):
    '''
    Data struct that keeps track of all losses and metrics during the training process
    '''
    def __init__(self):
        self.time = []
        self.train_loss = []
        self.val_loss = []
        self.sdrs = []
        self.sirs = []
        self.sars = []

    def update(self, train_loss, val_loss, sdr, sir, sar, output_period=1):
        if self.time:
            self.time.append(self.time[-1] + output_period)
        else:
            self.time = [0]
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.sdrs.append(sdr)
        self.sirs.append(sir)
        self.sars.append(sar)

<<<<<<< HEAD
def train_plot(vis, loss_metrics, eid=None, train_win='Loss Window'):
    '''
    Plots loss and metrics during the training process
    '''
||||||| merged common ancestors
def train_plot(vis, loss_metrics, eid=None, train_win='Loss Window'):
=======
def train_plot(vis, loss_metrics, eid=None, win=[None, None]):
>>>>>>> 1c6386f47bd2a5db9706d9d7be043c1f7c94ffcf
    # Loss plots
    data1 = [
        dict(
            x=loss_metrics.time, y=loss_metrics.train_loss, name='Training Loss',
            hoverinfo='y', line=dict(width=1), mode='lines', type='scatter'),
        dict(
            x=loss_metrics.time, y=loss_metrics.val_loss, name='Validation Loss',
            hoverinfo='y', line=dict( width=1), mode='lines', type='scatter')
    ]
    layout1 = dict(
        showlegend=True,
        legend=dict(orientation='h', y=1.1, bgcolor='rgba(0,0,0,0)'),
        margin=dict(r=30, b=40, l=50, t=50),
        font=dict(family='Bell Gothic Std'),
        xaxis=dict(autorange=True, title='Training Epochs'),
        yaxis=dict(autorange=True, title='Loss'),
        title=win[0]
    )
    vis._send(dict(data=data1, layout=layout1, win=win[0], eid=eid))

    # BSS_EVAL plots
    data2 = [
        # SDR
        dict(
            x=loss_metrics.time, y=loss_metrics.sdrs, name='SDR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
        # SIR
        dict(
            x=loss_metrics.time, y=loss_metrics.sirs, name='SIR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
        # SAR
        dict(
            x=loss_metrics.time, y=loss_metrics.sars, name='SAR',
            hoverinfo='name+y+lines', line=dict( width=1), mode='lines', type='scatter'),
    ]
    layout2 = dict(
        showlegend=True,
        legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
        margin=dict(r=30, b=40, l=50, t=50),
        font=dict(family='Bell Gothic Std'),
        xaxis=dict(autorange=True, title='Training samples'),
        yaxis=dict(autorange=True, title='dB'),
        yaxis2=dict(autorange=True, title='STOI', overlaying='y', side='right'),
        title=win[1]
    )
    vis._send(dict(data=data2, layout=layout2, win=win[1], eid=eid))

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--stride')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--no_adapt', '-no_adapt', action='store_true')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--train_noisy', '-tn',  default=None)
    parser.add_argument('--output_period', '-op', type=int, default=1)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=95.0)
    parser.add_argument('--l1_reg', '-l1r', type=float, default=0)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--model_file', '-mf', default='temp_model.model')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    train_dl, val_dl = make_data(args.batchsize, toy=args.toy)
    model, model_name = make_model(args.dropout, args.sparsity, train_noisy=args.train_noisy,
        toy=args.toy, adapt=not args.no_adapt)
    vis = visdom.Visdom(port=5800)
    loss_metrics = LossMetrics()
    model.to(device)
    loss = SignalDistortionRatio()
    print(model)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = train(model, train_dl, optimizer, loss=loss, device=device)

        if epoch % args.output_period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, loss=loss, device=device)
            sdr, sir, sar = bss_metrics.mean()
            loss_metrics.update(train_loss, val_loss, sdr, sir, sar,
                output_period=args.output_period)
            train_plot(vis, loss_metrics, eid='Ryley', win=['Loss', 'BSS Eval'])
            print('Validation Cost: ', val_loss)
            print('Val SDR: ', sdr)
            print('Val SIR: ', sir)
            print('Val SAR: ', sar)
            torch.save(model.state_dict(), 'models/' + args.model_file)
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

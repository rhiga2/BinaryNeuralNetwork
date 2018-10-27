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
import pdb

class BitwiseNetwork(nn.Module):
    '''
    Adaptive transform network inspired by Minje Kim
    '''
    def __init__(self, kernel_size=1024, stride=256, in_channels=1,
        combine_hidden=8, fc_sizes = [], dropout=0, sparsity=95,
        adapt=True, autoencode=False):
        super(BitwiseNetwork, self).__init__()
        # Initialize adaptive front end
        self.kernel_size = kernel_size
        self.cutoff = kernel_size // 2 + 1
        self.in_channels = in_channels
        self.transform_channels = 2*self.cutoff*in_channels
        self.conv1 = BitwiseConv1d(in_channels, self.transform_channels,
            kernel_size, stride=stride, biased=False,
            padding=kernel_size, groups=in_channels)
        fft = np.fft.fft(np.eye(kernel_size))
        real_fft = np.real(fft)[:self.cutoff]
        im_fft = np.real(fft)[:self.cutoff]
        fft_list = []
        for _ in range(in_channels):
            fft_list.append(real_fft)
            fft_list.append(im_fft)
        basis = torch.FloatTensor(np.concatenate(fft_list, axis=0))
        print(basis.shape)
        self.conv1.weight = nn.Parameter(basis.unsqueeze(1), requires_grad=adapt)
        self.in_scaler = ConvScaler1d(self.transform_channels)
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
            self.output_transform = bitwise_activation

        # Initialize inverse of front end transform
        self.scale = kernel_size / stride
        inv_basis = torch.t(torch.pinverse(self.scale*basis))
        self.conv1_transpose = BitwiseConvTranspose1d(
            self.transform_channels,
            in_channels, kernel_size, stride=stride, biased=False,
            groups=in_channels)
        self.conv1_transpose.weight = nn.Parameter(inv_basis.unsqueeze(1),
            requires_grad=adapt)

        self.sparsity = sparsity
        self.mode = 'real'

    def forward(self, x):
        '''
        Bitwise neural network forward
        * Input is a tensor of shape (batch, channels, time)
        * Output is a tensor of shape (batch, channels, time)
            * batch is the batch size
            * time is the sequence length
            * channels is the number of input channels = num bits in qad
        '''
        # (batch, channels, time)
        time = x.size(2)
        transformed_x = self.conv1(x)
        transformed_x = self.in_scaler(transformed_x)
        transformed_x = self.activation(transformed_x)

        if not self.autoencode:
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

        y_hat = self.activation(self.conv1_transpose(transformed_x))
        return y_hat[:, :, self.kernel_size:time+self.kernel_size]

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
        if self.autoencode:
            for layer in self.linear_list:
                layer.update_beta(sparsity=self.sparsity)

def make_data(batchsize, hop=256, toy=False, num_bits=8):
    '''
    Make two mixture dataset
    '''
    trainset, valset = make_mixture_set(hop=hop, toy=toy, num_bits=num_bits)
    collate = lambda x: collate_and_trim(x, axis=0)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate)
    return train_dl, val_dl

def biortho_loss(w, winv):
    '''
    Forces the back end transform to be a biorthogonal projection of the front
    end transform
    '''
    return torch.mse_loss(torch.mm(winv, w), torch.eye(w.size(1)))

def get_data_from_batch(batch, device=torch.device('cpu')):
    mix = batch['mixture'].to(device)
    target = batch['target'].to(device)
    inter = batch['interference'].to(device)
    return mix, target, inter

def train(model, dl, optimizer, loss=F.mse_loss, device=torch.device('cpu'), autoencode=False,
    quantizer=None, dequantizer=None):
    running_loss = 0
    for batch in dl:
        optimizer.zero_grad()
        mix, target, inter = get_data_from_batch(batch, device)
        if autoencode:
            mix = target
        if quantizer:
            mix = quantizer(mix)
        estimate = model(mix)
        if dequantizer:
            estimate = dequantizer(estimate)
        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss.item() * mix.size(0)
        reconst_loss.backward()
        optimizer.step()
    return running_loss / len(dl)

def val(model, dl, loss=F.mse_loss, device=torch.device('cpu'), autoencode=False,
    quantizer=None, dequantizer=None):
    running_loss = 0
    bss_metrics = BSSMetricsList()
    for batch in dl:
        mix, target, inter = get_data_from_batch(batch, device)
        if autoencode:
            mix = target
        if quantizer:
            mix = quantizer(mix, -2, 4/2**4)
        estimate = model(mix.unsqueeze(1)).squeeze(1)
        if dequantizer:
            estimate = dequantizer(estimate)
        reconst_loss = loss(estimate, target)
        running_loss += reconst_loss.item() * mix.size(0)
        sources = torch.stack([target, inter], dim=1)
        metrics = bss_eval_batch(estimates, sources)
        bss_metrics.extend(metrics)
    return running_loss / len(dl), bss_metrics

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--kernel', '-k', type=int, default=1024)
    parser.add_argument('--stride', '-s', type=int, default=128)
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--no_adapt', '-no_adapt', action='store_true')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--train_noisy', '-tn',  default=None)
    parser.add_argument('--output_period', '-op', type=int, default=1)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--l1_reg', '-l1r', type=float, default=0)
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--autoencode', action='store_true')
    parser.add_argument('--model_file', '-mf', default='temp_model.model')
    parser.add_argument('--num_bits', '-nb', type=int, default=8)
    args = parser.parse_args()

    # Initialize device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Make model and dataset
    train_dl, val_dl = make_data(args.batchsize, hop=args.stride, toy=args.toy,
        num_bits=args.num_bits)
    model = BitwiseNetwork(args.kernel, args.stride, fc_sizes=[2048, 2048],
        in_channels=args.num_bits, dropout=args.dropout,
        sparsity=args.sparsity, adapt=not args.no_adapt,
        autoencode=args.autoencode)
    if args.train_noisy:
        print('Noisy Network Training')
        model.load_state_dict(torch.load('models/' + args.train_noisy))
        model.noisy()
    else:
        print('Real Network Training')
    model.to(device)
    print(model)

    # Initialize loss function
    loss = SignalDistortionRatio()
    loss_metrics = LossMetrics()

    # Initialize quantizer and dequantizer
    delta = 4/(2**args.num_bits)
    quantizer = QuantizeDisperser(-2, delta, args.num_bits, device=device, dtype=torch.float32)
    dequantizer = DequantizeAccumulator(-2, delta, args.num_bits, device=device)

    # Initialize optimizer
    vis = visdom.Visdom(port=5800)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_loss = train(model, train_dl, optimizer, loss=loss, device=device,
            autoencode=args.autoencode, quantizer=quantizer, dequantizer=dequantizer)

        if epoch % args.output_period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss)
            model.eval()
            val_loss, bss_metrics = val(model, val_dl, loss=loss, device=device,
                autoencode=args.autoencode, quantizer=quantizer, dequantizer=dequantizer)
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

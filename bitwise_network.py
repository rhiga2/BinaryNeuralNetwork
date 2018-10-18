import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import numpy as np
from datasets.binary_data import *
from make_binary_data import *
from binary_layers import *
import argparse

class BitwiseNetwork(nn.Module):
    def __init__(self, kernel_size=1024, stride=256, fc_sizes = [], dropout=0, sparsity=95,
        adapt=True):
        super(BitwiseNetwork, self).__init__()

        # Initialize adaptive front end
        self.cutoff = kernel_size // 2 + 1
        self.conv1 = nn.Conv1d(1, 2*self.cutoff, kernel_size, stride=stride,
            bias=False)
        fft = np.fft.fft(np.eye(kernel_size))
        real_fft = np.real(fft)
        im_fft = np.imag(fft)
        basis = torch.FloatTensor(
            np.concatenate([real_fft[:self.cutoff], im_fft[:self.cutoff]], axis=0)
        )
        self.conv1.weight = nn.Parameter(basis.unsqueeze(1), requires_grad=adapt)

        # Initialize linear layers
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [kernel_size,]
        in_size = kernel_size
        self.linear_list = nn.ModuleList()
        self.scaler_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        self.activation = torch.tanh
        for i, out_size in enumerate(fc_sizes):
            self.linear_list.append(BitwiseLinear(in_size, out_size))
            in_size = out_size
            if i < self.num_layers - 1:
                self.scaler_list.append(Scaler(out_size))
                self.dropout_list.append(nn.Dropout(dropout))
        self.output_transform = self.activation

        # Initialize inverse of front end transform
        inv_basis = torch.FloatTensor(np.linalg.pinv(4*basis).T)
        self.conv1_transpose = nn.ConvTranspose1d(kernel_size, 1, kernel_size,
            stride=stride, bias=False)
        self.conv1_transpose.weight = nn.Parameter(inv_basis.unsqueeze(1), requires_grad=adapt)

        self.sparsity = sparsity
        self.mode = 'real'

    def forward(self, x):
        '''
        * Input is a tensor of shape (N, T)
        * Output is a tensor of shape (N, T)
        '''
        # (N, T) -> (N, 1, T)
        x = x.unsqueeze(1)
        transformed_x = self.conv1(x)
        real_part = transformed_x[:, :513, :]
        imag_part = transformed_x[:, 513:, :]
        mag = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        # Flatten (N, F, T') -> (NT', F)
        h = mag.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        for i in range(self.num_layers):
            h = self.linear_list[i](h)
            if i < self.num_layers - 1:
                if self.mode != 'inference':
                    h = self.scaler_list[i](h)
                h = self.activation(h)
                h = self.dropout_list[i](h)
        h = (self.output_transform(h) + 1) / 2

        # Unflatten (NT, F) -> (N, F, T)
        mask = h.view(x.size(0), x.size(2), -1).permute(0, 2, 1)
        mag = mag * mask
        reconstructed_x = torch.cat([mag*torch.cos(phase), mag*torch.sin(phase)], dim=1)
        y_hat = self.conv1_transpose(reconstructed_x)
        return y_hat.squeeze(1)

    def noisy(self):
        self.mode = 'noisy'
        self.activation = bitwise_activation
        for layer in self.linear_list:
            layer.noisy()

    def inference(self):
        self.mode = 'inference'
        self.activation = bitwise_activation
        for layer in self.linear_list:
            layer.inference()

    def update_betas(self):
        for layer in self.linear_list:
            if layer.mode == 'noisy':
                layer.update_beta(sparsity=self.sparsity)

def make_dataset(batchsize, toy=False):
    trainset, valset = make_mixture_set(toy=toy)
    collate = lambda x: collate_and_trim(x, dim=0)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True,
        collate_fn=collate_and_trim)
    val_dl = DataLoader(valset, batch_size=batchsize)

def make_model(dropout=0, sparsity=0, train_noisy=False, toy=False, adapt=True):
    if toy:
        model = BitwiseNetwork(2052, 513, fc_sizes=[1024, 1024],
            dropout=dropout, adapt=adapt, sparsity=sparsity)
        real_model = 'models/toy_real_network.model'
        bitwise_model = 'models/toy_bitwise_network.model'
    else:
        model = BitwiseNetwork(2052, 513, fc_sizes=[2048, 2048],
            dropout=dropout, sparsity=sparsity, adapt=adapt)
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

def model_loss(model, batch, device=torch.device('cpu')):
    mix, targ = batch['mixture'].cuda(device), batch['target'].cuda(device)
    estimate = model(mix)
    loss = F.mse_loss(estimate, targ)
    return loss

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--no_adapt', '-no_adapt', action='store_false')
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
    model, model_name = make_model(args.dropout, args.sparsity, args.train_noisy,
        toy=args.toy, adapt=not args.no_adapt)
    model.to(device)
    print(model)
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        for count, batch in enumerate(train_dl):
            optimizer.zero_grad()
            cost = model_loss(model, batch, device)
            total_cost += cost.data
            cost.backward()
            optimizer.step()
        avg_cost = total_cost / (count + 1)

        if epoch % args.output_period == 0:
            print('Epoch %d Training Cost: ' % epoch, avg_cost)
            total_cost = 0
            model.eval()
            for count, batch in enumerate(val_dl):
                cost = model_loss(model, batch, device)
                total_cost += cost.data
            avg_cost = total_cost / (count + 1)
            print('Validation Cost: ', avg_cost)
            torch.save(model.state_dict(), model_name)
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
import glob
import numpy as np
from bss_eval import *
from torch.utils.data import Dataset, DataLoader
from two_source_mixture import *
from binary_data import *
import argparse

class BinarizePreactivations(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return grad_output * (1 - torch.tanh(x)**2)

class BinarizeParams(Function):
    @staticmethod
    def forward(ctx, x, beta):
        return torch.tensor((x > beta) - (x < beta), dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

binarize_preactivations = BinarizePreactivations.apply
binarize_params = BinarizeParams.apply

class BitwiseNetwork(nn.Module):
    def __init__(self, input_size, output_size, fc_sizes = [], dropout=0):
        super(BitwiseNetwork, self).__init__()
        self.params = {}
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [output_size,]
        in_size = input_size
        self.dropout_list = nn.ModuleList()
        self.betas = []
        self.bin_mode = False
        for i, out_size in enumerate(fc_sizes):
            wname = 'weight%d' % (i+1,)
            bname = 'bias%d' % (i+1,)
            w = torch.empty(out_size, in_size)
            nn.init.xavier_uniform_(w)
            b = torch.zeros(out_size)
            in_size = out_size
            setattr(self, wname, nn.Parameter(w, requires_grad=True))
            setattr(self, bname, nn.Parameter(b, requires_grad=True))
            self.dropout_list.append(nn.Dropout(dropout))
            self.betas.append(0)

    def forward(self, x):
        '''
        * Input is a tensor of shape (N, F, T)
        * Output is a tensor of shape (N, F, T)
        '''
        # Flatten (N, F, T) -> (NT, F)
        h = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        for i in range(self.num_layers):
            wname = 'weight%d' % (i+1,)
            bname = 'bias%d' % (i+1,)
            weight = getattr(self, wname)
            bias = getattr(self, bname)

            if self.bin_mode:
                modified_w = binarize_params(weight, self.betas[i])
                modified_b = binarize_params(bias, self.betas[i])
            else:
                modified_w = torch.tanh(weight)
                modified_b = torch.tanh(bias)

            h = F.linear(h, modified_w, modified_b)
            if i != self.num_layers:
                if self.bin_mode:
                    h = torch.tanh(h)
                else:
                    h = binarize_preactivations(h)
        # Unflatten (NT, F) -> (N, F, T)
        y = h.view(x.size(0), x.size(2), -1).permute(0, 2, 1)
        return y

    def binarize(self):
        self.bin_mode = True
        for name, param in self.state_dict().items():
            setattr(self, name, nn.Parameter(torch.tanh(param), requires_grad=True))

    def update_betas(self, sparsity=0.95):
        self.betas = []
        for i in range(self.num_layers):
            wname = 'weight%d' % (i+1,)
            bname = 'bias%d' % (i+1,)
            weight = getattr(self, wname).data.cpu().numpy()
            bias = getattr(self, bname).data.cpu().numpy()
            layer_params = np.abs(np.concatenate((weight, np.expand_dims(bias, axis=1)), axis=1))
            self.betas.append(np.percentile(layer_params, sparsity))

def make_dataset(batchsize, seed=0):
    np.random.seed(seed)
    trainset = BinaryDataset('/media/data/binary_audio/train')
    valset = BinaryDataset('/media/data/binary_audio/val')
    collate_fn = lambda x: collate_and_trim(x, axis=1)
    train_dl = DataLoader(trainset, batch_size=batchsize, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(valset, batch_size=batchsize, collate_fn=collate_fn)
    return train_dl, val_dl

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=64,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--skip_real', action='store_true')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dl, val_dl = make_dataset(args.batchsize)
    model = BitwiseNetwork(2052, 513, fc_sizes=[1024, 1024], dropout=args.dropout).to(device)
    print(model)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def model_loss(model, binary_batch, compute_bss=False):
        bmag, ibm = batch['bmag'].cuda(device), batch['ibm'].cuda(device)
        premask = model(2*bmag-1)
        cost = loss(premask, ibm)
        return cost

    if not args.skip_real:
        print('Real Network Training')
        for epoch in range(args.epochs):
            total_cost = 0
            bss_metrics = BSSMetricsList()
            model.train()
            for count, batch in enumerate(train_dl):
                optimizer.zero_grad()
                cost = model_loss(model, batch)
                total_cost += cost.data
                cost.backward()
                optimizer.step()
            avg_cost = total_cost / (count + 1)

            if epoch % 4 == 0:
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
    else:
        model.load_state_dict(torch.load('models/real_network.model'))
        model.to(device)

    print('Noisy Training')
    model.binarize()
    for epoch in range(args.epochs):
        model.train()
        model.update_betas()
        total_cost = 0
        bss_metrics = BSSMetricsList()
        for count, batch in enumerate(train_dl):
            optimizer.zero_grad()
            cost = model_loss(model, batch)
            total_cost += cost.data
            cost.backward()
            optimizer.step()
        avg_cost = total_cost / (count + 1)

        if epoch % 4 == 0:
            print('Epoch %d Noisy Training Cost: ' % epoch, avg_cost)

            total_cost = 0
            bss_metrics = BSSMetricsList()
            model.eval()
            for count, batch in enumerate(val_dl):
                cost = model_loss(model, batch)
                total_cost += cost.data
            avg_cost = total_cost / (count + 1)
            print('Noisy Validation Cost: ', avg_cost)
            torch.save(model.state_dict(), 'models/bitwise_network.model')

if __name__ == '__main__':
    main()

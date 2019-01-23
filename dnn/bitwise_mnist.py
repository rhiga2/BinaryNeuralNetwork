import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from dnn.bitwise_ss import BitwiseMLP
from dnn.binary_layers import *
from datasets.quantized_data import *
import visdom
import argparse

def evaluate(model, dl, optimizer=None, loss=F.mse_loss, device=torch.device('cpu'),
    dtype=torch.float, train=True):
    running_loss = 0
    running_accuracy = 0
    for batch_idx, (data, target) in enumerate(dl):
        if optimizer:
            optimizer.zero_grad()
        data = data.to(device=device)
        target  = target.to(device=device)
        estimate = model(data)
        cost = loss(estimate, target)
        correct = torch.argmax(estimate, dim=1) == target
        accuracy = torch.mean(correct.float())
        running_accuracy += accuracy.item() * data.size(0)
        running_loss += cost.item() * data.size(0)
        if optimizer:
            cost.backward()
            optimizer.step()
    return running_accuracy / len(dl.dataset), running_loss / len(dl.dataset)

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=32,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0.2)
    parser.add_argument('--period', '-p', type=int, default=1)
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--in_bin', '-ib', default='identity')
    parser.add_argument('--weight_bin', '-wb', default='identity')
    parser.add_argument('--adaptive_scaling', '-as', action='store_true')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Make model and dataset
    flatten = lambda x : x.view(-1)
    train_data = datasets.MNIST('/media/data/MNIST', train=True, transform=transforms.Compose([
                           transforms.ToTensor(), flatten]), download=True)
    val_data = datasets.MNIST('/media/data/MNIST', train=False, transform=transforms.Compose([
                           transforms.ToTensor(), flatten]), download=True)
    train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batchsize, shuffle=False)

    in_bin = binary_layers.pick_activation(args.in_bin)
    weight_bin = binary_layers.pick_activation(args.weight_bin)
    model = BitwiseMLP(in_size=784, out_size=10, fc_sizes=[2048, 2048, 2048],
        activation=nn.ReLU(inplace=True), dropout=args.dropout,
        sparsity=args.sparsity, use_gate=args.use_gate,
        adaptive_scaling=args.adaptive_scaling)
    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))

    model.to(device=device)
    print(model)

    # Initialize loss function
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device=device)

    # Initialize optimizer
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_accuracy, train_loss = evaluate(model, train_dl, optimizer, loss=loss, device=device)

        if epoch % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss, train_accuracy)
            model.eval()
            val_accuracy, val_loss = evaluate(model, val_dl, loss=loss, device=device)
            print('Val Cost: ', val_loss, val_accuracy)
            torch.save(model.state_dict(), '../models/' + args.exp + '.model')
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

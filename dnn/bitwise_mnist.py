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
import dnn.bitwise_mlp as bitwise_mlp
import dnn.binary_layers as binary_layers
import loss_and_metrics.image_classification as image_classification
import visdom
import argparse

def forward(model, dl, optimizer=None, loss=F.mse_loss, device=torch.device('cpu'),
    dtype=torch.float, clip_weights=False):
    running_loss = 0
    running_accuracy = 0
    if optimizer:
        optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(dl):
        data = data.to(device=device)
        target  = target.to(device=device)
        estimate = model(data)
        cost = loss(estimate, target)
        if optimizer:
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            if clip_weights:
                model.clip_weights()
        correct = torch.argmax(estimate, dim=1) == target
        accuracy = torch.mean(correct.float())
        running_accuracy += accuracy.item() * data.size(0)
        running_loss += cost.item() * data.size(0)
    return running_accuracy / len(dl.dataset), running_loss / len(dl.dataset)

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=32,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=0.9)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-dropout', type=float, default=0)
    parser.add_argument('--period', '-p', type=int, default=1)
    parser.add_argument('--decay_period', '-dp', type=int, default=10)
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--binactiv', '-ba', default='identity')
    parser.add_argument('--adaptive_scaling', '-as', action='store_true')
    parser.add_argument('--bn_momentum', '-bnm', type=float, default=0.1)
    parser.add_argument('--clip_weights', '-cw', action='store_true')
    args = parser.parse_args()
    print(args)

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Make model and dataset
    vis = visdom.Visdom(port=5801)
    flatten = lambda x : x.view(-1)
    trans = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0,), (1.0,)),
            flatten])
    data = datasets.MNIST('/media/data/MNIST', train=True,
        transform=trans, download=True)
    val_size = int(0.1*len(data))
    train_size = len(data) - val_size
    train_data, val_data = torch.utils.data.random_split(data, (train_size, val_size))
    train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batchsize, shuffle=False)

    binactiv = binary_layers.pick_activation(args.binactiv)
    model = bitwise_mlp.BitwiseMLP(784, 10, fc_sizes=[2048, 2048, 2048],
        activation=F.relu, dropout=args.dropout,
        sparsity=args.sparsity, use_gate=args.use_gate,
        adaptive_scaling=args.adaptive_scaling, binactiv=binactiv,
        bn_momentum=args.bn_momentum)
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
    loss_metrics = image_classification.LossMetrics()

    max_accuracy = 0
    for epoch in range(args.epochs):
        total_cost = 0
        model.update_betas()
        model.train()
        train_accuracy, train_loss = forward(model, train_dl, optimizer,
            loss=loss, device=device, clip_weights=args.clip_weights)

        if epoch % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss, train_accuracy)
            model.eval()
            val_accuracy, val_loss = forward(model, val_dl, loss=loss, device=device)
            print('Val Cost: ', val_loss, val_accuracy)
            loss_metrics.update(train_loss, train_accuracy, val_loss,
                val_accuracy, period=args.period)
            image_classification.train_plot(vis, loss_metrics, eid='Ryley',
                win=['{} Loss'.format(args.exp), '{} Accuracy'.format(args.exp)])
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                torch.save(model.state_dict(), '../models/' + args.exp + '.model')

        if (epoch+1) % args.decay_period == 0 and args.lr_decay != 1:
            lr *= args.lr_decay
            print(lr)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

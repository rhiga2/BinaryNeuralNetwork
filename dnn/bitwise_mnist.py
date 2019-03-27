import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import math
import numpy as np
import dnn.bitwise_mlp as bitwise_mlp
import dnn.binary_layers as binary_layers
from dnn.solvers import ImageRecognitionSolver
import loss_and_metrics.image_classification as image_classification
import visdom
import argparse
import pickle as pkl

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=32,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--dropout', '-do', type=float, default=0)
    parser.add_argument('--period', '-p', type=int, default=1)
    parser.add_argument('--decay_period', '-dp', type=int, default=10)
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--in_binactiv', '-ib', default='identity')
    parser.add_argument('--w_binactiv', '-wb', default='identity')
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
    train_trans = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))        
    ])
    train_data = datasets.MNIST('/media/data/MNIST', train=True,
        transform=train_trans, download=True)
    val_data = datasets.MNIST('/media/data/MNIST', train=True,
        transform=val_trans, download=True)
    train_size = len(train_data)
    split = int(0.8 * train_size)
    indices = np.arange(train_size)
    np.random.shuffle(indices)
    train_indices = indices[:split]
    val_indices = indices[split:]
    print('Number of Training Examples: ', len(train_indices))
    print('Number of Validation Examples', len(val_indices))
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_dl = DataLoader(train_data, batch_size=args.batchsize, sampler=train_sampler)
    val_dl = DataLoader(val_data, batch_size=args.batchsize, sampler=val_sampler)

    in_binactiv = binary_layers.pick_activation(args.in_binactiv)
    w_binactiv = binary_layers.pick_activation(args.w_binactiv)
    model = bitwise_mlp.BitwiseMLP(784, 10, fc_sizes=[2048, 2048, 2048],
        dropout=args.dropout, sparsity=args.sparsity,
        use_gate=args.use_gate, scale_weights=None,
        scale_activations=None, in_binactiv=in_binactiv,
        w_binactiv=w_binactiv, bn_momentum=args.bn_momentum,
        bias=False)
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
    solver = ImageRecognitionSolver(model, loss=loss, flatten=True,
        optimizer=optimizer, device=device)
    scheduler = optim.lr_scheduler.StepLR(solver.optimizer, args.decay_period,
        gamma=args.lr_decay)
    max_accuracy = 0

    for epoch in range(args.epochs):
        scheduler.step()
        total_cost = 0
        model.update_betas()
        model.train()
        train_accuracy, train_loss = solver.train(train_dl,
            clip_weights=args.clip_weights)

        if (epoch+1) % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss, train_accuracy)
            val_accuracy, val_loss = solver.eval(val_dl)
            print('Val Cost: ', val_loss, val_accuracy)
            loss_metrics.update(train_loss, train_accuracy, val_loss,
                val_accuracy, period=args.period)
            image_classification.train_plot(vis, loss_metrics, eid=None,
                win=['{} Loss'.format(args.exp), '{} Accuracy'.format(args.exp)])
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                torch.save(model.state_dict(), '../models/' + args.exp + '.model')
            for i in range(model.num_layers):
                title = 'Weight {}'.format(i)
                image_classification.plot_weights(vis,
                    model.filter_list[i].weight.data.view(-1),
                    numbins=30, title=title, win=title)
    with open('../results/' + args.exp + '.pkl', 'wb') as f:
        pkl.dump(loss_metrics, f)

if __name__ == '__main__':
    main()

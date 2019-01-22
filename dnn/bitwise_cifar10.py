import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import dnn.binary_layers as binary_layers
import datasets.quantized_data import quantized_data
import visdom
import argparse

class BitwiseBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_gate=False,
            downsample=None, in_bin=binary_layers.identity,
            weight_bin=binary_layers.identity):
        super(BitwiseBasicBlock, self).__init__()
        self.conv1 = binary_layers.BitwiseConv2d(in_channels, out_channels, 3,
        stride=stride, padding=1, in_bin=in_bin, weight_bin=weight_bin,
        use_gate=use_gate, adaptive_scaling=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = binary_layers.BitwiseConv2d(out_channels, out_channels, 3,
        stride=stride, padding=1, in_bin=in_bin, weight_bin=weight_bin,
        use_gate=use_gate, adaptive_scaling=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample=downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        super(BitwiseResnet18, self).__init__()
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(x)
        out = self.bn2(out)
        return out

class BitwiseResnet18(nn.Module):
    def __init__(self, in_bin=binary_layers.identity,
        weight_bin=binary_layers.identity, use_gate=False):
        super(BitwiseResnet18, self).__init__()
        self.in_bin = in_bin
        self.weight_bin = weight_bin
        self.use_gate = use_gate
        self.conv1 = binary_layers.BitwiseConv2d(3, 64, kernel_size=7, stride=2,
        padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self.make_layer(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1)) # convert to binary
        self.fc = BitwiseLinear(512, num_classes, use_gate=self.use_gate,
            adaptive_scaling=True, in_bin=binary_layers.clipped_ste,
            weight_bin=binary_layers.clipped_ste)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _make_layer(self, in_channels, out_channels, stride=1):
        downsample=None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                binary_layers.BitwiseConv2d(in_channels, out_channels, 1,
                    stride=stride, padding=1, in_bin=self.in_bin,
                    weight_bin=self.weight_bin, use_gate=self.use_gate,
                    adaptive_scaling=True),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(BitwiseBasicBlock(in_channels, out_channels, stride=stride,
            use_gate=self.use_gate, downsample=downsample, in_bin=self.in_bin,
            weight_bin=self.weight_bin))
        layers.append(BitwiseBasicBlock(out_channels, out_channels, use_gate=self.use_gate,
            in_bin=self.in_bin, weight_bin=self.weight_bin))
        return nn.Sequential(*layers)

    def load_pretrained_state_dict(self, state_dict):
        state = self.state_dict()
        for name, param in state_dict.items():
            if state[name].size() == param.size():
                state[name].copy_(param)

def evaluate(model, dl, optimizer=None, loss=F.mse_loss,
    device=torch.device('cpu'), dtype=torch.float, train=True):
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
    parser.add_argument('--pretrained', '-pt', action='store_true')
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--in_bin', '-ib', default='identity')
    parser.add_argument('--weight_bin', '-wb', default='identity')
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
    train_data = datasets.CIFAR10('/media/data/CIFAR10', train=True,
        transform=transforms.Compose([transforms.ToTensor(), flatten]),
        download=True)
    val_data = datasets.CIFAR10('/media/data/CIFAR10', train=False,
        transform=transforms.Compose([transforms.ToTensor(), flatten]),
        download=True)
    train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batchsize, shuffle=False)

    in_bin = binary_layers.pick_activation(args.in_bin)
    weight_bin = binary_layers.pick_activation(args.weight_bin)
    model = BitwiseResnet18(in_bin=in_bin, weight_bin=weight_bin,
        num_classes=10)

    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))
    elif args.pretrained:
        resnet18 = torchvision.resnet18(pretrained=True)
        model.load_pretrained_state_dict(resnet18)

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

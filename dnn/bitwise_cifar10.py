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
import datasets.quantized_data as quantized_data
import loss_and_metrics.image_classification as image_classification
import visdom
import argparse

class BitwiseBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_gate=False,
            downsample=None, in_binactiv=None, w_binactiv=None,
            scale_weights=None, scale_activations=None,
            bn_momentum=0.1, num_binarizations=1, dropout=0.2):
        super(BitwiseBasicBlock, self).__init__()
        self.conv1 = binary_layers.BitwiseConv2d(in_channels, out_channels, 3,
            stride=stride, padding=1, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv, use_gate=use_gate,
            scale_weights=scale_weights, scale_activations=scale_activations,
            bias=False, num_binarizations=num_binarizations)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(p=dropout)
        self.conv2 = binary_layers.BitwiseConv2d(out_channels, out_channels, 3,
            padding=1, in_binactiv=in_binactiv, w_binactiv=w_binactiv,
            use_gate=use_gate, scale_weights=scale_weights,
            scale_activations=scale_activations, bias=False,
            num_binarizations=num_binarizations)
        self.dropout2 = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.downsample=downsample

    def forward(self, x):
        identity = x
        out = self.bn1(self.dropout1(self.conv1(x)))
        out = self.bn2(self.dropout2(self.conv2(out)))

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out

class BitwiseResnet18(nn.Module):
    def __init__(self, in_binactiv=None, w_binactiv=None, use_gate=False,
        num_classes=10, scale_weights=None, scale_activations=None,
        bn_momentum=0.1, num_binarizations=1, dropout=0.2):
        super(BitwiseResnet18, self).__init__()
        self.scale_weights = scale_weights
        self.scale_activations = scale_activations
        self.in_binactiv = in_binactiv
        self.in_binfunc = None
        if in_binactiv is not None:
            self.in_binfunc = in_binactiv()
        self.w_binactiv = w_binactiv
        self.use_gate = use_gate
        self.conv1 = binary_layers.BitwiseConv2d(3, 64, kernel_size=7, stride=1,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.bn_momentum = bn_momentum
        self.dropout = dropout
        self.num_binarizations = num_binarizations
        self.scale_weights = scale_weights
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.avgpool = nn.AvgPool2d(4) # convert to binary
        self.fc = binary_layers.BitwiseLinear(512, num_classes,
            use_gate=self.use_gate, scale_weights=scale_weights,
            scale_activations=scale_activations,
            num_binarizations=num_binarizations, bias=True)
        self.scale = binary_layers.ScaleLayer(num_channels=num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.in_binfunc is not None:
            x = self.in_binfunc(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_layer(x)
        return self.scale(self.fc(x))

    def _make_layer(self, in_channels, out_channels, stride=1):
        downsample=None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                binary_layers.BitwiseConv2d(in_channels, out_channels, 1,
                    stride=stride, in_binactiv=self.in_binactiv,
                    w_binactiv=self.w_binactiv, use_gate=self.use_gate,
                    scale_weights=self.scale_weights,
                    scale_activations=self.scale_activations, bias=False,
                    num_binarizations=self.num_binarizations),
                nn.BatchNorm2d(out_channels, momentum=self.bn_momentum)
            )
        layers = []
        layers.append(BitwiseBasicBlock(in_channels, out_channels, stride=stride,
            use_gate=self.use_gate, downsample=downsample,
            in_binactiv=self.in_binactiv, w_binactiv=self.w_binactiv,
            scale_weights=self.scale_weights,
            scale_activations=self.scale_activations,
            bn_momentum=self.bn_momentum,
            dropout=self.dropout))
        layers.append(BitwiseBasicBlock(out_channels, out_channels,
            use_gate=self.use_gate, in_binactiv=self.in_binactiv,
            w_binactiv=self.w_binactiv, scale_weights=self.scale_weights,
            scale_activations=self.scale_activations,
            bn_momentum=self.bn_momentum, dropout=self.dropout))
        return nn.Sequential(*layers)

    def load_pretrained_state_dict(self, state_dict):
        state = self.state_dict()
        for name, param in state_dict.items():
            if state[name].size() == param.size():
                state[name].data.copy_(param)
                print('Loaded {}'.format(name))

def forward(model, dl, optimizer=None, loss=F.mse_loss,
    device=torch.device('cpu'), dtype=torch.float, clip_weights=False):
    running_loss = 0
    running_accuracy = 0
    for batch_idx, (data, target) in enumerate(dl):
        if optimizer:
            optimizer.zero_grad()
        data = data.to(device=device)
        target = target.to(device=device)
        estimate = model(data)
        cost = loss(estimate, target)
        correct = torch.argmax(estimate, dim=1) == target
        running_accuracy += torch.sum(correct.float()).item()
        running_loss += cost.item() * data.size(0)
        if optimizer:
            cost.backward()
            optimizer.step()
            if clip_weights:
                model.clip_weights()
    if optimizer:
        optimizer.zero_grad()
    return running_accuracy / len(dl.dataset), running_loss / len(dl.dataset)

def main():
    parser = argparse.ArgumentParser(description='bitwise network')
    parser.add_argument('--epochs', '-e', type=int, default=32,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', '-lrd', type=float, default=1.0)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0)
    parser.add_argument('--period', '-p', type=int, default=1)
    parser.add_argument('--load_file', '-lf', type=str, default=None)
    parser.add_argument('--pretrained', '-pt', action='store_true')
    parser.add_argument('--sparsity', '-sparsity', type=float, default=0)
    parser.add_argument('--dropout', '-do', type=float, default=0)
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--in_binactiv', '-ib', default='identity')
    parser.add_argument('--w_binactiv', '-wb', default='identity')
    parser.add_argument('--decay_period', '-dp', type=int, default=10)
    parser.add_argument('--clip_weights', '-cw', action='store_true')
    parser.add_argument('--bn_momentum', '-bnm', type=float, default='0.1')
    args = parser.parse_args()

    # Initialize device
    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda:'+ str(args.device))
    else:
        device = torch.device('cpu')
    print('On device: ', device)

    # Make model and dataset
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = datasets.CIFAR10('/media/data/CIFAR10', train=True,
        transform=train_transform, download=True)
    val_data = datasets.CIFAR10('/media/data/CIFAR10', train=False,
        transform=test_transform, download=True)
    train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batchsize, shuffle=False)

    vis = visdom.Visdom(port=5801)
    in_binactiv = binary_layers.pick_activation(args.in_binactiv)
    w_binactiv = binary_layers.pick_activation(args.w_binactiv)
    model = BitwiseResnet18(in_binactiv=in_binactiv, w_binactiv=w_binactiv,
        num_classes=10, scale_weights=None, scale_activations=None,
        bn_momentum=args.bn_momentum, dropout=args.dropout,
        num_binarizations=1)
    print(model)

    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))
    elif args.pretrained:
        resnet18 = models.resnet18(pretrained=True)
        model.load_pretrained_state_dict(resnet18.state_dict())

    model.to(device=device)

    # Initialize loss function
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device=device)

    # Initialize optimizer
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr,
        weight_decay=args.weight_decay)
    loss_metrics = image_classification.LossMetrics()

    for epoch in range(args.epochs):
        total_cost = 0
        model.train()
        train_accuracy, train_loss = forward(model, train_dl, optimizer, loss=loss,
            device=device)

        if (epoch+1) % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss, train_accuracy)
            model.eval()
            val_accuracy, val_loss = forward(model, val_dl, loss=loss,
                device=device, clip_weights=args.clip_weights)
            print('Val Cost: ', val_loss, val_accuracy)
            loss_metrics.update(train_loss, train_accuracy, val_loss,
                val_accuracy, period=args.period)
            image_classification.train_plot(vis, loss_metrics, eid=None,
                win=['{} Loss'.format(args.exp), '{} Accuracy'.format(args.exp)])
            torch.save(model.state_dict(), '../models/' + args.exp + '.model')

        if (epoch+1) % args.decay_period == 0 and args.lr_decay != 1:
            lr *= args.lr_decay
            optimizer = optim.Adam(model.parameters(), lr=lr,
                weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()

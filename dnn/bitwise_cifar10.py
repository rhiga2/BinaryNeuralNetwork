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
            downsample=None, binactiv=None, scale_weights=False,
            bn_momentum=0.1, num_binarizations=1):
        super(BitwiseBasicBlock, self).__init__()
        self.conv1 = binary_layers.BitwiseConv2d(in_channels, out_channels, 3,
            stride=stride, padding=1, binactiv=binactiv,
            use_gate=use_gate, scale_weights=scale_weights, bias=False,
            num_binarizations=num_binarizations)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.conv2 = binary_layers.BitwiseConv2d(out_channels, out_channels, 3,
            padding=1, binactiv=binactiv, use_gate=use_gate,
            scale_weights=scale_weights, bias=False,
            num_binarizations=num_binarizations)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.downsample=downsample
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        out = self.bn2(out)
        return out

class BitwiseResnet18(nn.Module):
    def __init__(self, binactiv=None, use_gate=False, num_classes=10,
        scale_weights=False, bn_momentum=0.1, num_binarizations=1):
        super(BitwiseResnet18, self).__init__()
        self.scale_weights = scale_weights
        self.binactiv = binactiv
        self.use_gate = use_gate
        self.conv1 = binary_layers.BitwiseConv2d(3, 64, kernel_size=7, stride=2,
        padding=3, bias=False, num_binarizations=num_binarizations)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.bn_momentum = bn_momentum
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=1)
        self.layer3 = self._make_layer(128, 256, stride=1)
        self.layer4 = self._make_layer(256, 512, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # convert to binary
        self.scale_weights = scale_weights
        self.num_binarizations = num_binarizations
        self.fc = binary_layers.BitwiseLinear(512, num_classes,
            use_gate=self.use_gate, scale_weights=scale_weights,
            binactiv=binactiv, num_binarizations=num_binarizations)
        self.scale = ScaleLayer(num_classes)

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
                    stride=stride, binactiv=self.binactiv, use_gate=self.use_gate,
                    scale_weights=self.scale_weights, bias=False,
                    num_binarizations=self.num_binarizations),
                nn.BatchNorm2d(out_channels, momentum=self.bn_momentum)
            )
        layers = []
        layers.append(BitwiseBasicBlock(in_channels, out_channels, stride=stride,
            use_gate=self.use_gate, downsample=downsample, binactiv=self.binactiv,
            scale_weights=self.scale_weights, bn_momentum=self.bn_momentum))
        layers.append(BitwiseBasicBlock(out_channels, out_channels, use_gate=self.use_gate,
            binactiv=self.binactiv, scale_weights=self.scale_weights, bn_momentum=self.bn_momentum))
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
    if optimizer:
        optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(dl):
        data = data.to(device=device)
        target = target.to(device=device)
        estimate = model(data)
        cost = loss(estimate, target)
        correct = torch.argmax(estimate, dim=1) == target
        accuracy = torch.mean(correct.float())
        running_accuracy += accuracy.item() * data.size(0)
        running_loss += cost.item() * data.size(0)
        if optimizer:
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            if clip_weights:
                model.clip_weights()
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
    parser.add_argument('--exp', '-exp', default='temp')
    parser.add_argument('--use_gate', '-ug', action='store_true')
    parser.add_argument('--binactiv', '-ba', default='identity')
    parser.add_argument('--decay_period', '-dp', type=int, default=10)
    parser.add_argument('--scale_weights', '-sw', action='store_true')
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
    trans = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = datasets.CIFAR10('/media/data/CIFAR10', train=True,
        transform=trans, download=True)
    val_size = int(0.1*len(data))
    train_size = len(data) - val_size
    train_data, val_data = torch.utils.data.random_split(data, (train_size, val_size))
    print(torch.max(train_data[0][0]), torch.min(train_data[0][0]))
    train_dl = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batchsize, shuffle=False)

    vis = visdom.Visdom(port=5801)
    binactiv = binary_layers.pick_activation(args.binactiv)
    model = BitwiseResnet18(binactiv=binactiv, num_classes=10,
        scale_weights=args.scale_weights, bn_momentum=args.bn_momentum)
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

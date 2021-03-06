import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import math
import numpy as np
import dnn.binary_layers as binary_layers
from dnn.solvers import ImageRecognitionSolver
import datasets.quantized_data as quantized_data
import loss_and_metrics.image_classification as image_classification
import visdom
import argparse
import pickle as pkl

class BitwiseBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_gate=False,
            in_binactiv=None, w_binactiv=None,
            scale_weights=None, scale_activations=None,
            bn_momentum=0.1, dropout=0.2):
        super(BitwiseBasicBlock, self).__init__()

        self.conv1 = binary_layers.BitwiseConv2d(in_channels, out_channels, 3,
            stride=stride, padding=1, in_binactiv=in_binactiv,
            w_binactiv=w_binactiv, use_gate=use_gate,
            scale_weights=scale_weights, scale_activations=scale_activations,
            bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.dropout1 = nn.Dropout(p=dropout)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=bn_momentum),
                binary_layers.BitwiseConv2d(in_channels, out_channels, 1,
                    stride=stride, in_binactiv=in_binactiv,
                    w_binactiv=w_binactiv, use_gate=use_gate,
                    scale_weights=scale_weights,
                    scale_activations=scale_activations, bias=False)
            )

        self.conv2 = binary_layers.BitwiseConv2d(out_channels, out_channels, 3,
            padding=1, in_binactiv=in_binactiv, w_binactiv=w_binactiv,
            use_gate=use_gate, scale_weights=scale_weights,
            scale_activations=scale_activations, bias=False)
        self.dropout2 = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

    def forward(self, x):
        out = self.bn1(x)
        out = self.dropout1(self.conv1(out))
        out = self.bn2(out)
        out = self.dropout2(self.conv2(out))

        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x
        return out

    def clip_weights(self):
        if self.downsample is not None:
            self.downsample[1].clip_weights()
        self.conv1.clip_weights()
        self.conv2.clip_weights()

class BitwiseResnet18(nn.Module):
    def __init__(self, in_binactiv=None, w_binactiv=None, use_gate=False,
        num_classes=10, scale_weights=None, scale_activations=None,
        bn_momentum=0.1, dropout=0.2):
        super(BitwiseResnet18, self).__init__()
        self.scale_weights = scale_weights
        self.scale_activations = scale_activations
        self.in_binactiv = in_binactiv
        self.in_binfunc = None
        if in_binactiv is not None:
            self.in_binfunc = in_binactiv()
        self.w_binactiv = w_binactiv
        self.use_gate = use_gate
        self.conv1 = nn.Conv2d(3, 64, 7,
            stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.bn_momentum = bn_momentum
        self.dropout = dropout
        self.scale_weights = scale_weights
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.avgpool = nn.AvgPool2d(4) # convert to binary
        self.fc = binary_layers.BitwiseLinear(512, num_classes,
            use_gate=self.use_gate,
            in_binactiv=self.in_binactiv, w_binactiv=self.w_binactiv,
            scale_weights=self.scale_weights,
            scale_activations=self.scale_activations)
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
        layers = []
        layers.append(BitwiseBasicBlock(in_channels, out_channels,
            stride=stride, use_gate=self.use_gate,
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

    def clip_weights(self):
        self.layer1[0].clip_weights()
        self.layer1[1].clip_weights()
        self.layer2[0].clip_weights()
        self.layer2[1].clip_weights()
        self.layer3[0].clip_weights()
        self.layer3[1].clip_weights()
        self.layer4[0].clip_weights()
        self.layer4[1].clip_weights()

class BitwiseVGG(nn.Module):
    def __init__(self, cfg, in_binactiv=None, w_binactiv=None,
        num_classes=10, scale_weights=None, scale_activations=None,
        use_gate=False, bn_momentum=0.1, dropout=0):
        super().__init__()
        self.in_binactiv = in_binactiv
        self.w_binactiv = w_binactiv
        self.scale_activations = scale_activations
        self.scale_weights = scale_weights
        self.bn_momentum = bn_momentum
        self.use_gate = use_gate
        self.features = self._make_layers(cfg)
        self.classifier = [nn.BatchNorm1d(512)]
        self.classifier.append(binary_layers.BitwiseLinear(512, 512,
            use_gate=self.use_gate, bias=False,
            in_binactiv=self.in_binactiv, w_binactiv=self.w_binactiv,
            scale_weights=self.scale_weights,
            scale_activations=self.scale_activations))
        self.classifier.append(nn.Dropout(dropout))
        self.classifier.append(nn.BatchNorm1d(512))
        self.classifier.append(binary_layers.BitwiseLinear(512, 512,
            use_gate=self.use_gate, bias=False,
            in_binactiv=self.in_binactiv, w_binactiv=self.w_binactiv,
            scale_weights=self.scale_weights,
            scale_activations=self.scale_activations))
        self.classifier.append(nn.Dropout(dropout))
        self.classifier.append(nn.BatchNorm1d(512))
        self.classifier.append(binary_layers.BitwiseLinear(512, num_classes,
            use_gate=self.use_gate, bias=True,
            in_binactiv=self.in_binactiv, w_binactiv=self.w_binactiv,
            scale_weights=self.scale_weights,
            scale_activations=self.scale_activations))
        self.classifier = nn.Sequential(*self.classifier)
        self.scale = binary_layers.ScaleLayer(num_channels=num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.scale(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if i == 0:
                    layers.append(
                        nn.Conv2d(channels, v, 3, padding=1, bias=False)
                    )
                else:
                    layers.append(nn.BatchNorm2d(channels, momentum=self.bn_momentum))
                    layers.append(
                        binary_layers.BitwiseConv2d(
                            channels, v, kernel_size=3, padding=1,
                            in_binactiv=self.in_binactiv,
                            w_binactiv=self.w_binactiv,
                            scale_activations=self.scale_activations,
                            scale_weights=self.scale_weights,
                            use_gate=self.use_gate, bias=False
                        )
                    )
                channels = v
        return nn.Sequential(*layers)

    def clip_weights(self):
        for i, layer in enumerate(self.features):
            if i != 0 and hasattr(layer, 'bitwise'):
                layer.clip_weights()


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
    parser.add_argument('--model', '-model', default='resnet18')
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
    # reintialize same data with different transform
    val_data = datasets.CIFAR10('/media/data/CIFAR10', train=True,
        transform=test_transform, download=True)
    data_size = len(train_data)
    split = int(0.9*data_size)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    train_indices = indices[:split]
    val_indices = indices[split:]
    print('Number of Training Examples: ', len(train_indices))
    print('Number of Validation Examples: ', len(val_indices))
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_dl = DataLoader(train_data, batch_size=args.batchsize, sampler=train_sampler)
    val_dl = DataLoader(val_data, batch_size=args.batchsize, sampler=val_sampler)

    vis = visdom.Visdom(port=5801)
    in_binactiv = binary_layers.pick_activation(args.in_binactiv)
    w_binactiv = binary_layers.pick_activation(args.w_binactiv)
    if args.model == 'resnet18':
        model = BitwiseResnet18(in_binactiv=in_binactiv, w_binactiv=w_binactiv,
            num_classes=10, scale_weights=None, scale_activations=None,
            bn_momentum=args.bn_momentum, dropout=args.dropout,
            use_gate=args.use_gate)
    elif args.model == 'vgg': 
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        model = BitwiseVGG(cfg, in_binactiv=in_binactiv, w_binactiv=w_binactiv,
            num_classes=10, scale_weights=None, scale_activations=None,
            bn_momentum=args.bn_momentum, use_gate=args.use_gate, dropout=args.dropout)
    print(model)

    if args.load_file:
        model.load_state_dict(torch.load('../models/' + args.load_file))
    elif args.pretrained:
        if args.model == 'resnet18':
            pretrained = models.resnet18(pretrained=True)
        elif args.model == 'vgg':
            pretrained = models.vgg16(pretrained=True)
        model.load_pretrained_state_dict(pretrained.state_dict())

    # Initialize loss function
    loss = nn.CrossEntropyLoss()

    # Initialize optimizer
    lr = args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=lr,
        weight_decay=args.weight_decay)
    solver = ImageRecognitionSolver(model, loss=loss,
        optimizer=optimizer, device=device)
    scheduler = optim.lr_scheduler.StepLR(solver.optimizer, args.decay_period,
        gamma=args.lr_decay)
    loss_metrics = image_classification.LossMetrics()
    max_accuracy = 0

    for epoch in range(args.epochs):
        scheduler.step()
        total_cost = 0
        train_accuracy, train_loss = solver.train(train_dl, clip_weights=args.clip_weights)

        if (epoch+1) % args.period == 0:
            print('Epoch %d Training Cost: ' % epoch, train_loss,
                train_accuracy)
            val_accuracy, val_loss = solver.eval(val_dl)
            print('Val Cost: ', val_loss, val_accuracy)
            loss_metrics.update(train_loss, train_accuracy, val_loss,
                val_accuracy, period=args.period)
            image_classification.train_plot(vis, loss_metrics, eid=None,
                win=['{} Loss'.format(args.exp), '{} Accuracy'.format(args.exp)])
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                torch.save(model.state_dict(), '../models/' + args.exp + '.model')

    with open('../results/' + args.exp + '.pkl', 'wb') as f:
        pkl.dump(loss_metrics, f)

if __name__ == '__main__':
    main()

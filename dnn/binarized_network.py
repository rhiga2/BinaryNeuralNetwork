import sys , os
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.quantized import *
from dnn.binary_layers import *
import argparse

class BinarizedMLP(nn.Module):
    def __init__(self, input_size, output_size, fc_sizes = [],
        dropout=0, output_activation=binarize):
        super(BinarizedMLP, self).__init__()
        self.params = {}
        self.num_layers = len(fc_sizes) + 1
        fc_sizes = fc_sizes + [output_size,]
        in_size = input_size
        self.linear_list = nn.ModuleList()
        self.batchnorm_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        self.activation = binarize
        for i, out_size in enumerate(fc_sizes):
            self.linear_list.append(BinLinear(in_size, out_size))
            in_size = out_size
            if i < self.num_layers - 1:
                self.batchnorm_list.append(nn.BatchNorm1d(out_size))
                self.dropout_list.append(nn.Dropout(dropout))
        self.output_activation = output_activation

    def forward(self, x):
        for i in range(self.num_layers):
            h = self.linear_list[i](h)
            if i < self.num_layers - 1:
                h = self.batchnorm_list[i](h)
                h = self.activation(h)
                h = self.dropout_list[i](h)
        if self.output_activation:
            h = self.output_activation(h)
        return h

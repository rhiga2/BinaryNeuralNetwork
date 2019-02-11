import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from binary_layers import *

class TestBinaryLayers(unittest.TestCase):
    def setUp(self):
        # Create dataset
        self.blinear = BitwiseLinear(3, 3, use_gate=False, scale_weights=True,
            binactiv=clipped_ste)
        linear_weight = 0.1 * torch.FloatTensor([
            [1, 2, 0],
            [3, 1, 2],
            [0, -1, 2]
        ])
        self.blinear.weight = nn.Parameter(linear_weight)
        self.blinear.bias = nn.Parameter(torch.zeros_like(self.blinear.bias.data))
        self.scale = ScaleLayer(3)
        gamma = torch.FloatTensor([10, 2, 100])
        self.scale.gamma = nn.Parameter(gamma)

    def test_nodrop(self):
        x = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [-0.4, 0.1, 0.3],
            [0.5, -0.15, 0.12]
        ])
        y_hat = drop_weights(x, gate=None, binactiv=clipped_ste, beta=0)
        self.assertTrue(torch.all(torch.eq(x, y_hat)))

    def test_drop_weights_with_beta(self):
        x = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [-0.4, 0.1, 0.3],
            [0.5, -0.15, 0.12]
        ])
        y = torch.FloatTensor([
            [0, -0.3, 0.5],
            [-0.4, 0, 0.3],
            [0.5, 0, 0]
        ])
        beta = 0.25
        y_hat = drop_weights(x, gate=None, binactiv=None, beta=beta)
        self.assertTrue(torch.all(torch.eq(y, y_hat)))

    def test_drop_weights_with_gate(self):
        x = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [-0.4, 0.1, 0.3],
            [0.5, -0.15, 0.12]
        ])
        y = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [0, 0, 0.3],
            [0.5, -0.15, 0]
        ])
        gate = torch.FloatTensor([
            [0.6, 0.34, 0.5],
            [-0.35, -0.1, 0.3],
            [0.5, 0.15, -0.12]
        ])
        y_hat = drop_weights(x, gate=gate, binactiv=clipped_ste, beta=0)
        self.assertTrue(torch.all(torch.eq(y, y_hat)))

    def test_linear_scale_weights(self):
        '''
        Compare my pytorch implementation of bss eval with Shrikant's numpy implementation
        '''
        x = torch.FloatTensor([
            [1, 2, 3],
            [4, -5, 6]
        ])
        y = 0.1 * torch.FloatTensor([
            [4, 12, 0],
            [0, 10, 10]
        ])
        y_hat = self.blinear(x)
        print(y_hat)
        self.assertTrue(torch.all(torch.eq(y, y_hat)))

    def test_no_binactiv(self):
        x = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [-0.4, 0.1, 0.3],
            [0.5, -0.15, 0.12]
        ])
        x_hat, w_hat = binarize_weights_and_inputs(x, self.blinear.weight)
        self.assertTrue(torch.all(torch.eq(x, x_hat[0])))
        self.assertTrue(torch.all(torch.eq(self.blinear.weight, w_hat)))

    def test_scale_layer(self):
        x = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [-0.4, 0.1, 0.2],
            [0.5, -0.15, 0.12]
        ])
        y = torch.FloatTensor([
            [2, -0.6, 50],
            [-4, 0.2, 20],
            [5, -0.3, 12]
        ])
        y_hat = self.scale(x)
        self.assertTrue(torch.all(torch.eq(y, y_hat)))


if __name__ == '__main__':
    unittest.main()

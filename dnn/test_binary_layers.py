import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from binary_layers import *

class TestBitwiseLinear(unittest.TestCase):
    def setUp(self):
        # Create dataset
        in_binactiv = pick_activation('clipped_ste')
        w_binactiv = pick_activation('clipped_ste')
        self.blinear = BitwiseLinear(3, 3, use_gate=False,
            in_binactiv=in_binactiv, w_binactiv=w_binactiv)
        linear_weight = 0.1 * torch.FloatTensor([
            [1, 2, 0],
            [3, 1, 2],
            [0, -1, 2]
        ])
        self.blinear.weight = nn.Parameter(linear_weight)
        self.scale = ScaleLayer(3)
        gamma = torch.FloatTensor([10, 2, 100])
        self.scale.gamma = nn.Parameter(gamma)

    def test_initalization(self):
        self.assertTrue(self.blinear.bitwise)
        self.assertTrue(self.blinear.gate is None)
        self.assertTrue(self.blinear.scale_weights is None)

    def test_drop_with_gate(self):
        w = torch.FloatTensor([
            [0.1, 0, 0],
            [0, 0, 0.2],
            [0, -0.1, 0]
        ])
        self.blinear.gate = nn.Parameter(torch.FloatTensor([
            [1, -3, 5],
            [-2, -4, 0.5],
            [6, 7.5, -0.01],
        ]))
        w_hat = self.blinear.drop_weights()
        self.assertTrue(torch.allclose(w, w_hat, rtol=0))

        # we don't want test interferring with other tests
        self.blinear.gate = None
        self.assertTrue(self.blinear.gate is None)

    def test_drop_with_beta(self):
        w = torch.FloatTensor([
            [0, 0.2, 0],
            [0.3, 0, 0.2],
            [0, 0, 0.2]
        ])
        self.blinear.beta = nn.Parameter(torch.tensor(0.15), requires_grad=False)
        w_hat = self.blinear.drop_weights()
        self.assertTrue(torch.allclose(w, w_hat, rtol=0))

        # we don't want test interferring with other tests
        self.blinear.beta = nn.Parameter(torch.tensor(0, dtype=torch.float),
            requires_grad=False)
        self.assertTrue(self.blinear.beta == 0)

    def test_nodrop(self):
        w_hat = self.blinear.drop_weights()
        self.assertTrue(torch.allclose(self.blinear.weight, w_hat, rtol=0))

    def test_binarize_inputs(self):
        x = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [-0.4, 0.1, 0.3],
            [0.5, -0.15, 0.12]
        ])
        y = torch.FloatTensor([
            [1, -1, 1],
            [-1, 1, 1],
            [1, -1, 1]
        ])
        y_hat = self.blinear.binarize_inputs(x)
        self.assertTrue(torch.allclose(y, y_hat, rtol=0))

    def test_binarize_average_inputs(self):
        self.blinear.scale_activations = 'average'
        x = torch.FloatTensor([
            [0.2, -0.3, 0.5],
            [-0.4, 0.1, 0.3],
            [0.5, -0.15, 0.12]
        ])
        y = torch.FloatTensor([
            [1/3, -1/3, 1/3],
            [-0.8/3, 0.8/3, 0.8/3],
            [0.77/3, -0.77/3, 0.77/3]
        ])
        y_hat = self.blinear.binarize_inputs(x)
        self.blinear.scale_activations = None
        self.assertTrue(torch.allclose(y, y_hat, rtol=0))

    def test_binarize_weights(self):
        w = torch.FloatTensor([
            [1, 1, 0],
            [1, 1, 1],
            [0, -1, 1]
        ])
        self.blinear.scale_weights = False
        w_hat = self.blinear.binarize_weights()
        self.assertTrue(torch.allclose(w, w_hat, rtol=0))

        self.blinear.scale_weights = True
        self.assertTrue(self.blinear.scale_weights)


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
        self.assertTrue(torch.allclose(y, y_hat, rtol=0))


if __name__ == '__main__':
    unittest.main()

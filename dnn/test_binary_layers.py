import unittest
import torch
import numpy as np
from binary_layers import *

class TestBinaryLayers(unittest.TestCase):
    def setUp(self):
        # Create dataset
        self.linear1 = BitwiseLinear(3, 3, use_gate=False, adaptive_scaling=True,
            in_bin=clipped_ste, weight_bin=clipped_ste)
        linear1_weight = torch.FloatTensor([
            [1, 2, 0],
            [3, 1, 2],
            [0, -1, 2]
        ])
        self.linear1.weight = nn.Parameter(linear1_weight)

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

    def test_linear_adaptive_scaling(self):
        '''
        Compare my pytorch implementation of bss eval with Shrikant's numpy implementation
        '''
        x = torch.FloatTensor([
            [1, 2, 3],
            [4,- 5, 6]
        ])
        y = torch.FloatTensor([
            [4, 12, 0],
            [0, 10, 10]
        ])
        y_hat = self.linear1(x)
        self.assertTrue(torch.all(torch.eq(y, y_hat)))


if __name__ == '__main__':
    unittest.main()

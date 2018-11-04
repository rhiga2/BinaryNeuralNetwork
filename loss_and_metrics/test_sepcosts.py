import unittest
import torch
from sepcosts import *

class TestSepcosts(unittest.TestCase):
    def setUp(self):
        self.dw_loss = DiscreteWasserstein(3, mode='interger')

    def test_discrete_wasserstein(self):
        x = torch.tensor(np.array([
            [0.9, 0.1, 0],
            [0.1, 0.2, 0.7]
            ]).T,
            dtype=torch.float).unsqueeze(0)
        y = torch.tensor(np.array([0, 2]),
            dtype=torch.float).unsqueeze(0)
        ans = 0.25
        estimate = self.dw_loss(x, y).item()
        print(self.assertEqual(ans, estimate))

if __name__ == '__main__':
    unittest.main()

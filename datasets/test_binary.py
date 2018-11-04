import unittest
import torch
from binary_data import *

class TestQuantize(unittest.TestCase):
    def setUp(self):
        num_bits=4
        self.quantizer = Quantizer(0, 1, num_bits=num_bits)
        self.disperser = Disperser(num_bits)
        self.accumulator = Accumulator(num_bits)
        self.one_hot_transform = OneHotTransform(num_bits)

    def test_quantize(self):
        x = torch.tensor(np.array([1.1, 2.1, 5.5, 15.2, -43]), dtype=torch.float)
        x = x.unsqueeze(0)
        ans = torch.tensor(np.array([2, 3, 6, 15, 0]), dtype=torch.float)
        estimate = self.quantizer(x).squeeze(0)
        self.assertTrue(torch.equal(estimate, ans))

    def test_one_hot(self):
        x = torch.tensor(np.array([2, 3, 6, 15, 0]), dtype=torch.float)
        x = x.unsqueeze(0)
        ans = torch.tensor(
            np.array([
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0]
            ]),
            dtype=torch.float
        )
        estimate = self.one_hot_transform(x).squeeze(0)
        self.assertTrue(torch.equal(estimate, ans))

    def test_disperser(self):
        x = torch.tensor(np.array([2, 3, 6, 15, 0]), dtype=torch.float)
        x = x.unsqueeze(0)
        ans = torch.tensor(
            np.array([
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 0]
            ]),
            dtype=torch.float
        )
        estimate = self.disperser(x).squeeze(0)
        self.assertTrue(torch.equal(estimate, ans))

    def test_accumulator(self):
        x = torch.tensor(
            np.array([
                [-1, 1, -1, 1, -1],
                [1, 1, 1, 1, -1],
                [-1, -1, 1, 1, -1],
                [-1, -1, -1, 1, -1]]),
            dtype=torch.float32
        ).unsqueeze(0)
        pass

    def test_bucketize(self):
        x = torch.FloatTensor([0.5, 3.5, 2.4, 1.9, 4.2, 3.1, 1.5])
        bins = torch.FloatTensor([1.8, 2, 3, 4])
        soln = torch.LongTensor([0, 3, 2, 1, 4, 3, 0])
        bucket_x = bucketize(x, bins)
        all_match = torch.all(torch.eq(bucket_x, soln))
        self.assertTrue(all_match)

if __name__ == '__main__':
    unittest.main()

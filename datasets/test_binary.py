import unittest
import torch
from binary_data import *

class TestQuantize(unittest.TestCase):
    def setUp(self):
        self.quantizer = QuantizeDisperser(0, 1, num_bits=4, dtype=torch.uint8)
        self.dequantizer = DequantizeAccumulator(0, 1, num_bits=4)
        self.disperser = Disperser(4, 1)

    def test_qad_shape(self):
        torch.manual_seed(0)
        x = torch.rand((10, 5, 10))
        ans = torch.Size([10, 4, 5, 10])
        estimate = self.quantizer(x)
        self.assertEqual(ans, estimate.size())

    def test_quantize_and_disperse(self):
        x = torch.tensor(np.array([1.1, 2.1, 5.5, 15.2, -43]), dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(2)
        ans = torch.tensor(
            np.array([
                [-1, 1, -1, 1, -1],
                [1, 1, 1, 1, -1],
                [-1, -1, 1, 1, -1],
                [-1, -1, -1, 1, -1]]),
            dtype=torch.float
        )
        estimate = self.quantizer(x).squeeze(0).squeeze(2)
        self.assertTrue(torch.equal(estimate, ans))

    def test_dequantize_and_accumulate(self):
        x = torch.tensor(
            np.array([
                [-1, 1, -1, 1, -1],
                [1, 1, 1, 1, -1],
                [-1, -1, 1, 1, -1],
                [-1, -1, -1, 1, -1]]),
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(3)
        ans = torch.tensor(np.array([1.5, 2.5, 5.5, 14.5, -0.5]), dtype=torch.float32)
        estimate = self.dequantizer(x).squeeze(0).squeeze(1)
        self.assertTrue(torch.equal(estimate, ans))

    def test_bucketize(self):
        x = torch.FloatTensor([0.5, 3.5, 2.4, 1.9, 4.2, 3.1, 1.5])
        bins = torch.FloatTensor([1.8, 2, 3, 4])
        soln = torch.LongTensor([0, 3, 2, 1, 4, 3, 0])
        bucket_x = bucketize(x, bins)
        all_match = torch.all(torch.eq(bucket_x, soln))
        self.assertTrue(all_match)

if __name__ == '__main__':
    unittest.main()

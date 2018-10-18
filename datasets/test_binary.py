import unittest
import torch
from binary_data import *

class TestQuantize(unittest.TestCase):
    def test_binarize(self):
        x = np.array([[1.1, 2.1, 1.5],
                     [3.4, 5.5, 6.9]])
        bins = np.array(list(range(15)))
        ans = np.array([[0, 0, 0],
                        [0, 0, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                        [0, 0, 0],
                        [1, 1, 1],
                        [0, 1, 1],
                        [0, 0, 1]], dtype=np.uint8)
        estimate = binarize(x, bins)
        np.testing.assert_equal(estimate, ans)

    def test_kmeans_quantizer(self):
        np.random.seed(0)
        x = np.random.normal(size=(100, 100))
        xmax = np.max(x)
        xmin = np.min(x)
        flatten_x = x.reshape(-1)
        kcenters, kbins = kmeans_qlevels(flatten_x)
        kquantized = quantize(x, kbins, kcenters)
        kerror = np.mean(np.abs(kquantized - x))
        print()
        print('KMeans Max Quantizer Error: ', kerror)
        self.assertLess(kerror, (xmax - xmin)/32)

    def test_uniform_quantizers(self):
        np.random.seed(0)
        x = np.random.normal(size=(100, 100))
        xmax = np.max(x)
        xmin = np.min(x)
        flatten_x = x.reshape(-1)
        ucenters, ubins = uniform_qlevels(flatten_x)
        uquantized = quantize(x, ubins, ucenters)
        uerror = np.mean(np.abs(uquantized - x))
        print()
        print('Uniform Quantizer Error: ', uerror)
        self.assertLess(uerror, (xmax - xmin)/32)

    def test_bucketize(self):
        x = torch.FloatTensor([0.5, 3.5, 2.4, 1.9, 4.2, 3.1, 1.5])
        bins = torch.FloatTensor([1.8, 2, 3, 4])
        soln = torch.ByteTensor([0, 3, 2, 1, 4, 3, 0])
        bucket_x = bucketize(x, bins)
        all_match = torch.all(torch.eq(bucket_x, soln))
        self.assertTrue(all_match)

if __name__ == '__main__':
    unittest.main()

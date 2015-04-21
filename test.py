import unittest
from numpy import *
from kNN import kNN


class TestApplication(unittest.TestCase):
    def test_kNN(self):
        # prepare data
        k = 3
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        self.assertEqual(kNN.computeKNN([1.0, 1.0], group, labels, k), 'A')
        self.assertEqual(kNN.computeKNN([0, 0], group, labels, k), 'B')

if __name__ == '__main__':
    unittest.main()
